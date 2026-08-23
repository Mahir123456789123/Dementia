import torch
import numpy as np
from backend.services.model_loader import model_manager
from backend.config import MOCK_ANALYSIS_MODE, MAX_AUDIO_SAMPLES

def get_severity_label(severity_idx):
    mapping = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    return mapping.get(severity_idx, "Unknown")

class InferenceEngine:
    def __init__(self):
        self.negation_words = {"not", "no", "never", "don't", "doesn't", "didn't", "can't", "won't", "couldn't"}
        
    def run_single_inference(self, transcript_text, audio_array):
        if MOCK_ANALYSIS_MODE:
            return {
                "dementia_probability": 0.73,
                "predicted_label": "Dementia",
                "threshold": 0.50,
                "confidence": 0.46,
                "mmse_prediction": 21.4,
                "severity_prediction": "Mild",
                "attribution": [
                    {"word": "falling", "delta": 0.084},
                    {"word": "um", "delta": 0.052}
                ]
            }
            
        model = model_manager.model
        tokenizer = model_manager.tokenizer
        processor = model_manager.audio_processor
        device = model_manager.device
        threshold = model_manager.threshold
        
        # 1. Process audio
        if len(audio_array) > MAX_AUDIO_SAMPLES:
            audio_array = audio_array[:MAX_AUDIO_SAMPLES]
            
        audio_inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        audio_tensor = audio_inputs.input_values.to(device)
        
        # 2. Process text
        text_encoding = tokenizer(
            transcript_text, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = text_encoding['input_ids'].to(device)
        attention_mask = text_encoding['attention_mask'].to(device)
        
        # 3. Base inference
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                w2v_out = model.forward_audio_features(audio_tensor)
                diag_logits, mmse_preds, severity_logits = model.forward_from_features(
                    input_ids, attention_mask, w2v_out, calibrate=True
                )
                
            base_prob = torch.sigmoid(diag_logits).item()
            predicted_label = "Dementia" if base_prob > threshold else "Control"
            
            # Confidence logic from prompt
            confidence = abs(base_prob - threshold) / max(threshold, 1.0 - threshold)
            confidence = min(1.0, float(confidence))
            
            # MMSE clamping
            mmse_prediction = max(0.0, min(30.0, float(mmse_preds.item())))
            
            # Severity mapping
            severity_idx = torch.argmax(severity_logits).item()
            severity_prediction = get_severity_label(severity_idx)
            
        # 4. Occlusion attribution
        attribution = self.get_occlusion_importance(
            transcript_text, 
            audio_tensor, 
            w2v_out,
            base_prob
        )
        
        # Cleanup
        del audio_tensor
        del input_ids
        del attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            "dementia_probability": round(base_prob, 4),
            "predicted_label": predicted_label,
            "threshold": round(threshold, 4),
            "confidence": round(confidence, 4),
            "mmse_prediction": round(mmse_prediction, 1),
            "severity_prediction": severity_prediction,
            "attribution": attribution
        }

    def get_occlusion_importance(self, text, audio_tensor, w2v_out, base_prob, max_passes=40):
        words = text.split()
        if not words:
            return []
            
        blocks = []
        i = 0
        while i < len(words):
            word = words[i].lower()
            clean_word = "".join(c for c in word if c.isalpha() or c == "'")
            
            if clean_word in self.negation_words and i + 1 < len(words):
                blocks.append({
                    "indices": [i, i+1],
                    "text": f"{words[i]} {words[i+1]}"
                })
                i += 2
            else:
                blocks.append({
                    "indices": [i],
                    "text": words[i]
                })
                i += 1
                
        # Limit blocks to max_passes
        if len(blocks) > max_passes:
            blocks = blocks[:max_passes]
            
        model = model_manager.model
        tokenizer = model_manager.tokenizer
        device = model_manager.device
        
        results = []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                for block in blocks:
                    # Replace block with blanks
                    occluded_words = list(words)
                    for idx in block["indices"]:
                        occluded_words[idx] = "[UNK]"
                        
                    occluded_text = " ".join(occluded_words)
                    
                    text_encoding = tokenizer(
                        occluded_text, max_length=512, padding='max_length',
                        truncation=True, return_tensors='pt'
                    )
                    input_ids = text_encoding['input_ids'].to(device)
                    attention_mask = text_encoding['attention_mask'].to(device)
                    
                    diag_logits, _, _ = model.forward_from_features(
                        input_ids, attention_mask, w2v_out, calibrate=True
                    )
                    
                    occ_prob = torch.sigmoid(diag_logits).item()
                    delta = base_prob - occ_prob
                    
                    results.append({
                        "word": block["text"],
                        "delta": round(delta, 4)
                    })
                    
        # Sort by delta (highest positive delta means word strongly pushed toward dementia)
        results.sort(key=lambda x: x["delta"], reverse=True)
        return results[:10]  # Return top 10 influencers

inference_engine = InferenceEngine()
