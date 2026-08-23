# MODEL INTEGRATION ANALYSIS

Based on the inspection of `Major_Dementia.ipynb`, here is the breakdown of the model architecture, configuration, and any discrepancies found.

## 1. Final Model Architecture
- **Class**: `MultimodalMTLDementiaDetector`
- **Text Encoder**: `emilyalsentzer/Bio_ClinicalBERT` (LoRA rank=8, last layer unfrozen)
- **Audio Encoder**: `facebook/wav2vec2-base` (LoRA rank=8, last layer unfrozen)
- **Audio Downsampler**: `AudioDownsample` (Conv1d with stride=6, kernel_size=9)
- **Cross-Modal Fusion**: `MultiheadAttention` (4 heads)
- **Outputs**:
  - `diag_logits` (Binary dementia probability via sigmoid)
  - `mmse_preds` (Regression head, raw values to be clamped 0-30)
  - `severity_logits` (4-class cross-entropy)

## 2. Checkpoint Path
- **Path**: `./kfold_checkpoints/dementia_mtl_FINAL.pth`
- The checkpoint contains `state_dict` and `threshold`.

## 3. Dependencies
- `torch`, `torchaudio`
- `transformers`
- `peft`
- `librosa`
- `numpy`, `pandas`, `scikit-learn`
- **`TODO — VERIFY AGAINST NOTEBOOK`**: `CrisperWhisper` is missing from the notebook's environment setup and code.

## 4. Transcription Configuration
- **`TODO — VERIFY AGAINST NOTEBOOK`**: `TranscriptionEngine` and CrisperWhisper logic (mode="verbatim", word_timestamps=True) are NOT present in the provided notebook. I will need the exact implementation or I must stub it based on the prompt's description.

## 5. Audio Preprocessing
- **Sampling Rate**: 16,000 Hz (loaded via `librosa.load(audio_path, sr=16000)`)
- **Max Audio Samples Discrepancy**: The prompt mentions `MAX_SAMPLES = 16000 * 30`, but the notebook contains `MAX_AUDIO_SAMPLES = 16000 * 75`. **`TODO — VERIFY AGAINST NOTEBOOK`**: Which one to use? I will default to `16000 * 30` per the prompt's explicit instruction, but note this discrepancy.
- Passed through `Wav2Vec2Processor`.

## 6. Tokenizer Configuration
- **Model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Max Length**: 512
- **Padding**: `max_length`
- **Truncation**: `True`

## 7. Model Input Shapes
- `input_ids`: `(Batch, 512)`
- `attention_mask`: `(Batch, 512)`
- `audio_waveforms`: `(Batch, Max_Samples)`

## 8. Output Semantics
- **Dementia Probability**: `sigmoid(diag_logits / temperature)`. Note: Calibration temperature is applied at inference.
- **MMSE**: Estimated MMSE from `mmse_preds`, clipped to `[0, 30]`.
- **Severity**: Derived from `severity_logits` (0=Normal, 1=Mild, 2=Moderate, 3=Severe).

## 9. Threshold Behavior
- Loaded from checkpoint `checkpoint['threshold']`.
- Fallback is 0.5.

## 10. Explanation/Occlusion Behavior
- **`TODO — VERIFY AGAINST NOTEBOOK`**: `get_occlusion_importance()`, `ClinicalExplanationGenerator`, `SpeechBiomarkerExtractor`, `ClinicalEvidenceScorer`, and `ClinicalReportGenerator` are entirely missing from the provided notebook code. These need to be provided or I will have to implement them based strictly on the prompt's description without exact notebook reference.

## 11. Required Model Files
- `./kfold_checkpoints/dementia_mtl_FINAL.pth`
- The `MultimodalMTLDementiaDetector` class definition and its dependencies (`AttentionPool`, `AudioDownsample`, `FFNResidual`).
