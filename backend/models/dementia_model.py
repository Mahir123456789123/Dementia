import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2Model
from peft import LoraConfig, get_peft_model

# --- Compatibility patch from notebook ---
from transformers import Wav2Vec2Model as _Wav2Vec2Model
_Wav2Vec2Model.get_input_embeddings = lambda self: self.feature_extractor

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, hidden_states, attention_mask=None):
        scores = self.score(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)

def masked_mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

class AudioDownsample(nn.Module):
    def __init__(self, dim=768, stride=6, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.act(self.norm(out))

class FFNResidual(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class MultimodalMTLDementiaDetector(nn.Module):
    def __init__(self, bert_r=8, wav2vec_r=8, unfreeze_last_n=1):
        super(MultimodalMTLDementiaDetector, self).__init__()

        base_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        base_wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        base_wav2vec.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base_bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if bert_r > 0:
            bert_lora_config = LoraConfig(
                r=bert_r, lora_alpha=bert_r * 2,
                target_modules=["query", "value"],
                lora_dropout=0.1, bias="none"
            )
            self.bert = get_peft_model(base_bert, bert_lora_config)
        else:
            self.bert = base_bert
            for param in self.bert.parameters():
                param.requires_grad = False

        if wav2vec_r > 0:
            wav2vec_lora_config = LoraConfig(
                r=wav2vec_r, lora_alpha=wav2vec_r * 2,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                lora_dropout=0.1, bias="none"
            )
            self.wav2vec = get_peft_model(base_wav2vec, wav2vec_lora_config)
            self.wav2vec.enable_input_require_grads()
        else:
            self.wav2vec = base_wav2vec
            for param in self.wav2vec.parameters():
                param.requires_grad = False
            base_wav2vec.enable_input_require_grads()

        if unfreeze_last_n > 0:
            self._unfreeze_last_layers(self.bert, unfreeze_last_n, kind="bert")
            self._unfreeze_last_layers(self.wav2vec, unfreeze_last_n, kind="wav2vec")

        self.text_proj = nn.Linear(768, 256)
        self.audio_proj = nn.Linear(768, 256)
        self.audio_attn_pool = AttentionPool(768)
        self.audio_downsample = AudioDownsample(dim=768, stride=6, kernel_size=9)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(256)
        self.fusion_ffn = FFNResidual(256, expansion=4, dropout=0.1)

        self.mmse_features = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU()
        )
        self.mmse_head = nn.Linear(32, 1)

        self.diagnosis_branch = nn.Sequential(
            nn.Linear(289, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.severity_branch = nn.Sequential(
            nn.Linear(289, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )

        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    @staticmethod
    def _unfreeze_last_layers(module, n, kind):
        base = module
        if hasattr(base, "base_model"):
            base = base.base_model
        if hasattr(base, "model"):
            base = base.model
        try:
            if kind == "bert":
                layers = base.encoder.layer
            else:
                layers = base.encoder.layers
        except AttributeError:
            return
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward_audio_features(self, audio_waveforms):
        # Used for occlusion caching
        w2v_out = self.wav2vec(audio_waveforms).last_hidden_state  
        w2v_out = self.audio_downsample(w2v_out)                    
        return w2v_out
        
    def forward_from_features(self, input_ids, attention_mask, w2v_out, calibrate=False):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = bert_out.last_hidden_state  
        
        text_seq_proj = self.text_proj(text_seq)     
        audio_seq_proj = self.audio_proj(w2v_out)     

        text_key_padding_mask = (attention_mask == 0)
        attn_out, _ = self.cross_attn(
            query=audio_seq_proj, key=text_seq_proj, value=text_seq_proj,
            key_padding_mask=text_key_padding_mask
        )

        fused_seq = self.fusion_norm(attn_out + audio_seq_proj)
        fused_seq = self.fusion_ffn(fused_seq)
        fused = fused_seq.mean(dim=1)  

        mmse_hidden = self.mmse_features(fused)
        mmse_preds = self.mmse_head(mmse_hidden)          

        mmse_signal = mmse_preds / 30.0                    
        fused_infused = torch.cat([fused, mmse_hidden, mmse_signal], dim=-1)

        diag_logits = self.diagnosis_branch(fused_infused)
        severity_logits = self.severity_branch(fused_infused)

        if calibrate:
            diag_logits = diag_logits / self.temperature

        return diag_logits, mmse_preds, severity_logits

    def forward(self, input_ids, attention_mask, audio_waveforms, calibrate=False):
        w2v_out = self.forward_audio_features(audio_waveforms)
        return self.forward_from_features(input_ids, attention_mask, w2v_out, calibrate)
