# ============================================================
# ENVIRONMENT SETUP — run once, then restart the kernel
# ============================================================
import subprocess, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("STDERR:\n", result.stderr[-3000:])
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result

# 1. PyTorch with CUDA 12.1 build (Windows-native wheel)
run(f'"{sys.executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
run(f'"{sys.executable}" -m pip uninstall -y torchao')
run(f'"{sys.executable}" -m pip install "transformers==4.46.0" "tokenizers<0.21" "huggingface_hub<0.26"')
# 2. Rest of the notebook's dependencies
run(f'"{sys.executable}" -m pip install "torchao>=0.16.0" peft pylangacq '
    f'librosa soundfile scikit-learn pandas numpy tqdm matplotlib seaborn '
    f'sentencepiece tiktoken')

# 3. Verify GPU is actually visible to PyTorch
import torch
print("\n" + "="*50)
print("torch version   :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name        :", torch.cuda.get_device_name(0))
    print("VRAM total (GB) :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
else:
    print("⚠️  CUDA not available — check your NVIDIA driver (run `nvidia-smi` in a regular terminal).")
print("="*50)

# 4. Reduce memory fragmentation on a 6GB card
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import os
DATA_ROOT = "./"   # <- point this at your local folder

import os
import glob
import csv

def parse_chat_files(folder_path, group_label):
    extracted_data = []
    valid_mmse_scores = []

    print(f"Reading files in {folder_path}...")
    search_pattern = os.path.join(folder_path, '*.cha')

    for file_path in glob.glob(search_pattern):
        # STRICT RULE: ID defaults to UNKNOWN and is only changed by @Media
        file_id, age, gender, mmse = 'UNKNOWN', 'NA', 'NA', 'NA'

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:

                # Stop reading the header the moment dialogue begins
                if line.startswith('*'):
                    break

                # 1. STRICTLY extract ID from the @Media line
                if line.lower().startswith('@media:'):
                    # Gets "S003, audio" -> strips whitespace -> splits by comma -> takes "S003"
                    content = line.split(':', 1)[1].strip()
                    file_id = content.split(',')[0].strip()

                # 2. Extract metadata from the @ID line
                elif line.startswith('@ID:'):
                    content = line.split(':', 1)[1].strip()
                    parts = content.split('|')

                    # Make sure it's the PAR (Participant) line
                    if len(parts) > 2 and parts[2] == 'PAR':
                        raw_age = parts[3].strip()

                        # Split by ';' to fix the 70;00. bug
                        if ';' in raw_age:
                            age = raw_age.split(';')[0]
                        else:
                            age = raw_age

                        age = age.replace('.', '').strip()
                        gender = parts[4].strip()

                        # Extract MMSE safely
                        if len(parts) > 8 and parts[8].strip() != '':
                            mmse = parts[8].strip()
                            try:
                                valid_mmse_scores.append(float(mmse))
                            except ValueError:
                                pass

        extracted_data.append({
            'ID': file_id,
            'age': age,
            'gender': gender,
            'mmse': mmse,
            'group': group_label
        })

    return extracted_data, valid_mmse_scores

def create_metadata_csv(cc_folder, cd_folder, output_csv):
    # 1. Parse both folders
    cc_data, cc_mmse_scores = parse_chat_files(cc_folder, 'cc')
    cd_data, cd_mmse_scores = parse_chat_files(cd_folder, 'cd')

    # 2. Calculate the averages
    cc_avg = sum(cc_mmse_scores) / len(cc_mmse_scores) if cc_mmse_scores else 0
    cd_avg = sum(cd_mmse_scores) / len(cd_mmse_scores) if cd_mmse_scores else 0

    cc_avg = round(cc_avg, 1)
    cd_avg = round(cd_avg, 1)

    print(f"\nCalculated Averages:")
    print(f"Control (cc) Average MMSE: {cc_avg}")
    print(f"Clinical (cd) Average MMSE: {cd_avg}\n")

    # 3. Combine data and impute missing values
    all_data = cc_data + cd_data

    for row in all_data:
        if row['mmse'] == 'NA' or row['mmse'] == '':
            row['mmse'] = cc_avg if row['group'] == 'cc' else cd_avg

    # 4. Write everything to the CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'age', 'gender', 'mmse'])

        for row in all_data:
            writer.writerow([row['ID'], row['age'], row['gender'], row['mmse']])

    print(f"Done! Combined metadata saved to: {output_csv}")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    cc_folder_path = os.path.join(DATA_ROOT, 'train', 'transcription', 'cc')
    cd_folder_path = os.path.join(DATA_ROOT, 'train', 'transcription', 'cd')

    output_filename = 'combined_metadata.csv'

    create_metadata_csv(cc_folder_path, cd_folder_path, output_filename)

import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

WAV_ROOT = os.path.join(DATA_ROOT, "train", "Full_wave_enhanced_audio")
durations = []
file_info = []

wav_files = []

for root, _, files in os.walk(WAV_ROOT):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_files.append(os.path.join(root, f))

print(f"Found {len(wav_files)} wav files")

for wav_path in tqdm(wav_files):
    try:
        info = sf.info(wav_path)
        duration = info.frames / info.samplerate

        durations.append(duration)
        file_info.append((wav_path, duration))

    except Exception as e:
        print(f"Failed: {wav_path}")
        print(e)

durations = np.array(durations)

print("\n===== AUDIO LENGTH STATISTICS =====")
print(f"Files               : {len(durations)}")
print(f"Mean length         : {np.mean(durations):.2f} sec")
print(f"Median length       : {np.median(durations):.2f} sec")
print(f"Min length          : {np.min(durations):.2f} sec")
print(f"Max length          : {np.max(durations):.2f} sec")
print(f"95th percentile     : {np.percentile(durations,95):.2f} sec")
print(f"99th percentile     : {np.percentile(durations,99):.2f} sec")

print("\n===== TOP 20 LONGEST FILES =====")

for path, dur in sorted(file_info, key=lambda x: x[1], reverse=True)[:20]:
    print(f"{dur:8.2f}s  {os.path.basename(path)}")

# !pip install peft

# !pip install pylangacq -q

# !pip install --upgrade torchao>=0.16.0 peft transformers

# !pip install transformers torchaudio librosa scikit-learn -q

import os
import re
import pandas as pd

def parse_cha_file(filepath):
    """Extract participant speech lines from a .cha CHAT transcript file."""
    participant_text = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Lines starting with *PAR: are the patient's speech
                if line.startswith('*PAR:'):
                    # Strip the speaker tag and clean up
                    speech = re.sub(r'\*PAR:\s*', '', line)
                    # Remove CHAT annotations like [: word], &word, <>, etc.
                    speech = re.sub(r'\[.*?\]', '', speech)
                    speech = re.sub(r'&\w+', '', speech)
                    speech = re.sub(r'<|>', '', speech)
                    speech = re.sub(r'\x15.*?\x15', '', speech)  # remove time codes
                    speech = speech.strip()
                    if speech:
                        participant_text.append(speech)
    except Exception as e:
        print(f"[WARN] Could not parse {filepath}: {e}")

    return ' '.join(participant_text)


# --- Build transcript dataframe from all .cha files ---
CHA_CC_PATH = os.path.join(DATA_ROOT, 'train', 'transcription', 'cc')
CHA_CD_PATH = os.path.join(DATA_ROOT, 'train', 'transcription', 'cd')

records = []

for folder, label in [(CHA_CC_PATH, 0), (CHA_CD_PATH, 1)]:
    for filename in os.listdir(folder):
        if filename.endswith('.cha'):
            subject_id = filename.replace('.cha', '').strip()
            filepath = os.path.join(folder, filename)
            transcript = parse_cha_file(filepath)
            records.append({
                'subject_id': subject_id,
                'text': transcript,
                'label': label
            })

master_df = pd.DataFrame(records)
print(f"Loaded {len(master_df)} transcripts")
print(f"Empty transcripts: {(master_df['text'] == '').sum()}")
display(master_df.head())

# Make sure a transcript looks real
print(master_df.iloc[0]['subject_id'])
print(master_df.iloc[0]['text'][:300])

import pandas as pd

# --- 1. Load the Combined Metadata File ---
# Update this path to point to your newly generated combined_metadata.csv
combined_meta_path = os.path.join(DATA_ROOT, 'train', 'combined_metadata.csv')
# Read the CSV (our extraction script used standard commas, so we don't need sep=';' anymore)
combined_meta_df = pd.read_csv(combined_meta_path)

# --- 2. Clean the Combined Data ---
combined_meta_df.columns = combined_meta_df.columns.str.strip()
combined_meta_df['ID'] = combined_meta_df['ID'].astype(str).str.strip()
combined_meta_df['mmse'] = pd.to_numeric(combined_meta_df['mmse'], errors='coerce')

print(f"Total subjects loaded: {len(combined_meta_df)}")
print(f"Missing MMSE scores in metadata: {combined_meta_df['mmse'].isna().sum()}")

# --- 3. Merge with your Main Audio/Text DataFrame ---
if 'mmse' in master_df.columns:
    master_df = master_df.drop(columns=['mmse'])

# Ensure the join key is clean in both dataframes
master_df['subject_id'] = master_df['subject_id'].astype(str).str.strip()

master_df = pd.merge(
    master_df,
    combined_meta_df[['ID', 'mmse']],
    left_on='subject_id',
    right_on='ID',
    how='left'
)

# --- 4. Final Cleanup ---
master_df.drop(columns=['ID'], inplace=True, errors='ignore')

# Fill missing values with -1.0 for the Masked Loss function
# Note: If the earlier script already imputed group averages, this will only
# catch transcripts that didn't match an ID in the metadata at all.
master_df['mmse'] = master_df['mmse'].fillna(-1.0)

print("Merge complete! Successfully updated master_df with MMSE scores.")
display(master_df[['subject_id', 'mmse']].head())

# !pip install sentencepiece tiktoken

run(f'"{sys.executable}" -m pip uninstall -y torchao')

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_HOLDOUT_FRAC = 0.10

subject_labels_full = master_df.groupby('subject_id')['label'].first()
subject_ids_arr = subject_labels_full.index.to_numpy()
subject_labels_arr = subject_labels_full.to_numpy()

trainval_subjects, test_subjects = train_test_split(
    subject_ids_arr, test_size=TEST_HOLDOUT_FRAC,
    stratify=subject_labels_arr, random_state=RANDOM_SEED
)

test_df = master_df[master_df['subject_id'].isin(test_subjects)].reset_index(drop=True)
overlap_check = set(master_df[master_df['subject_id'].isin(trainval_subjects)]['subject_id']) & set(test_df['subject_id'])
assert not overlap_check, f"Patient leakage into test set: {overlap_check}"

print(f"Test set: {len(test_df)} rows / {len(test_subjects)} patients")
print(f"Train+Val pool: {len(master_df) - len(test_df)} rows / {len(trainval_subjects)} patients")

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model, Wav2Vec2Processor
from peft import LoraConfig, get_peft_model
import librosa
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, classification_report, mean_squared_error,
                              balanced_accuracy_score, f1_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix)
from tqdm import tqdm
import copy
# --- Compatibility patch: Wav2Vec2 has no embeddings, but newer peft
# assumes every model does when preparing gradient checkpointing ---
from transformers import Wav2Vec2Model as _Wav2Vec2Model
_Wav2Vec2Model.get_input_embeddings = lambda self: self.feature_extractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Firing up Hierarchical Clinical-Reasoning MTL Architecture on: {device} ---")

MMSE_MISSING = -1.0
MMSE_MAX = 30.0

# Severity buckets derived from MMSE (standard clinical cutoffs)
# 0=Normal(>=24) 1=Mild(18-23) 2=Moderate(10-17) 3=Severe(<10)
def mmse_to_severity(mmse):
    if mmse >= 24:
        return 0
    elif mmse >= 18:
        return 1
    elif mmse >= 10:
        return 2
    else:
        return 3


# ============================================================
# TEXT CLEANING
# ============================================================
def clean_transcript(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r'\*\w+:', ' ', t)
    t = re.sub(r'&-\w+', ' ', t)
    t = re.sub(r'\[/+\]', ' ', t)
    t = re.sub(r'\[.*?\]', ' ', t)
    t = re.sub(r'\bxxx\b|\byyy\b', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\(\.+\)', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# ============================================================
# DATASET
# ============================================================
class Wav2VecMultimodalDataset(Dataset):
    def __init__(self, df, wav_base_path, verbose_name=""):
        self.data = df.reset_index(drop=True)
        self.wav_base_path = wav_base_path
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", use_fast=False
        )
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self._audit_text_sources(verbose_name)

    def _audit_text_sources(self, name):
        has_text_col = 'text' in self.data.columns
        has_transcript_col = 'transcript' in self.data.columns
        n_real, n_fallback, n_over_len = 0, 0, 0
        col = 'text' if has_text_col else ('transcript' if has_transcript_col else None)
        for i in range(len(self.data)):
            raw = self.data.iloc[i][col] if col else None
            if col and isinstance(raw, str) and raw.strip() and raw.strip().lower() != 'nan':
                n_real += 1
                n_tokens = len(self.text_tokenizer.tokenize(clean_transcript(raw)))
                if n_tokens > 512:
                    n_over_len += 1
            else:
                n_fallback += 1
        print(f"[AUDIT:{name}] text_col={has_text_col} transcript_col={has_transcript_col} "
              f"| real_transcripts={n_real} | synthetic_fallback={n_fallback} "
              f"| truncated_over_512={n_over_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        subject_id = str(row['subject_id']).strip()
        label = float(row['label'])

        raw_mmse = row.get('mmse', MMSE_MISSING)
        if pd.isna(raw_mmse):
            raw_mmse = MMSE_MISSING
        mmse = float(raw_mmse)
        severity = mmse_to_severity(mmse) if mmse != MMSE_MISSING else -1

        raw_text = row.get('text', None)
        if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)):
            raw_text = row.get('transcript', None)

        if raw_text is not None and isinstance(raw_text, str) and raw_text.strip():
            text = clean_transcript(raw_text)
            if not text:
                text = None
        else:
            text = None

        if text is None:
            mlu = row.get('mlu', 0)
            noun_count = row.get('noun_count', 0)
            verb_count = row.get('verb_count', 0)
            text = (
                f"Patient linguistic profile: Mean length of utterance is {mlu}. "
                f"Noun count is {noun_count}. Verb count is {verb_count}."
            )

        text_encoding = self.text_tokenizer(
            text, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        folder = "cc" if label == 0 else "cd"
        audio_path = os.path.join(self.wav_base_path, folder, f"{subject_id}.wav")

        try:
            speech_array, sr = librosa.load(audio_path, sr=16000)
            MAX_AUDIO_SAMPLES = 16000 * 75
            if len(speech_array) > MAX_AUDIO_SAMPLES:
                speech_array = speech_array[:MAX_AUDIO_SAMPLES]
            audio_inputs = self.audio_processor(
                speech_array, sampling_rate=16000, return_tensors="pt"
            )
            audio_tensor = audio_inputs.input_values.squeeze(0)
        except Exception as e:
            print(f"[WARN] Audio load failed for {subject_id}: {e}")
            audio_tensor = torch.zeros(16000)

        return {
            'subject_id': subject_id,
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'audio_waveforms': audio_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'mmse': torch.tensor(mmse, dtype=torch.float32),
            'severity': torch.tensor(severity, dtype=torch.long),
        }


# ============================================================
# SMALL BUILDING BLOCKS
# ============================================================
class AttentionPool(nn.Module):
    """Learnable attention pooling over a sequence of hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, hidden_states, attention_mask=None):
        scores = self.score(hidden_states).squeeze(-1)  # (B, T)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (hidden_states * weights).sum(dim=1)


def masked_mean_pool(hidden_states, attention_mask):
    """Masked mean pooling over all token embeddings (replaces CLS pooling)."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class AudioDownsample(nn.Module):
    """Shrinks the Wav2Vec2 frame sequence using a LEARNED weighted
    combination of overlapping frames (not a uniform average). Overlap
    (kernel_size > stride) means no frame is ever fully discarded — it
    always contributes to at least one output step — and because the
    weights are trainable, the model can learn to emphasize the frames
    that matter instead of blending everything equally like avg-pooling."""
    def __init__(self, dim=768, stride=6, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, C)
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T', C)
        return self.act(self.norm(out))


class FFNResidual(nn.Module):
    """Feed-forward block with residual connection, used after cross-attention fusion."""
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


class FocalLoss(nn.Module):
    """Binary focal loss on logits, with optional class weighting via alpha."""
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ============================================================
# MODEL — hierarchical clinical reasoning:
# audio+text -> encoders -> cross-modal fusion -> shared representation
# -> MMSE estimation -> feature refinement using predicted MMSE
# -> final dementia classification
# ============================================================
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

        bert_trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        w2v_trainable = sum(p.numel() for p in self.wav2vec.parameters() if p.requires_grad)
        print(f"[INFO] BERT   trainable params: {bert_trainable:,} (r={bert_r})")
        print(f"[INFO] Wav2Vec trainable params: {w2v_trainable:,} (r={wav2vec_r})")

        if wav2vec_r > 0 and w2v_trainable == 0:
            raise RuntimeError("Wav2Vec LoRA attached ZERO trainable params!")

        self.text_proj = nn.Linear(768, 256)
        self.audio_proj = nn.Linear(768, 256)
        self.audio_attn_pool = AttentionPool(768)

        # Info-preserving downsample of the ~3750-frame Wav2Vec2 sequence
        # (75s @ 20ms/frame) before it meets the 512-token text sequence
        # in cross-attention, using overlapping learned conv windows
        # instead of a hard average that discards detail permanently.
        self.audio_downsample = AudioDownsample(dim=768, stride=6, kernel_size=9)

        # Sequence-level cross-attention fusion (full token/frame sequences,
        # not single pooled vectors)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(256)
        self.fusion_ffn = FFNResidual(256, expansion=4, dropout=0.1)

        # --- Stage 1: MMSE as an intermediate clinical feature ---
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
        self.mmse_head = nn.Linear(32, 1)  # raw MMSE units, clamp [0,30] only at inference

        # --- Stage 2: final diagnosis, informed by fused features AND cognitive estimate ---
        # Input = fused(256) + mmse_features(32) + mmse_prediction(1) = 289
        self.diagnosis_branch = nn.Sequential(
            nn.Linear(289, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Severity is likewise cognitively-informed
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
            print(f"[WARN] Could not locate transformer layers to unfreeze for {kind}")
            return
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask, audio_waveforms, calibrate=False):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = bert_out.last_hidden_state  # (B, T_text, 768)
        text_pooled = masked_mean_pool(text_seq, attention_mask)

        w2v_out = self.wav2vec(audio_waveforms).last_hidden_state  # (B, T_audio, 768)
        w2v_out = self.audio_downsample(w2v_out)                    # (B, T_audio', 768)
        audio_pooled = self.audio_attn_pool(w2v_out)

        text_seq_proj = self.text_proj(text_seq)     # (B, T_text, 256)
        audio_seq_proj = self.audio_proj(w2v_out)     # (B, T_audio', 256)

        text_key_padding_mask = (attention_mask == 0)

        attn_out, _ = self.cross_attn(
            query=audio_seq_proj, key=text_seq_proj, value=text_seq_proj,
            key_padding_mask=text_key_padding_mask
        )

        fused_seq = self.fusion_norm(attn_out + audio_seq_proj)
        fused_seq = self.fusion_ffn(fused_seq)
        fused = fused_seq.mean(dim=1)  # shared patient representation

        # 1. Estimate cognitive status FIRST
        mmse_hidden = self.mmse_features(fused)
        mmse_preds = self.mmse_head(mmse_hidden)          # raw MMSE units

        # 2. Refine the shared representation with that estimate
        mmse_signal = mmse_preds / 30.0                    # scale-matched to fused features
        fused_infused = torch.cat([fused, mmse_hidden, mmse_signal], dim=-1)

        # 3. Diagnose using BOTH latent features and the cognitive estimate
        diag_logits = self.diagnosis_branch(fused_infused)
        severity_logits = self.severity_branch(fused_infused)

        if calibrate:
            diag_logits = diag_logits / self.temperature

        return diag_logits, mmse_preds, severity_logits


# ============================================================
# SETUP
# ============================================================
WAV_BASE_PATH = os.path.join(DATA_ROOT, "train", "Full_wave_enhanced_audio")
DRIVE_SAVE_DIR = os.path.join(DATA_ROOT, "kfold_checkpoints")
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

PROGRESS_STATE_PATH = os.path.join(DRIVE_SAVE_DIR, "progress_state.json")
RESULTS_CSV_PATH = os.path.join(DRIVE_SAVE_DIR, "kfold_summary_running.csv")

import json

def load_progress_state():
    if os.path.exists(PROGRESS_STATE_PATH):
        with open(PROGRESS_STATE_PATH, 'r') as f:
            return json.load(f)
    return {"completed": [], "last_checkpoint_path": None}

def save_progress_state(state):
    with open(PROGRESS_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)

def append_result_row(row):
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(RESULTS_CSV_PATH)
    df_row.to_csv(RESULTS_CSV_PATH, mode='a', header=write_header, index=False)

N_FOLDS = 5
RANDOM_SEED = 42


def collate_fn(batch):
    audios = [item['audio_waveforms'] for item in batch]
    max_len = max(a.shape[0] for a in audios)
    padded_audio = torch.stack([
        torch.nn.functional.pad(a, (0, max_len - a.shape[0])) for a in audios
    ])
    return {
        'subject_id': [item['subject_id'] for item in batch],
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'audio_waveforms': padded_audio,
        'label': torch.stack([item['label'] for item in batch]),
        'mmse': torch.stack([item['mmse'] for item in batch]),
        'severity': torch.stack([item['severity'] for item in batch]),
    }


# ============================================================
# METRICS HELPERS
# ============================================================
def compute_classification_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity_recall': sensitivity,
        'specificity': specificity,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['roc_auc'] = float('nan')
    return metrics


def find_best_threshold(y_true, y_prob):
    best_thresh, best_score = 0.5, -1.0
    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (np.asarray(y_prob) > thresh).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score, best_thresh = score, thresh
    return best_thresh, best_score


def run_fold(fold_idx, train_df, val_df, bert_r, wav2vec_r, epochs=50, alpha=0.5, beta=0.2,
             patience=7, pos_class_weight=None, use_focal=False, label_smoothing=0.05,
             grad_clip_norm=1.0):
    """Train + evaluate one fold. Returns metrics dict and saves the best
    checkpoint for this fold to Drive."""

    print(f"\n{'='*60}\nFOLD {fold_idx + 1}/{N_FOLDS}\n{'='*60}")

    train_dataset = Wav2VecMultimodalDataset(train_df, WAV_BASE_PATH, verbose_name=f"FOLD{fold_idx+1}-TRAIN")
    val_dataset = Wav2VecMultimodalDataset(val_df, WAV_BASE_PATH, verbose_name=f"FOLD{fold_idx+1}-VAL")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                               pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True, collate_fn=collate_fn)

    model = MultimodalMTLDementiaDetector(bert_r=bert_r, wav2vec_r=wav2vec_r).to(device)

    if use_focal:
        criterion_diag = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=label_smoothing)
    else:
        pos_weight = torch.tensor([pos_class_weight], device=device) if pos_class_weight else None
        criterion_diag_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        def criterion_diag(logits, targets):
            if label_smoothing > 0:
                targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
            return criterion_diag_bce(logits, targets)

    criterion_mmse = nn.SmoothL1Loss(beta=1.0)
    criterion_severity = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

    lora_params, new_head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (lora_params if (("bert" in name or "wav2vec" in name) and "branch" not in name)
         else new_head_params).append(param)

    optimizer = optim.AdamW([
        {'params': lora_params, 'lr': 2e-4},
        {'params': new_head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 1.0 - progress)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    best_bal_acc = -1.0
    best_val_loss = float('inf')
    best_weights = None
    best_threshold_for_fold = 0.5
    patience_counter = 0
    GRAD_ACCUM_STEPS = 4

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, leave=False, desc=f"Fold {fold_idx+1} Epoch [{epoch+1}/{epochs}]")
        optimizer.zero_grad()
        for step, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_waveforms = batch['audio_waveforms'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            mmse_raw = batch['mmse'].to(device).unsqueeze(1)
            severity = batch['severity'].to(device)

            valid_mask = (mmse_raw != MMSE_MISSING).squeeze(1)

            with torch.amp.autocast('cuda'):
                diag_logits, mmse_preds, severity_logits = model(input_ids, attention_mask, audio_waveforms)
                loss_diag = criterion_diag(diag_logits, labels)

                if valid_mask.sum() > 0:
                    loss_mmse = criterion_mmse(mmse_preds[valid_mask] / MMSE_MAX, mmse_raw[valid_mask] / MMSE_MAX)
                    loss_severity = criterion_severity(severity_logits, severity)
                else:
                    loss_mmse = torch.tensor(0.0, device=device)
                    loss_severity = torch.tensor(0.0, device=device)

                loss = loss_diag + alpha * loss_mmse + beta * loss_severity
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            loop.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_total_loss = 0.0
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio_waveforms = batch['audio_waveforms'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                mmse_raw = batch['mmse'].to(device).unsqueeze(1)
                severity = batch['severity'].to(device)
                valid_mask = (mmse_raw != MMSE_MISSING).squeeze(1)

                with torch.amp.autocast('cuda'):
                    diag_logits, mmse_preds, severity_logits = model(input_ids, attention_mask, audio_waveforms)
                    loss_diag = criterion_diag(diag_logits, labels)
                    if valid_mask.sum() > 0:
                        loss_mmse = criterion_mmse(mmse_preds[valid_mask], mmse_raw[valid_mask])
                        loss_severity = criterion_severity(severity_logits, severity)
                    else:
                        loss_mmse = torch.tensor(0.0, device=device)
                        loss_severity = torch.tensor(0.0, device=device)
                    val_loss = loss_diag + alpha * loss_mmse + beta * loss_severity
                    val_total_loss += val_loss.item()

                val_probs.extend(torch.sigmoid(diag_logits).cpu().numpy().flatten().tolist())
                val_labels_list.extend(labels.cpu().numpy().flatten().tolist())

        avg_val_loss = val_total_loss / len(val_loader)

        best_thresh, _ = find_best_threshold(val_labels_list, val_probs)
        val_preds = (np.array(val_probs) > best_thresh).astype(int)
        bal_acc = balanced_accuracy_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)

        improved = bal_acc > best_bal_acc
        print(f"Fold {fold_idx+1} | Epoch {epoch+1:>3}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val BalAcc: {bal_acc:.4f} | Val F1: {val_f1:.4f} | "
              f"Thresh: {best_thresh:.2f}" + (" ✅ best" if improved else ""))

        if improved:
            best_bal_acc = bal_acc
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_threshold_for_fold = best_thresh
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹ Fold {fold_idx+1}: early stopping after {epoch+1} epochs.")
                break

    model.load_state_dict(best_weights)

    fold_save_path = os.path.join(
        DRIVE_SAVE_DIR, f"dementia_mtl_b{bert_r}_w{wav2vec_r}_fold{fold_idx+1}_best.pth"
    )
    torch.save({'state_dict': best_weights, 'threshold': float(best_threshold_for_fold)}, fold_save_path)
    print(f"[SAVED] Fold {fold_idx+1} checkpoint -> {fold_save_path}")

    state = load_progress_state()
    prev_checkpoint = state.get("last_checkpoint_path")
    if prev_checkpoint and prev_checkpoint != fold_save_path and os.path.exists(prev_checkpoint):
        os.remove(prev_checkpoint)
        print(f"[DELETED] Old checkpoint -> {prev_checkpoint}")

    state["last_checkpoint_path"] = fold_save_path
    state["completed"].append([bert_r, wav2vec_r, fold_idx])
    save_progress_state(state)

    # ---- Full evaluation with the tuned threshold ----
    model.eval()
    all_probs, all_labels = [], []
    all_mmse_preds, all_mmse_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_waveforms = batch['audio_waveforms'].to(device)
            labels = batch['label'].numpy()
            mmse_raw = batch['mmse'].numpy()

            with torch.amp.autocast('cuda'):
                diag_logits, mmse_preds, severity_logits = model(input_ids, attention_mask, audio_waveforms)

            probs = torch.sigmoid(diag_logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels)

            for pred, target in zip(mmse_preds.cpu().numpy().flatten(), mmse_raw.flatten()):
                if target != MMSE_MISSING:
                    all_mmse_preds.append(float(np.clip(pred, 0.0, MMSE_MAX)))
                    all_mmse_targets.append(target)

    all_preds = (np.array(all_probs) > best_threshold_for_fold).astype(int)
    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
    fold_rmse = (math.sqrt(mean_squared_error(all_mmse_targets, all_mmse_preds))
                 if len(all_mmse_targets) > 0 else None)
    fold_report = classification_report(all_labels, all_preds, target_names=['Control', 'Dementia'],
                                         output_dict=True, zero_division=0)

    print(f"\n--- FOLD {fold_idx+1} RESULTS ---")
    print(f"BalAcc: {metrics['balanced_accuracy']:.4f} | Sens: {metrics['sensitivity_recall']:.4f} | "
          f"Spec: {metrics['specificity']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f} | "
          f"MMSE RMSE: {fold_rmse if fold_rmse is not None else 'N/A'}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    result_row = {
        'fold': fold_idx + 1, 'bert_rank': bert_r, 'wav2vec_rank': wav2vec_r,
        'threshold': best_threshold_for_fold,
        **metrics,
        'best_val_loss': best_val_loss, 'mmse_rmse': fold_rmse,
        'dementia_f1': fold_report['Dementia']['f1-score'], 'control_f1': fold_report['Control']['f1-score'],
        'checkpoint_path': fold_save_path,
    }
    append_result_row(result_row)
    return result_row


# ============================================================
# STRATIFIED, PATIENT-LEVEL K-FOLD CROSS VALIDATION DRIVER
# Excludes the held-out test_subjects (from the earlier test-split cell)
# so nothing in the ablation ever sees the test set.
# ============================================================
print(f"\n🚀 Starting {N_FOLDS}-Fold Cross-Validation Ablation Study...")

skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
master_df_reset = master_df[~master_df['subject_id'].isin(test_subjects)].reset_index(drop=True)
groups = master_df_reset['subject_id'].astype(str)

n_pos = (master_df_reset['label'] == 1).sum()
n_neg = (master_df_reset['label'] == 0).sum()
POS_CLASS_WEIGHT = (n_neg / max(n_pos, 1))
print(f"[INFO] Class balance -> pos:{n_pos} neg:{n_neg} | pos_class_weight={POS_CLASS_WEIGHT:.3f}")

experiments = [
    (8, 8),
    (0, 0),
    (32, 32),
    (8, 0),
    (0, 8)
]

all_experiment_results = []
progress = load_progress_state()
completed_set = {tuple(x) for x in progress["completed"]}
for (bert_r, wav2vec_r) in experiments:
    print(f"\n\n{'*'*70}")
    print(f"🔬 STARTING EXPERIMENT: BERT_Rank={bert_r} | Wav2Vec_Rank={wav2vec_r}")
    print(f"{'*'*70}")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(master_df_reset, master_df_reset['label'], groups=groups)
    ):
        if (bert_r, wav2vec_r, fold_idx) in completed_set:
            print(f"[SKIP] b{bert_r}_w{wav2vec_r} fold {fold_idx+1} already done, resuming past it")
            continue

        fold_train_df = master_df_reset.iloc[train_idx]
        fold_val_df = master_df_reset.iloc[val_idx]

        overlap = set(fold_train_df['subject_id']) & set(fold_val_df['subject_id'])
        assert not overlap, f"Patient leakage detected in fold {fold_idx+1}: {overlap}"

        result = run_fold(
            fold_idx, fold_train_df, fold_val_df,
            bert_r=bert_r, wav2vec_r=wav2vec_r, epochs=50, alpha=0.5, beta=0.2, patience=7,
            pos_class_weight=POS_CLASS_WEIGHT, use_focal=False, label_smoothing=0.05
        )
        fold_results.append(result)

    acc_arr = np.array([r['balanced_accuracy'] for r in fold_results]) if fold_results else np.array([])
    if len(acc_arr):
        print(f"\n✅ EXPERIMENT (b{bert_r}_w{wav2vec_r}) COMPLETE!")
        print(f"Mean Balanced Accuracy: {acc_arr.mean():.4f} ± {acc_arr.std():.4f}")

    for r in fold_results:
        r['bert_rank'] = bert_r
        r['wav2vec_rank'] = wav2vec_r
        all_experiment_results.append(r)

summary_df = pd.DataFrame(all_experiment_results)
summary_path = os.path.join(DRIVE_SAVE_DIR, "master_ablation_kfold_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n🎉 ALL EXPERIMENTS FINISHED! Master summary saved to -> {summary_path}")

if not summary_df.empty:
    print("\n--- FINAL ABLATION RESULTS (Balanced Accuracy) ---")
    print(summary_df.groupby(['bert_rank', 'wav2vec_rank'])['balanced_accuracy'].mean())

# ============================================================
# FINAL MODEL — trained on ~all data, saved as ONE checkpoint
# ============================================================
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, accuracy_score, 
    confusion_matrix, classification_report
)
from tqdm.auto import tqdm

FINAL_BERT_R = 8
FINAL_WAV2VEC_R = 8
DRIVE_SAVE_DIR = os.path.join(DATA_ROOT, "kfold_checkpoints")
FINAL_MODEL_PATH = os.path.join(DRIVE_SAVE_DIR, "dementia_mtl_FINAL.pth")

def train_final_model(full_df, bert_r, wav2vec_r, epochs=50, alpha=0.5, beta=0.2,
                       patience=7, holdout_frac=0.05, pos_class_weight=None,
                       use_focal=False, label_smoothing=0.05, grad_clip_norm=1.0):
    
    # 1. Patient-level holdout split (No leakage!)
    subject_labels = full_df.groupby('subject_id')['label'].first()
    subject_ids_arr = subject_labels.index.to_numpy()
    subject_labels_arr = subject_labels.to_numpy()

    train_subjects, val_subjects = train_test_split(
        subject_ids_arr, test_size=holdout_frac,
        stratify=subject_labels_arr, random_state=RANDOM_SEED
    )
    train_df = full_df[full_df['subject_id'].isin(train_subjects)]
    val_df = full_df[full_df['subject_id'].isin(val_subjects)]
    
    overlap = set(train_df['subject_id']) & set(val_df['subject_id'])
    assert not overlap, f"Patient leakage detected in final holdout split: {overlap}"
    
    print(f"Final model: training on {len(train_df)} rows, monitoring on {len(val_df)} held-out rows "
          f"({len(train_subjects)} / {len(val_subjects)} patients)")

    train_dataset = Wav2VecMultimodalDataset(train_df, WAV_BASE_PATH, verbose_name="FINAL-TRAIN")
    val_dataset   = Wav2VecMultimodalDataset(val_df,   WAV_BASE_PATH, verbose_name="FINAL-VAL")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = MultimodalMTLDementiaDetector(bert_r=bert_r, wav2vec_r=wav2vec_r).to(device)

    if use_focal:
        criterion_diag = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=label_smoothing)
    else:
        pos_weight = torch.tensor([pos_class_weight], device=device) if pos_class_weight else None
        criterion_diag_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        def criterion_diag(logits, targets):
            if label_smoothing > 0:
                targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
            return criterion_diag_bce(logits, targets)

    criterion_mmse = nn.SmoothL1Loss(beta=1.0)
    criterion_severity = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

    lora_params, new_head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (lora_params if (("bert" in name or "wav2vec" in name) and "branch" not in name) else new_head_params).append(param)

    optimizer = optim.AdamW([
        {'params': lora_params, 'lr': 2e-4},
        {'params': new_head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 1.0 - progress)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    best_bal_acc = -1.0
    best_val_loss = float('inf')
    best_weights = None
    best_threshold = 0.5
    patience_counter = 0
    GRAD_ACCUM_STEPS = 4

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        loop = tqdm(train_loader, leave=False, desc=f"FINAL Epoch [{epoch+1}/{epochs}]")
        
        for step, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_waveforms = batch['audio_waveforms'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            mmse_raw = batch['mmse'].to(device).unsqueeze(1)
            severity = batch['severity'].to(device)
            valid_mask = (mmse_raw != MMSE_MISSING).squeeze(1)

            with torch.amp.autocast('cuda'):
                diag_logits, mmse_preds, severity_logits = model(input_ids, attention_mask, audio_waveforms)
                loss_diag = criterion_diag(diag_logits, labels)

                if valid_mask.sum() > 0:
                    # ✅ FIX: Training MMSE on raw 0-30 units so gradients don't vanish
                    loss_mmse = criterion_mmse(mmse_preds[valid_mask], mmse_raw[valid_mask])
                    loss_severity = criterion_severity(severity_logits, severity)
                else:
                    loss_mmse = torch.tensor(0.0, device=device)
                    loss_severity = torch.tensor(0.0, device=device)

                loss = loss_diag + alpha * loss_mmse + beta * loss_severity
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            total_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_total_loss = 0.0
        val_probs, val_labels_list = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio_waveforms = batch['audio_waveforms'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                mmse_raw = batch['mmse'].to(device).unsqueeze(1)
                severity = batch['severity'].to(device)
                valid_mask = (mmse_raw != MMSE_MISSING).squeeze(1)
                
                with torch.amp.autocast('cuda'):
                    diag_logits, mmse_preds, severity_logits = model(input_ids, attention_mask, audio_waveforms)
                    loss_diag = criterion_diag(diag_logits, labels)
                    if valid_mask.sum() > 0:
                        # ✅ FIX: Validating MMSE on raw units to match training
                        loss_mmse = criterion_mmse(mmse_preds[valid_mask], mmse_raw[valid_mask])
                        loss_severity = criterion_severity(severity_logits, severity)
                    else:
                        loss_mmse = torch.tensor(0.0, device=device)
                        loss_severity = torch.tensor(0.0, device=device)
                    val_total_loss += (loss_diag + alpha * loss_mmse + beta * loss_severity).item()

                val_probs.extend(torch.sigmoid(diag_logits).cpu().numpy().flatten().tolist())
                val_labels_list.extend(labels.cpu().numpy().flatten().tolist())

        avg_val_loss = val_total_loss / len(val_loader)

        thresh, _ = find_best_threshold(val_labels_list, val_probs)
        val_preds = (np.array(val_probs) > thresh).astype(int)
        bal_acc = balanced_accuracy_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)

        improved = bal_acc > best_bal_acc
        print(f"FINAL | Epoch {epoch+1:>3}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val BalAcc: {bal_acc:.4f} | Val F1: {val_f1:.4f} | "
              f"Thresh: {thresh:.2f}" + (" ✅ best" if improved else ""))

        if improved:
            best_bal_acc = bal_acc
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_threshold = thresh
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹ Final model: early stopping after {epoch+1} epochs.")
                break

    model.load_state_dict(best_weights)
    torch.save({'state_dict': best_weights, 'threshold': float(best_threshold)}, FINAL_MODEL_PATH)
    print(f"[SAVED] Final model -> {FINAL_MODEL_PATH} (threshold={best_threshold:.2f}, val BalAcc={best_bal_acc:.4f})")

    # Export globals for legacy cells just in case
    globals()['val_loader'] = val_loader
    globals()['val_dataset'] = val_dataset
    globals()['best_threshold_for_fold'] = best_threshold

    return model, val_loader, best_threshold

# ----------------- EXECUTION & EVALUATION -----------------
print("\n" + "*"*70)
print("🏋️ TRAINING FINAL MODEL ON FULL POOL")
print("*"*70)

# 1. Filter out the test subjects so the final model doesn't see them
master_df_reset = master_df[~master_df['subject_id'].isin(test_subjects)].reset_index(drop=True)

n_pos = (master_df_reset['label'] == 1).sum()
n_neg = (master_df_reset['label'] == 0).sum()
POS_CLASS_WEIGHT = (n_neg / max(n_pos, 1))

# 2. Train the model
final_model, final_val_loader, opt_threshold = train_final_model(
    master_df_reset, FINAL_BERT_R, FINAL_WAV2VEC_R, epochs=50, alpha=0.5, beta=0.2,
    patience=7, pos_class_weight=POS_CLASS_WEIGHT, use_focal=False, label_smoothing=0.05
)

# 3. Post-Training Evaluation on the Holdout Set
print("\n" + "="*70)
print("📊 FINAL HOLDOUT SET EVALUATION (Confusion Matrix & Metrics)")
print("="*70)

final_model.eval()
eval_labels = []
eval_probs = []

with torch.no_grad():
    for batch in final_val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_waveforms = batch['audio_waveforms'].to(device)
        labels = batch['label'].numpy()
        
        with torch.amp.autocast('cuda'):
            diag_logits, _, _ = final_model(input_ids, attention_mask, audio_waveforms)
            
        probs = torch.sigmoid(diag_logits).cpu().numpy().flatten()
        eval_probs.extend(probs)
        eval_labels.extend(labels)

# Calculate final metrics using the optimized threshold
eval_preds = (np.array(eval_probs) > opt_threshold).astype(int)

acc = accuracy_score(eval_labels, eval_preds)
bal_acc = balanced_accuracy_score(eval_labels, eval_preds)
cm = confusion_matrix(eval_labels, eval_preds)

print(f"Optimal Decision Threshold : {opt_threshold:.2f}")
print(f"Overall Accuracy           : {acc:.4f} ({acc*100:.1f}%)")
print(f"Balanced Accuracy          : {bal_acc:.4f} ({bal_acc*100:.1f}%)")
print("\nCONFUSION MATRIX:")
print("                 Pred Control   Pred Dementia")
print(f"True Control   |      {cm[0][0]:3d}       |      {cm[0][1]:3d}      |")
if len(cm) > 1:
    print(f"True Dementia  |      {cm[1][0]:3d}       |      {cm[1][1]:3d}      |")

print("\nCLASSIFICATION REPORT:")
print(classification_report(eval_labels, eval_preds, target_names=["Control", "Dementia"], zero_division=0))
print("="*70)



from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, mean_squared_error,
                              balanced_accuracy_score, roc_curve, precision_recall_curve, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import torch.nn.functional as F
import numpy as np

# 1. Load the absolute best weights from the early stopping process
model = final_model
model.eval()

# NEW: temperature scaling — fit on validation logits before evaluating test set
def fit_temperature(model, val_loader, device, max_iter=50):
    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_waveforms = batch['audio_waveforms'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            with torch.amp.autocast('cuda'):
                diag_logits, _, _ = model(input_ids, attention_mask, audio_waveforms)
            logits_list.append(diag_logits.float().cpu())
            labels_list.append(labels.float().cpu())
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    print(f"[CALIBRATION] Fitted temperature = {temperature.item():.4f}")
    return temperature.item()

fitted_temperature = fit_temperature(model, val_loader, device)
model.temperature.data = torch.tensor([fitted_temperature], device=device)

all_preds = []
all_labels = []
all_probs_raw = []
all_probs_calibrated = []
all_mmse_preds = []
all_mmse_targets = []

# threshold tuned during training (fall back to 0.5 if not available)
CLASSIFICATION_THRESHOLD = globals().get('best_threshold_for_fold', 0.5)

def tta_audio_forward(model, input_ids, attention_mask, audio_waveforms, n_crops=3, crop_frac=0.8):
    """NEW: test-time augmentation for audio — average predictions over multiple
    random crops of the waveform (optional, controlled by n_crops)."""
    diag_logits_list, mmse_list, sev_list = [], [], []
    T = audio_waveforms.shape[-1]
    crop_len = max(int(T * crop_frac), 16000)
    for i in range(n_crops):
        if T > crop_len:
            start = np.random.randint(0, T - crop_len + 1)
            crop = audio_waveforms[:, start:start + crop_len]
        else:
            crop = audio_waveforms
        with torch.amp.autocast('cuda'):
            dl, mp, sl = model(input_ids, attention_mask, crop, calibrate=True)
        diag_logits_list.append(dl)
        mmse_list.append(mp)
        sev_list.append(sl)
    diag_logits = torch.stack(diag_logits_list).mean(0)
    mmse_preds = torch.stack(mmse_list).mean(0)
    severity_logits = torch.stack(sev_list).mean(0)
    return diag_logits, mmse_preds, severity_logits

USE_TTA = False  # flip to True for TTA-averaged inference
test_dataset = Wav2VecMultimodalDataset(test_df, WAV_BASE_PATH, verbose_name="TEST")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                          pin_memory=True, collate_fn=collate_fn)
print("Evaluating Final Multi-Task Model on Test Set...")

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_waveforms = batch['audio_waveforms'].to(device)

        labels = batch['label'].numpy()
        mmse_targets = batch['mmse'].numpy()

        if USE_TTA:
            diag_logits, mmse_preds, severity_logits = tta_audio_forward(
                model, input_ids, attention_mask, audio_waveforms
            )
            raw_logits = diag_logits * model.temperature  # undo calibration for "raw" record
        else:
            with torch.amp.autocast('cuda'):
                raw_logits, mmse_preds, severity_logits = model(input_ids, attention_mask, audio_waveforms)
            diag_logits = raw_logits / model.temperature

        probs_raw = torch.sigmoid(raw_logits).cpu().numpy()
        probs_calibrated = torch.sigmoid(diag_logits).cpu().numpy()
        preds_binary = (probs_calibrated > CLASSIFICATION_THRESHOLD).astype(int).flatten()

        all_preds.extend(preds_binary)
        all_labels.extend(labels)
        all_probs_raw.extend(probs_raw.flatten())
        all_probs_calibrated.extend(probs_calibrated.flatten())

        # NEW: raw MMSE prediction, clamped to [0,30] only at inference
        for pred, target in zip(mmse_preds.cpu().numpy().flatten(), mmse_targets.flatten()):
            if target != -1.0:
                all_mmse_preds.append(float(np.clip(pred, 0.0, 30.0)))
                all_mmse_targets.append(target)

# --- METRICS CALCULATIONS (NEW: full metric suite) ---
accuracy = accuracy_score(all_labels, all_preds)
bal_acc = balanced_accuracy_score(all_labels, all_preds)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
auc = roc_auc_score(all_labels, all_probs_calibrated) if len(set(all_labels)) > 1 else float('nan')

print(f"\n{'='*30}")
print(f"--- EVALUATION RESULTS ---")
print(f"{'='*30}")
print(f"Threshold used            : {CLASSIFICATION_THRESHOLD:.2f}")
print(f"Accuracy                  : {accuracy:.4f}")
print(f"Balanced Accuracy         : {bal_acc:.4f}")
print(f"Sensitivity (Recall)      : {sensitivity:.4f}")
print(f"Specificity               : {specificity:.4f}")
print(f"ROC-AUC                   : {auc:.4f}")

if len(all_mmse_targets) > 0:
    rmse = math.sqrt(mean_squared_error(all_mmse_targets, all_mmse_preds))
    print(f"MMSE Regression RMSE      : {rmse:.4f} points (Lower is better)")
else:
    print("MMSE Regression RMSE      : No valid MMSE scores in test set.")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Control', 'Dementia']))

# --- CONFUSION MATRIX ---
plt.figure(figsize=(6, 4))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Control', 'Dementia'], yticklabels=['Control', 'Dementia'])
plt.title('MTL Architecture: Diagnosis Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# --- NEW: ROC CURVE ---
if len(set(all_labels)) > 1:
    fpr, tpr, _ = roc_curve(all_labels, all_probs_calibrated)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # --- NEW: PRECISION-RECALL CURVE ---
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs_calibrated)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# NEW: tokens/fragments to strip from the explainability output
FILTERED_TOKENS = {'.', ',', '?', '!', '&', '+', '[CLS]', '[SEP]', '[PAD]'}

def is_filtered_token(token: str) -> bool:
    if token in FILTERED_TOKENS:
        return True
    if token.startswith('##'):
        # WordPiece continuation fragments are still shown (merged into the word)
        return False
    if all(ch in '.,?!&+' for ch in token):
        return True
    return False


# ============================================================
# NEW: PAUSE / SILENCE ANALYSIS (acoustic disfluency biomarker)
# ============================================================
def analyze_pauses(audio_waveform_np, sr=16000, top_db=30, min_pause_sec=0.3):
    """Detects silence gaps directly from the raw waveform using
    energy-based silence splitting (librosa.effects.split). Long/frequent
    pauses are a well-established speech biomarker for word-finding
    difficulty and cognitive load in dementia -- this doesn't rely on
    manually annotated .cha pause markers, so it works on any audio."""
    y = audio_waveform_np.astype(np.float32)
    total_duration = len(y) / sr
    if total_duration <= 0:
        return None

    intervals = librosa.effects.split(y, top_db=top_db)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        gap = (start - prev_end) / sr
        if gap >= min_pause_sec:
            pauses.append({'start_sec': round(prev_end / sr, 2),
                            'end_sec': round(start / sr, 2),
                            'duration_sec': round(gap, 2)})
        prev_end = end

    trailing_gap = (len(y) - prev_end) / sr
    if trailing_gap >= min_pause_sec:
        pauses.append({'start_sec': round(prev_end / sr, 2),
                        'end_sec': round(total_duration, 2),
                        'duration_sec': round(trailing_gap, 2)})

    total_pause_time = sum(p['duration_sec'] for p in pauses)
    pauses_per_min = len(pauses) / (total_duration / 60) if total_duration > 0 else 0
    pauses_sorted = sorted(pauses, key=lambda p: p['duration_sec'], reverse=True)

    return {
        'total_duration_sec': round(total_duration, 2),
        'num_pauses': len(pauses),
        'total_pause_time_sec': round(total_pause_time, 2),
        'pause_ratio': round(total_pause_time / total_duration, 3) if total_duration > 0 else 0.0,
        'pauses_per_minute': round(pauses_per_min, 2),
        'longest_pauses': pauses_sorted[:5],
    }


# ============================================================
# NEW: REPETITION DETECTION (perseveration -- a known dementia linguistic marker)
# ============================================================
def detect_repetitions(text, min_word_len=3):
    """Flags content words repeated close together in the transcript.
    Word-finding difficulty in dementia often shows up as repeated words
    or restarted phrases, distinct from normal speech redundancy."""
    words = [w.lower().strip('.,?!') for w in text.split()]
    words = [w for w in words if len(w) >= min_word_len]
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    repeated = {w: c for w, c in counts.items() if c >= 3}
    return dict(sorted(repeated.items(), key=lambda x: x[1], reverse=True)[:8])


def explain_patient_diagnosis(model, dataset, tokenizer, patient_index=0, threshold=0.5):
    """
    Full clinical explainability report for one patient:
      - predicted diagnosis, calibrated probability, confidence
      - predicted MMSE (as used to inform the diagnosis, per the
        cascaded architecture) and severity
      - top clinically significant words (filtered attention)
      - repeated words / perseveration
      - pause / silence analysis from the raw audio
    """
    model.eval()

    item = dataset[patient_index]
    input_ids = item['input_ids'].unsqueeze(0).to(device)
    attention_mask = item['attention_mask'].unsqueeze(0).to(device)

    actual_label = "Dementia" if item['label'].item() == 1.0 else "Control"
    actual_mmse = item['mmse'].item()

    raw_bert = model.bert
    if hasattr(raw_bert, 'base_model'):
        raw_bert = raw_bert.base_model
    if hasattr(raw_bert, 'model'):
        raw_bert = raw_bert.model

    original_output_attentions = raw_bert.config.output_attentions
    original_attn_impl = getattr(raw_bert.config, '_attn_implementation', 'eager')

    raw_bert.config._attn_implementation = "eager"
    raw_bert.config.output_attentions = True

    with torch.no_grad():
        bert_out = raw_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        audio_waveforms = item['audio_waveforms'].unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            # calibrate=True applies temperature scaling for a calibrated probability;
            # mmse_preds here is the SAME value the cascade fed into the diagnosis
            # branch, not a separately computed number
            diag_logits, mmse_preds, severity_logits = model(
                input_ids, attention_mask, audio_waveforms, calibrate=True
            )

        calibrated_prob = torch.sigmoid(diag_logits).item()
        confidence = abs(calibrated_prob - threshold) / max(threshold, 1 - threshold)
        confidence = float(min(confidence, 1.0))

        pred_label = "Dementia" if calibrated_prob > threshold else "Control"
        pred_mmse = float(np.clip(mmse_preds.item(), 0.0, 30.0))
        severity_names = ["Normal", "Mild", "Moderate", "Severe"]
        pred_severity = severity_names[torch.argmax(severity_logits, dim=-1).item()]

    raw_bert.config._attn_implementation = original_attn_impl
    raw_bert.config.output_attentions = original_output_attentions

    if not bert_out.attentions:
        print("Error: The model still refused to return attention weights.")
        return

    last_layer_attention = bert_out.attentions[-1]
    avg_attention = torch.mean(last_layer_attention[0], dim=0)

    mask = attention_mask[0].cpu().numpy().astype(bool)
    token_importance = avg_attention.cpu().numpy()[mask][:, :].mean(axis=0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    word_weights = []
    full_words = []  # for repetition detection, unfiltered reconstruction
    for token, weight in zip(tokens, token_importance):
        if is_filtered_token(token):
            continue
        clean_token = token.replace('##', '')
        word_weights.append((clean_token, weight))
        full_words.append(clean_token)

    if not word_weights:
        print("No valid text found to visualize for this patient.")
        return

    weights_only = [w for _, w in word_weights]
    max_weight = max(weights_only)
    min_weight = min(weights_only)

    # NEW: repetition / perseveration check
    repeated_words = detect_repetitions(' '.join(full_words))

    # NEW: pause analysis on the raw audio for this patient
    pause_stats = analyze_pauses(item['audio_waveforms'].numpy())

    html_string = f"<h3>XAI Analysis: Patient #{patient_index}</h3>"
    html_string += f"<b>Actual Diagnosis:</b> {actual_label} (MMSE: {actual_mmse})<br>"
    html_string += (f"<b>Model Predicted:</b> {pred_label} | Calibrated Prob: {calibrated_prob:.2f} | "
                     f"Confidence: {confidence:.2f}<br>")
    html_string += f"<b>Est. MMSE (used to inform diagnosis above):</b> {pred_mmse:.1f} | <b>Est. Severity:</b> {pred_severity}<br><br>"
    html_string += "<div style='line-height: 1.8; font-size: 16px; padding: 15px; border: 1px solid #ccc; border-radius: 5px;'>"

    for word, weight in word_weights:
        intensity = (weight - min_weight) / (max_weight - min_weight + 1e-9)
        color = f"rgba(255, 0, 0, {intensity * 0.8})"
        html_string += f"<span style='background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 1px;'>{word}</span> "

    html_string += "</div>"

    word_weights.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"CLINICAL EXPLAINABILITY REPORT -- Patient #{patient_index}")
    print(f"{'='*60}")
    print(f"Prediction        : {pred_label}  (actual: {actual_label})")
    print(f"Calibrated Prob   : {calibrated_prob:.4f}  (threshold={threshold:.2f})")
    print(f"Confidence        : {confidence:.4f}")
    print(f"MMSE Prediction   : {pred_mmse:.2f} / 30  (actual: {actual_mmse})")
    print(f"Severity          : {pred_severity}")

    print(f"\n--- Top {min(10, len(word_weights))} Clinically Significant Words (filtered attention) ---")
    for i in range(min(10, len(word_weights))):
        print(f"{i+1}. '{word_weights[i][0]}' (Score: {word_weights[i][1]:.4f})")

    print(f"\n--- Repetition / Perseveration Check ---")
    if repeated_words:
        for word, count in repeated_words.items():
            print(f"  '{word}' repeated {count} times")
    else:
        print("  No significant word repetition detected.")

    print(f"\n--- Pause / Silence Analysis (acoustic disfluency marker) ---")
    if pause_stats:
        print(f"  Total speech duration : {pause_stats['total_duration_sec']}s")
        print(f"  Number of pauses      : {pause_stats['num_pauses']}")
        print(f"  Total pause time      : {pause_stats['total_pause_time_sec']}s")
        print(f"  Pause ratio           : {pause_stats['pause_ratio']*100:.1f}% of recording")
        print(f"  Pauses per minute     : {pause_stats['pauses_per_minute']}")
        if pause_stats['longest_pauses']:
            print(f"  Longest pauses (top 5):")
            for p in pause_stats['longest_pauses']:
                print(f"    {p['start_sec']}s -> {p['end_sec']}s  ({p['duration_sec']}s)")
        if pause_stats['pause_ratio'] > 0.35 or pause_stats['pauses_per_minute'] > 12:
            print("  NOTE: elevated pause frequency/duration -- consistent with word-finding")
            print("        difficulty patterns often seen in cognitive impairment. This is a")
            print("        descriptive signal, not a diagnostic claim on its own.")
    else:
        print("  No audio available for pause analysis.")

    display(HTML(html_string))

# === EXECUTE IT ===
explain_patient_diagnosis(model, test_dataset, train_dataset.text_tokenizer, patient_index=5,
                           threshold=globals().get('best_threshold_for_fold', 0.5))



"""
============================================================
INFERENCE SCRIPT -- Hierarchical Clinical-Reasoning Dementia Detector
============================================================
Loads your trained checkpoint (dementia_mtl_FINAL.pth or any
kfold_checkpoints/*.pth) and runs it on completely unseen
.wav/.mp3 + .cha pairs, printing the diagnosis, calibrated
probability, confidence, MMSE estimate, severity, and filtered
word-level attention explanation for each file.

Pipeline: audio+text -> encoders -> cross-modal fusion ->
shared representation -> MMSE estimation -> feature refinement
using predicted MMSE -> final dementia classification.

USAGE
-----
1. Set CHECKPOINT_PATH, AUDIO_FOLDER, TRANSCRIPT_FOLDER, BERT_R, WAV2VEC_R below.
2. Audio/transcript files matched by filename stem, e.g.
     audio_folder/S045.wav (or .mp3)  <->  transcript_folder/S045.cha
3. (Optional) Fill in TRUE_LABELS for an accuracy report.
4. (Optional) Set USE_TTA = True to average predictions over
   multiple random audio crops.
5. Set RUN_INFERENCE_SCRIPT = True, then run this cell.
"""

import os
import re
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model, Wav2Vec2Processor
from peft import LoraConfig, get_peft_model

# ============================================================
# CONFIG -- EDIT THESE
# ============================================================
CHECKPOINT_PATH   = "./kfold_checkpoints/dementia_mtl_FINAL.pth"
AUDIO_FOLDER      = "./lu/audio/cd"
TRANSCRIPT_FOLDER = "./lu/transcription/cd"

BERT_R    = 8
WAV2VEC_R = 8
UNFREEZE_LAST_N = 1

TRUE_LABELS = {}  # e.g. {"S045": 1, "S046": 0}; 1 = dementia (cd), 0 = control (cc)

TOP_K_WORDS = 10
MAX_AUDIO_SAMPLES = 16000 * 75
MMSE_MAX = 30.0
DEFAULT_THRESHOLD = 0.5

USE_TTA = False
TTA_N_CROPS = 3
TTA_CROP_FRAC = 0.8

RUN_INFERENCE_SCRIPT = True  # flip to True only when you actually want this to run

FILTERED_TOKENS = {'.', ',', '?', '!', '&', '+', '[CLS]', '[SEP]', '[PAD]'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TEXT CLEANING + .cha PARSING
# ============================================================
def clean_transcript(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r'\*\w+:', ' ', t)
    t = re.sub(r'&-\w+', ' ', t)
    t = re.sub(r'\[/+\]', ' ', t)
    t = re.sub(r'\[.*?\]', ' ', t)
    t = re.sub(r'\bxxx\b|\byyy\b', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\(\.+\)', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def parse_cha_file(filepath):
    participant_text = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('*PAR:'):
                    speech = re.sub(r'\*PAR:\s*', '', line)
                    speech = re.sub(r'\[.*?\]', '', speech)
                    speech = re.sub(r'&\w+', '', speech)
                    speech = re.sub(r'<|>', '', speech)
                    speech = re.sub(r'\x15.*?\x15', '', speech)
                    speech = speech.strip()
                    if speech:
                        participant_text.append(speech)
    except Exception as e:
        print(f"[WARN] Could not parse {filepath}: {e}")
    return ' '.join(participant_text)


def is_filtered_token(token: str) -> bool:
    if token in FILTERED_TOKENS:
        return True
    if token.startswith('##'):
        return False
    if all(ch in '.,?!&+' for ch in token):
        return True
    return False


# ============================================================
# SMALL BUILDING BLOCKS (must exactly match the training notebook)
# ============================================================
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


# ============================================================
# MODEL (must exactly match the training architecture)
# ============================================================
class MultimodalMTLDementiaDetector(nn.Module):
    def __init__(self, bert_r=8, wav2vec_r=8, unfreeze_last_n=1):
        super().__init__()
        base_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        base_wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        if bert_r > 0:
            bert_lora_config = LoraConfig(
                r=bert_r, lora_alpha=bert_r * 2,
                target_modules=["query", "value"], lora_dropout=0.1, bias="none"
            )
            self.bert = get_peft_model(base_bert, bert_lora_config)
        else:
            self.bert = base_bert

        if wav2vec_r > 0:
            wav2vec_lora_config = LoraConfig(
                r=wav2vec_r, lora_alpha=wav2vec_r * 2,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                lora_dropout=0.1, bias="none"
            )
            base_wav2vec.enable_input_require_grads = lambda: None
            self.wav2vec = get_peft_model(base_wav2vec, wav2vec_lora_config)
        else:
            self.wav2vec = base_wav2vec

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

        # Stage 1: MMSE as intermediate clinical feature
        self.mmse_features = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU()
        )
        self.mmse_head = nn.Linear(32, 1)

        # Stage 2: diagnosis informed by fused features + cognitive estimate (256+32+1=289)
        self.diagnosis_branch = nn.Sequential(
            nn.Linear(289, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.severity_branch = nn.Sequential(
            nn.Linear(289, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 4)
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
            layers = base.encoder.layer if kind == "bert" else base.encoder.layers
        except AttributeError:
            return
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask, audio_waveforms, calibrate=False):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = bert_out.last_hidden_state

        w2v_out = self.wav2vec(audio_waveforms).last_hidden_state
        w2v_out = self.audio_downsample(w2v_out)

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

        # 1. Estimate cognitive status FIRST
        mmse_hidden = self.mmse_features(fused)
        mmse_preds = self.mmse_head(mmse_hidden)

        # 2. Refine the shared representation with that estimate
        mmse_signal = mmse_preds / 30.0
        fused_infused = torch.cat([fused, mmse_hidden, mmse_signal], dim=-1)

        # 3. Diagnose using BOTH latent features and the cognitive estimate
        diag_logits = self.diagnosis_branch(fused_infused)
        severity_logits = self.severity_branch(fused_infused)

        if calibrate:
            diag_logits = diag_logits / self.temperature

        return diag_logits, mmse_preds, severity_logits


# ============================================================
# SINGLE-PATIENT PREPROCESSING
# ============================================================
def prepare_single_sample(wav_path, cha_path, tokenizer, audio_processor):
    raw_text = parse_cha_file(cha_path)
    text = clean_transcript(raw_text)
    if not text:
        text = "Patient linguistic profile unavailable."

    text_encoding = tokenizer(
        text, max_length=512, padding='max_length', truncation=True, return_tensors='pt'
    )

    try:
        speech_array, sr = librosa.load(wav_path, sr=16000)
        if len(speech_array) > MAX_AUDIO_SAMPLES:
            speech_array = speech_array[:MAX_AUDIO_SAMPLES]
        audio_inputs = audio_processor(speech_array, sampling_rate=16000, return_tensors="pt")
        audio_tensor = audio_inputs.input_values
    except Exception as e:
        print(f"[WARN] Audio load failed for {wav_path}: {e}")
        audio_tensor = torch.zeros(1, 16000)

    return {
        'input_ids': text_encoding['input_ids'],
        'attention_mask': text_encoding['attention_mask'],
        'audio_waveforms': audio_tensor,
        'raw_text': text,
    }


def tta_forward(model, input_ids, attention_mask, audio_waveforms, n_crops=TTA_N_CROPS,
                 crop_frac=TTA_CROP_FRAC):
    diag_list, mmse_list, sev_list = [], [], []
    T = audio_waveforms.shape[-1]
    crop_len = max(int(T * crop_frac), 16000)
    for _ in range(n_crops):
        if T > crop_len:
            start = np.random.randint(0, T - crop_len + 1)
            crop = audio_waveforms[:, start:start + crop_len]
        else:
            crop = audio_waveforms
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            dl, mp, sl = model(input_ids, attention_mask, crop, calibrate=True)
        diag_list.append(dl)
        mmse_list.append(mp)
        sev_list.append(sl)
    return (torch.stack(diag_list).mean(0), torch.stack(mmse_list).mean(0),
            torch.stack(sev_list).mean(0))


# ============================================================
# EXPLAINABILITY
# ============================================================
def explain_prediction(model, sample, tokenizer, top_k=TOP_K_WORDS):
    input_ids = sample['input_ids'].to(device)
    attention_mask = sample['attention_mask'].to(device)

    raw_bert = model.bert
    if hasattr(raw_bert, 'base_model'):
        raw_bert = raw_bert.base_model
    if hasattr(raw_bert, 'model'):
        raw_bert = raw_bert.model

    original_output_attentions = raw_bert.config.output_attentions
    original_attn_impl = getattr(raw_bert.config, '_attn_implementation', 'eager')
    raw_bert.config._attn_implementation = "eager"
    raw_bert.config.output_attentions = True

    with torch.no_grad():
        bert_out = raw_bert(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, return_dict=True
        )

    raw_bert.config._attn_implementation = original_attn_impl
    raw_bert.config.output_attentions = original_output_attentions

    if not bert_out.attentions:
        return []

    last_layer_attention = bert_out.attentions[-1]
    avg_attention = torch.mean(last_layer_attention[0], dim=0)

    mask = attention_mask[0].cpu().numpy().astype(bool)
    token_importance = avg_attention.cpu().numpy()[mask].mean(axis=0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    word_scores = []
    for tok, score in zip(tokens, token_importance):
        if is_filtered_token(tok):
            continue
        clean_tok = tok.replace('##', '')
        word_scores.append((clean_tok, float(score)))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:top_k]


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"--- Running inference on: {device} ---")
    print("Loading tokenizer / audio processor...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=False)
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    print("Building model architecture...")
    model = MultimodalMTLDementiaDetector(
        bert_r=BERT_R, wav2vec_r=WAV2VEC_R, unfreeze_last_n=UNFREEZE_LAST_N
    ).to(device)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        threshold = checkpoint.get('threshold', DEFAULT_THRESHOLD)
    else:
        model.load_state_dict(checkpoint)
        threshold = DEFAULT_THRESHOLD
    print(f"Using classification threshold: {threshold:.2f}")
    model.eval()

    severity_names = ["Normal", "Mild", "Moderate", "Severe"]

    wav_files = sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.wav"))) + \
                sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3")))
    if not wav_files:
        print(f"No .wav/.mp3 files found in {AUDIO_FOLDER}")
        return

    results = []
    for wav_path in wav_files:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        cha_path = os.path.join(TRANSCRIPT_FOLDER, f"{stem}.cha")

        if not os.path.exists(cha_path):
            print(f"[SKIP] No matching .cha for {stem}")
            continue

        sample = prepare_single_sample(wav_path, cha_path, tokenizer, audio_processor)
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        audio_waveforms = sample['audio_waveforms'].to(device)

        with torch.no_grad():
            if USE_TTA:
                diag_logits, mmse_pred, severity_logits = tta_forward(
                    model, input_ids, attention_mask, audio_waveforms
                )
            else:
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    diag_logits, mmse_pred, severity_logits = model(
                        input_ids, attention_mask, audio_waveforms, calibrate=True
                    )

            prob = torch.sigmoid(diag_logits).item()
            confidence = abs(prob - threshold) / max(threshold, 1 - threshold)
            confidence = float(min(confidence, 1.0))
            pred_label = "Dementia" if prob > threshold else "Control"
            pred_mmse = float(np.clip(mmse_pred.item(), 0.0, MMSE_MAX))
            pred_severity = severity_names[torch.argmax(severity_logits, dim=-1).item()]

        top_words = explain_prediction(model, sample, tokenizer)

        print(f"\n{'='*60}")
        print(f"Subject: {stem}")
        print(f"{'='*60}")
        print(f"Predicted diagnosis    : {pred_label}")
        print(f"Calibrated probability : {prob:.4f}  (threshold={threshold:.2f})")
        print(f"Confidence              : {confidence:.4f}")
        print(f"Estimated MMSE (used to inform diagnosis above) : {pred_mmse:.1f} / 30")
        print(f"Estimated severity      : {pred_severity}")
        print(f"Top attended words      : {', '.join(f'{w}({s:.3f})' for w, s in top_words)}")
        if stem in TRUE_LABELS:
            true_label = "Dementia" if TRUE_LABELS[stem] == 1 else "Control"
            match = "OK" if true_label == pred_label else "X"
            print(f"Ground truth            : {true_label}  {match}")

        results.append({
            'subject_id': stem, 'pred_label': pred_label, 'prob_dementia': prob,
            'confidence': confidence, 'pred_mmse': pred_mmse, 'pred_severity': pred_severity,
            'top_words': top_words, 'true_label': TRUE_LABELS.get(stem),
        })

    labeled = [r for r in results if r['true_label'] is not None]
    if labeled:
        correct = sum(1 for r in labeled if (r['pred_label'] == "Dementia") == (r['true_label'] == 1))
        print(f"\n{'='*60}")
        print(f"ACCURACY ON UNSEEN LABELED DATA: {correct}/{len(labeled)} ({correct/len(labeled)*100:.1f}%)")
        print(f"{'='*60}")

    return results


if __name__ == "__main__" and RUN_INFERENCE_SCRIPT:
    main()

!pip install -U crisperwhisper

import sys
!{sys.executable} -m pip install ctranslate2

from crisperwhisper import CrisperWhisperModel

model = CrisperWhisperModel(
    "turbo",
    backend="transformers"
)

print("CrisperWhisper loaded!")
print("Backend:", model.backend)

import os
import re
import glob
import math
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model, Wav2Vec2Processor
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, 
    roc_auc_score, mean_squared_error, classification_report
)

# Import CrisperWhisper from Nyra Health
from crisperwhisper import CrisperWhisperModel

# =====================================================================
# CONFIGURATION — EDIT THESE LOCAL PATHS & SETTINGS
# =====================================================================
CHECKPOINT_PATH   = "./kfold_checkpoints/dementia_mtl_FINAL.pth"
AUDIO_FOLDER      = "./lu/audio/cd"

# Optional: Path to a folder of Control-only audio files for normative threshold calibration.
# Set to None to use textbook default clinical thresholds.
CONTROL_AUDIO_FOLDER = None

BERT_R          = 8
WAV2VEC_R       = 8
UNFREEZE_LAST_N = 1
MMSE_MAX        = 30.0
DEFAULT_THRESH  = 0.5
MAX_SAMPLES     = 16000 * 75

# Ground Truth Dictionary for accuracy evaluation (Subject ID -> 1: Dementia, 0: Control)
TRUE_LABELS = {
    # Example: "002-0": 0, "003-0": 1
}

# Ground Truth MMSE scores for RMSE evaluation (Subject ID -> float score)
TRUE_MMSE = {
    # Example: "002-0": 30.0, "003-0": 18.5
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# MODULE 0: AUTONOMOUS TRANSCRIPTION ENGINE (CrisperWhisper)
# =====================================================================
class TranscriptionEngine:
    """
    Handles autonomous audio-to-text conversion using CrisperWhisper by Nyra Health.
    Preserves disfluencies and extracts timestamped words for clinical pause analysis.
    """
    def __init__(self, model_size="turbo"):
        # "turbo", "medium", "small", or empty string "" for large
        print(f"Loading Nyra Health CrisperWhisper ({model_size}) ...")
        self.model = CrisperWhisperModel(
            model_size,
            backend="transformers"
        )
    def transcribe(self, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        # CrisperWhisper's verbatim mode accurately maintains fillers and stutters
        result = self.model.transcribe(
            audio_path,
            language="en",
            mode="verbatim",
            word_timestamps=True
        )
        
        segments = []
        full_text = ""
        
        # Safely parse the returned result (handling dict vs object structures)
        if isinstance(result, dict):
            full_text = result.get("text", "") or ""
            raw_elements = result.get("words") or result.get("segments") or []
        else:
            full_text = getattr(result, "text", "") or ""
            raw_elements = (
                getattr(result, "words", None)
                or getattr(result, "segments", None)
                or []
            )
            
        for el in raw_elements:
            if isinstance(el, dict):
                segments.append({
                    "start": el.get("start", 0.0), 
                    "end": el.get("end", 0.0), 
                    "text": el.get("word", el.get("text", ""))
                })
            else:
                segments.append({
                    "start": getattr(el, "start", 0.0), 
                    "end": getattr(el, "end", 0.0), 
                    "text": getattr(el, "word", getattr(el, "text", ""))
                })
                
        return full_text, segments


# =====================================================================
# MODULE 1: SPEECH BIOMARKER EXTRACTION
# =====================================================================
class SpeechBiomarkerExtractor:
    def __init__(self):
        self.filled_pauses_vocab = {"uh", "um", "erm", "ah", "hmm", "like", "you know"}
        self.vague_pronouns = {"he", "she", "it", "they", "that", "this", "those", "thing", "stuff"}
        self.function_words = {
            "the", "a", "an", "in", "on", "of", "at", "by", "for", "with", "about",
            "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "out", "off",
            "over", "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "any", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
            "just", "don", "should", "now", "and", "but", "if", "or", "because",
            "as", "until", "while", "is", "am", "are", "was", "were", "be", "been", "being"
        }

    def _clean_words(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
        return [w for w in words if len(w) > 0]

    def _split_sentences(self, text: str) -> List[str]:
        sents = re.split(r'[.?!]|\.{3}|--', text)
        return [s.strip() for s in sents if len(s.strip()) > 0]

    def _compute_lexical_metrics(self, words: List[str]) -> Dict[str, Any]:
        total_words = len(words)
        if total_words == 0:
            return {
                "total_words": 0, "unique_words": 0, "lexical_diversity": 0.0,
                "vocabulary_richness": 0.0, "hapax_legomena": 0, "pronoun_ratio": 0.0,
                "content_word_ratio": 0.0
            }

        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        unique_words = len(word_counts)
        hapax_legomena = sum(1 for count in word_counts.values() if count == 1)
        ttr = unique_words / total_words
        vocab_richness = unique_words / (total_words ** 0.5)

        pronoun_count = sum(1 for w in words if w in self.vague_pronouns)
        pronoun_ratio = pronoun_count / total_words
        content_words = sum(1 for w in words if w not in self.function_words)
        content_word_ratio = content_words / total_words

        return {
            "total_words": total_words, "unique_words": unique_words,
            "lexical_diversity": round(ttr, 3), "vocabulary_richness": round(vocab_richness, 3),
            "hapax_legomena": hapax_legomena, "pronoun_ratio": round(pronoun_ratio, 3),
            "content_word_ratio": round(content_word_ratio, 3)
        }

    def _compute_disfluencies(self, text: str, words: List[str]) -> Dict[str, Any]:
        total_words = max(len(words), 1)

        filled_count = 0
        for fp in self.filled_pauses_vocab:
            filled_count += len(re.findall(r'\b' + re.escape(fp) + r'\b', text.lower()))
        filled_ratio = filled_count / total_words

        hesitation_symbols = len(re.findall(r'\.{3}|--', text))
        hesitation_count = hesitation_symbols + filled_count

        repeated_words_list = []
        repeat_count = 0
        max_repeat_streak = 0
        current_streak = 1

        for i in range(1, len(words)):
            if words[i] == words[i - 1] and len(words[i]) > 2:
                repeat_count += 1
                current_streak += 1
                if words[i] not in repeated_words_list:
                    repeated_words_list.append(words[i])
            else:
                max_repeat_streak = max(max_repeat_streak, current_streak)
                current_streak = 1
        max_repeat_streak = max(max_repeat_streak, current_streak)

        return {
            "filled_pause_count": filled_count,
            "filled_pause_ratio": round(filled_ratio, 3),
            "hesitation_count": hesitation_count,
            "hesitation_ratio": round(hesitation_count / total_words, 3),
            "repeated_words": repeated_words_list,
            "repeat_frequency": repeat_count,
            "repeat_ratio": round(repeat_count / total_words, 3),
            "max_repeated_word_count": max_repeat_streak if repeat_count > 0 else 0
        }

    def _compute_pauses_from_whisper(self, segments: List[Dict[str, Any]], audio_duration: float) -> Dict[str, Any]:
        if not segments or audio_duration <= 0:
            return {
                "pause_count": 0, "total_pause_duration": 0.0, "avg_pause": 0.0,
                "longest_pause": 0.0, "pause_ratio": 0.0, "pauses_per_minute": 0.0
            }

        pauses = []
        for i in range(len(segments) - 1):
            gap = segments[i + 1]["start"] - segments[i]["end"]
            if gap >= 0.4:  
                pauses.append(gap)

        pause_count = len(pauses)
        total_pause_duration = sum(pauses)
        avg_pause = total_pause_duration / max(pause_count, 1)
        longest_pause = max(pauses) if pauses else 0.0
        pause_ratio = total_pause_duration / max(audio_duration, 1e-5)
        pauses_per_min = pause_count / (max(audio_duration, 1e-5) / 60.0)

        return {
            "pause_count": pause_count,
            "total_pause_duration": round(total_pause_duration, 2),
            "avg_pause": round(avg_pause, 2),
            "longest_pause": round(longest_pause, 2),
            "pause_ratio": round(min(pause_ratio, 1.0), 3),
            "pauses_per_minute": round(pauses_per_min, 2)
        }

    def _compute_composite_fluency(self, speech_rate: float, pause_ratio: float, hesitation_ratio: float, repeat_ratio: float) -> float:
        rate_score = min(speech_rate / 130.0, 1.0) * 40.0
        pause_penalty = min(pause_ratio / 0.40, 1.0) * 30.0
        pause_score = max(30.0 - pause_penalty, 0.0)
        hes_penalty = min(hesitation_ratio / 0.15, 1.0) * 15.0
        hes_score = max(15.0 - hes_penalty, 0.0)
        rep_penalty = min(repeat_ratio / 0.10, 1.0) * 15.0
        rep_score = max(15.0 - rep_penalty, 0.0)
        composite = rate_score + pause_score + hes_score + rep_score
        return round(min(max(composite, 0.0), 100.0), 1)

    def extract(
        self,
        transcript: str,
        audio_waveform_np: np.ndarray,
        whisper_segments: List[Dict[str, Any]],
        sr: int = 16000
    ) -> Dict[str, Any]:
        audio_duration = len(audio_waveform_np) / sr
        words = self._clean_words(transcript)
        sentences = self._split_sentences(transcript)

        lex = self._compute_lexical_metrics(words)
        sent_lengths = [len(self._clean_words(s)) for s in sentences if len(self._clean_words(s)) > 0]
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
        max_sent_len = max(sent_lengths) if sent_lengths else 0
        min_sent_len = min(sent_lengths) if sent_lengths else 0

        pause_stats = self._compute_pauses_from_whisper(whisper_segments, audio_duration)
        disf = self._compute_disfluencies(transcript, words)

        speaking_duration = max(audio_duration - pause_stats["total_pause_duration"], 0.1)
        wpm = (lex["total_words"] / max(audio_duration, 1e-5)) * 60.0
        speech_rate = (lex["total_words"] / max(speaking_duration, 1e-5)) * 60.0
        spm = (len(sent_lengths) / max(audio_duration, 1e-5)) * 60.0

        fluency_score = self._compute_composite_fluency(wpm, pause_stats["pause_ratio"], disf["hesitation_ratio"], disf["repeat_ratio"])

        return {
            "words_per_minute": round(wpm, 1),
            "speech_rate": round(speech_rate, 1),
            "sentences_per_minute": round(spm, 1),
            "speaking_duration": round(speaking_duration, 2),
            "fluency_score": fluency_score,
            **pause_stats, **disf, **lex,
            "avg_sentence_length": round(avg_sent_len, 1),
            "maximum_sentence_length": max_sent_len,
            "minimum_sentence_length": min_sent_len
        }


# =====================================================================
# MODULE 1.5: THRESHOLD CALIBRATION
# =====================================================================
DEFAULT_PERCENTILES = {
    "pause_ratio_high": 75,
    "speech_rate_low": 25,
    "repeat_ratio_high": 75,
    "lexical_diversity_low": 25,
    "pronoun_ratio_high": 75,
    "avg_sentence_length_low": 25,
}

def calibrate_thresholds(control_biomarkers: List[Dict[str, Any]], percentiles: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    percentiles = percentiles or DEFAULT_PERCENTILES
    if len(control_biomarkers) < 10:
        raise ValueError(f"Only {len(control_biomarkers)} Control-group samples given.")

    def col(key):
        vals = [b[key] for b in control_biomarkers if key in b]
        return np.array(vals)

    return {
        "pause_ratio_high": float(np.percentile(col("pause_ratio"), percentiles["pause_ratio_high"])),
        "speech_rate_low": float(np.percentile(col("words_per_minute"), percentiles["speech_rate_low"])),
        "repeat_ratio_high": float(np.percentile(col("repeat_ratio"), percentiles["repeat_ratio_high"])),
        "lexical_diversity_low": float(np.percentile(col("lexical_diversity"), percentiles["lexical_diversity_low"])),
        "pronoun_ratio_high": float(np.percentile(col("pronoun_ratio"), percentiles["pronoun_ratio_high"])),
        "avg_sentence_length_low": float(np.percentile(col("avg_sentence_length"), percentiles["avg_sentence_length_low"])),
    }


# =====================================================================
# MODULE 2: CLINICAL EVIDENCE SCORING (WITH OCCLUSION SUPPORT)
# =====================================================================
class ClinicalEvidenceScorer:
    DEFAULT_THRESHOLDS = {
        "pause_ratio_high": 0.22,
        "speech_rate_low": 105.0,
        "repeat_ratio_high": 0.04,
        "lexical_diversity_low": 0.50,
        "pronoun_ratio_high": 0.14,
        "avg_sentence_length_low": 7.0
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)

    def score_evidence(self, biomarkers: Dict[str, Any], occlusion_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        evidence_list = []

        pr = biomarkers.get("pause_ratio", 0.0)
        if pr >= self.thresholds["pause_ratio_high"]:
            evidence_list.append({
                "rank": "★★★★★", "feature": "Long pause ratio",
                "finding": f"{int(pr * 100)}% of recording spent in silent pauses (threshold: {self.thresholds['pause_ratio_high']*100:.0f}%)",
                "clinical_relevance": "Primary acoustic marker of word-retrieval latency and executive hesitation.",
                "severity_weight": 5
            })

        wpm = biomarkers.get("words_per_minute", 120.0)
        if wpm <= self.thresholds["speech_rate_low"]:
            evidence_list.append({
                "rank": "★★★★★", "feature": "Slow speech rate",
                "finding": f"{wpm} words per minute (threshold: <= {self.thresholds['speech_rate_low']:.0f} WPM)",
                "clinical_relevance": "Reduced motor-linguistic processing speed and fluency impairment.",
                "severity_weight": 5
            })

        rep = biomarkers.get("repeat_ratio", 0.0)
        if rep >= self.thresholds["repeat_ratio_high"] or biomarkers.get("repeat_frequency", 0) >= 3:
            evidence_list.append({
                "rank": "★★★★☆", "feature": "Word repetition / Perseveration",
                "finding": f"Repeated words ratio at {int(rep * 100)}% (words: {', '.join(biomarkers.get('repeated_words', [])[:3]) or 'n/a'})",
                "clinical_relevance": "Indicates verbal perseveration and short-term working memory loop errors.",
                "severity_weight": 4
            })

        pushing_toward_dementia = [r for r in occlusion_results if r['delta'] > 0.01]
        if pushing_toward_dementia:
            top_str = ", ".join(f"'{r['text']}' (ΔP={r['delta']:+.3f})" for r in pushing_toward_dementia[:4])
            evidence_list.append({
                "rank": "★★★★☆", "feature": "Neural occlusion evidence (causally tested)",
                "finding": f"Removing these from the transcript measurably lowered the model's own P(dementia): {top_str}",
                "clinical_relevance": "A direct causal test against the trained network's actual output.",
                "severity_weight": 4
            })

        ttr = biomarkers.get("lexical_diversity", 1.0)
        if ttr <= self.thresholds["lexical_diversity_low"]:
            evidence_list.append({
                "rank": "★★★☆☆", "feature": "Reduced lexical diversity",
                "finding": f"Type-Token Ratio of {ttr} (threshold: <= {self.thresholds['lexical_diversity_low']:.2f})",
                "clinical_relevance": "Reflects semantic memory degradation and shrinkage of active vocabulary.",
                "severity_weight": 3
            })

        pron_ratio = biomarkers.get("pronoun_ratio", 0.0)
        if pron_ratio >= self.thresholds["pronoun_ratio_high"]:
            evidence_list.append({
                "rank": "★★★☆☆", "feature": "Elevated vague pronoun usage",
                "finding": f"Pronoun-to-word ratio of {int(pron_ratio * 100)}% (threshold: {self.thresholds['pronoun_ratio_high']*100:.0f}%)",
                "clinical_relevance": "Replacement of specific target nouns with empty references ('it', 'thing', 'they').",
                "severity_weight": 3
            })

        evidence_list.sort(key=lambda x: x["severity_weight"], reverse=True)
        return evidence_list


# =====================================================================
# MODULE 3: CLINICAL EXPLANATION GENERATOR
# =====================================================================
class ClinicalExplanationGenerator:
    def generate(self, biomarkers: Dict[str, Any], prediction_meta: Dict[str, Any], evidence_list: List[Dict[str, Any]]) -> str:
        prob = prediction_meta.get("dementia_probability", 0.0)
        mmse = prediction_meta.get("mmse_prediction", 30.0)
        severity = prediction_meta.get("severity_prediction", "Normal")
        threshold = prediction_meta.get("threshold", 0.5)

        is_positive = prob >= threshold
        has_biomarkers = len(evidence_list) > 0

        if is_positive and not has_biomarkers:
            return (
                f"Although no individual handcrafted biomarker exceeded predefined normative thresholds, "
                f"the learned neural representation detected a combination of subtle acoustic and semantic "
                f"speech characteristics associated with dementia (Predicted Probability: {prob*100:.1f}%). "
                f"The network's estimated cognitive score maps to {mmse:.1f} on the MMSE scale."
            )

        evidence_features = [e["feature"].lower() for e in evidence_list]
        
        acoustic_clauses = []
        if any("pause" in f for f in evidence_features):
            pr = biomarkers.get("pause_ratio", 0.0)
            acoustic_clauses.append(f"frequent acoustic hesitations ({int(pr * 100)}% of recording in silence)")
        if any("speech rate" in f for f in evidence_features):
            wpm = biomarkers.get("words_per_minute", 130.0)
            acoustic_clauses.append(f"a markedly slow articulation rate ({wpm} WPM)")

        if acoustic_clauses:
            sent_acoustic = "The participant demonstrated " + " and ".join(acoustic_clauses) + ", indicative of motor-linguistic slowing or word-retrieval latency."
        else:
            sent_acoustic = "Acoustic fluency and speech rate were relatively preserved."

        lex_clauses = []
        if any("repetition" in f for f in evidence_features):
            lex_clauses.append("instances of verbal perseveration")
        if any("lexical diversity" in f for f in evidence_features):
            lex_clauses.append("reduced vocabulary richness")
        if any("pronoun" in f for f in evidence_features):
            lex_clauses.append("an over-reliance on empty pronouns instead of specific target nouns")

        if lex_clauses:
            sent_lex = "Semantic analysis revealed " + ", alongside ".join(lex_clauses) + "."
        else:
            sent_lex = "Lexical selection and semantic specificity appeared intact."

        if is_positive:
            sent_synth = (
                f"These extracted features strongly align with the neural network's estimated cognitive score "
                f"of {mmse:.1f} (MMSE-scale), supporting a classification of {severity} cognitive impairment "
                f"(Predicted Probability: {prob*100:.1f}%)."
            )
        else:
            sent_synth = (
                f"These findings align with the model's estimated cognitive score of {mmse:.1f} (MMSE-scale), "
                f"showing no clinically actionable speech features of neurodegenerative impairment."
            )

        return f"{sent_acoustic} {sent_lex} {sent_synth}"


# =====================================================================
# MODULE 4: CLINICAL REPORT GENERATOR
# =====================================================================
class ClinicalReportGenerator:
    def generate_report(self, prediction_meta: Dict[str, Any], biomarkers: Dict[str, Any], evidence_list: List[Dict[str, Any]], clinical_explanation: str) -> str:
        prob = prediction_meta.get("dementia_probability", 0.0)
        mmse = prediction_meta.get("mmse_prediction", 30.0)
        severity = prediction_meta.get("severity_prediction", "Normal")
        threshold = prediction_meta.get("threshold", 0.5)
        
        pred_label = "POSITIVE (Dementia Detected)" if prob >= threshold else "NEGATIVE (Normal Cognition)"

        lines = [
            "====================================================================",
            "                   DEMENTIA SPEECH ASSESSMENT REPORT                ",
            "====================================================================",
            "",
            "1. MODEL PREDICTION SUMMARY",
            "--------------------------------------------------------------------",
            f"   Diagnostic Prediction : {pred_label}",
            f"   Predicted Probability : {prob * 100:.1f}%",
            f"   Est. Cognitive Score  : {mmse:.1f} (MMSE-Scale)",
            f"   Estimated Severity    : {severity}",
            "",
            "2. KEY SPEECH & ACOUSTIC BIOMARKERS",
            "--------------------------------------------------------------------",
            f"   Fluency Score         : {biomarkers.get('fluency_score', 0):.1f} / 100",
            f"   Speech Rate           : {biomarkers.get('words_per_minute', 0):.1f} WPM",
            f"   Pause Ratio           : {biomarkers.get('pause_ratio', 0)*100:.1f}% of total speaking duration",
            f"   Significant Pauses    : {biomarkers.get('pause_count', 0)} (Avg: {biomarkers.get('avg_pause', 0):.2f}s | Max: {biomarkers.get('longest_pause', 0):.2f}s)",
            f"   Lexical Diversity     : {biomarkers.get('lexical_diversity', 0):.2f} (Type-Token Ratio)",
            f"   Pronoun-to-Word Ratio : {biomarkers.get('pronoun_ratio', 0)*100:.1f}%",
            f"   Word Repetition Ratio : {biomarkers.get('repeat_ratio', 0)*100:.1f}% ({biomarkers.get('repeat_frequency', 0)} repeated instances)",
            "",
            "3. CLINICAL EXPLANATION & INTERPRETATION",
            "--------------------------------------------------------------------",
            f"   {self._wrap_text(clinical_explanation, width=64, indent='   ')}",
            "",
            "4. EVIDENCE RANKING (IN ORDER OF CLINICAL SIGNIFICANCE)",
            "--------------------------------------------------------------------"
        ]

        if not evidence_list:
            lines.append("   No individual handcrafted biomarkers exceeded normative clinical thresholds.")
        else:
            for item in evidence_list:
                lines.append(f"   {item['rank']}  {item['feature'].upper()}")
                lines.append(f"            Finding   : {item['finding']}")
                lines.append(f"            Relevance : {item['clinical_relevance']}")
                lines.append("")

        lines.extend([
            "5. CLINICAL RECOMMENDATIONS",
            "--------------------------------------------------------------------",
            "   • Correlate acoustic speech findings with formal neuropsychological",
            "     testing (e.g., MoCA, standard MMSE, or Boston Naming Test)."
        ])

        if prob >= threshold:
            if severity in ["Moderate", "Severe"] or prob > 0.85:
                lines.append("   • ACTION: Given the high predicted probability and severity, consider")
                lines.append("     immediate referral for comprehensive neurological evaluation.")
            else:
                lines.append("   • ACTION: Consider longitudinal speech follow-up in 3-6 months")
                lines.append("     to assess the rate of cognitive and linguistic decline.")
        else:
            if prob > (threshold - 0.15):
                lines.append("   • ACTION: Prediction is negative but borderline. Recommend additional")
                lines.append("     cognitive screening in 6 months to rule out early MCI.")
            else:
                lines.append("   • ACTION: Speech biomarkers remain within normative boundaries;")
                lines.append("     routine annual screening is sufficient.")

        lines.extend([
            "",
            "6. IMPORTANT CONSTRAINTS & DISCLAIMERS",
            "--------------------------------------------------------------------",
            "   • This report is generated as an assistive post-inference explainability",
            "     layer and does NOT constitute a standalone medical diagnosis.",
            "   • Acoustic pause ratios may be influenced by recording quality, room",
            "     acoustics, or non-cognitive speech disorders (e.g., dysarthria).",
            "===================================================================="
        ])
        return "\n".join(lines)

    def _wrap_text(self, text: str, width: int = 64, indent: str = "") -> str:
        words = text.split()
        wrapped = []
        current_line = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > width:
                wrapped.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
            else:
                current_line.append(word)
                current_len += len(word) + 1
        if current_line:
            wrapped.append(" ".join(current_line))
        return ('\n' + indent).join(wrapped)


# =====================================================================
# MODULE ARCHITECTURE: MULTIMODAL MTL NEURAL NETWORK
# =====================================================================
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, hidden_states, attention_mask=None):
        scores = self.score(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)

class AudioDownsample(nn.Module):
    def __init__(self, dim=768, stride=6, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
    def forward(self, x):
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.act(self.norm(out))

class FFNResidual(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x + self.net(x))

class MultimodalMTLDementiaDetector(nn.Module):
    def __init__(self, bert_r=8, wav2vec_r=8, unfreeze_last_n=1):
        super().__init__()
        base_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        base_wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        if bert_r > 0:
            bert_lora_config = LoraConfig(r=bert_r, lora_alpha=bert_r * 2, target_modules=["query", "value"], lora_dropout=0.1, bias="none")
            self.bert = get_peft_model(base_bert, bert_lora_config)
        else:
            self.bert = base_bert

        if wav2vec_r > 0:
            wav2vec_lora_config = LoraConfig(r=wav2vec_r, lora_alpha=wav2vec_r * 2, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], lora_dropout=0.1, bias="none")
            base_wav2vec.enable_input_require_grads = lambda: None
            self.wav2vec = get_peft_model(base_wav2vec, wav2vec_lora_config)
        else:
            self.wav2vec = base_wav2vec

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
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU()
        )
        self.mmse_head = nn.Linear(32, 1)

        self.diagnosis_branch = nn.Sequential(
            nn.Linear(289, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.severity_branch = nn.Sequential(
            nn.Linear(289, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 4)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    @staticmethod
    def _unfreeze_last_layers(module, n, kind):
        base = module
        if hasattr(base, "base_model"): base = base.base_model
        if hasattr(base, "model"): base = base.model
        try:
            layers = base.encoder.layer if kind == "bert" else base.encoder.layers
        except AttributeError:
            return
        for layer in list(layers)[-n:]:
            for p in layer.parameters(): p.requires_grad = True

    def forward(self, input_ids, attention_mask, audio_waveforms, calibrate=False):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = bert_out.last_hidden_state

        w2v_out = self.wav2vec(audio_waveforms).last_hidden_state
        w2v_out = self.audio_downsample(w2v_out)

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


# =====================================================================
# CAUSAL OCCLUSION ATTRIBUTION
# =====================================================================
def get_occlusion_importance(
    model: MultimodalMTLDementiaDetector,
    tokenizer: AutoTokenizer,
    transcript: str,
    audio_tensor: torch.Tensor,
    base_prob: float
) -> List[Dict[str, Any]]:
    model.eval()
    words = transcript.split()
    if not words:
        return []

    blocks = []
    i = 0
    negations = {"not", "no", "never", "don't", "doesn't", "didn't", "can't", "won't", "couldn't"}
    while i < len(words):
        w_clean = re.sub(r"[^\w']", "", words[i].lower())
        if w_clean in negations and i + 1 < len(words):
            blocks.append((i, i + 2, f"{words[i]} {words[i+1]}"))
            i += 2
        else:
            blocks.append((i, i + 1, words[i]))
            i += 1

    occlusion_results = []
    for start_idx, end_idx, block_text in blocks:
        occluded_words = words[:start_idx] + [""] * (end_idx - start_idx) + words[end_idx:]
        occluded_transcript = " ".join(w for w in occluded_words if w)
        if not occluded_transcript:
            occluded_transcript = "..."

        text_encoding = tokenizer(occluded_transcript, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = text_encoding['input_ids'].to(device)
        attention_mask = text_encoding['attention_mask'].to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                diag_logits, _, _ = model(input_ids, attention_mask, audio_tensor, calibrate=True)
            occ_prob = torch.sigmoid(diag_logits).item()

        delta = base_prob - occ_prob
        occlusion_results.append({
            'text': re.sub(r"[^\w'\s]", "", block_text),
            'delta': delta
        })

    occlusion_results.sort(key=lambda x: x['delta'], reverse=True)
    return occlusion_results


# =====================================================================
# INFERENCE ENGINE
# =====================================================================
def run_single_inference(
    model: MultimodalMTLDementiaDetector,
    tokenizer: AutoTokenizer,
    audio_processor: Wav2Vec2Processor,
    audio_path: str,
    transcript: str,
    threshold: float = 0.5
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]]]:
    
    speech_array, sr = librosa.load(audio_path, sr=16000)
    if len(speech_array) > MAX_SAMPLES:
        speech_array = speech_array[:MAX_SAMPLES]
    audio_tensor = audio_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)

    if not transcript:
        transcript = "..."
    text_encoding = tokenizer(transcript, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = text_encoding['input_ids'].to(device)
    attention_mask = text_encoding['attention_mask'].to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            diag_logits, mmse_pred, severity_logits = model(input_ids, attention_mask, audio_tensor, calibrate=True)

        prob = torch.sigmoid(diag_logits).item()
        pred_mmse = float(np.clip(mmse_pred.item(), 0.0, MMSE_MAX))
        severity_names = ["Normal", "Mild", "Moderate", "Severe"]
        pred_severity = severity_names[torch.argmax(severity_logits, dim=-1).item()]

    occlusion_results = get_occlusion_importance(model, tokenizer, transcript, audio_tensor, prob)

    meta = {
        "dementia_probability": prob,
        "mmse_prediction": pred_mmse,
        "severity_prediction": pred_severity,
        "threshold": threshold
    }
    return meta, speech_array, occlusion_results


# =====================================================================
# BATCH EXECUTION & REPORTING PIPELINE
# =====================================================================
class DementiaBatchPipeline:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.extractor = SpeechBiomarkerExtractor()
        self.scorer = ClinicalEvidenceScorer(thresholds=thresholds)
        self.explainer = ClinicalExplanationGenerator()
        self.reporter = ClinicalReportGenerator()

    def process_single(
        self,
        prediction_meta: dict,
        transcript: str,
        audio_waveform_np: np.ndarray,
        occlusion_results: List[Dict[str, Any]],
        whisper_segments: List[Dict[str, Any]]
    ) -> dict:
        biomarkers = self.extractor.extract(
            transcript=transcript, 
            audio_waveform_np=audio_waveform_np, 
            whisper_segments=whisper_segments,
            sr=16000
        )
        evidence_ranking = self.scorer.score_evidence(biomarkers, occlusion_results)
        clinical_explanation = self.explainer.generate(biomarkers, prediction_meta, evidence_ranking)
        report_text = self.reporter.generate_report(prediction_meta, biomarkers, evidence_ranking, clinical_explanation)

        return {
            "biomarkers": biomarkers,
            "evidence_ranking": evidence_ranking,
            "clinical_explanation": clinical_explanation,
            "report_text": report_text
        }

    @staticmethod
    def generate_final_evaluation_report(
        batch_results: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> str:
        y_true = []
        y_pred = []
        y_prob = []
        mmse_true = []
        mmse_pred = []

        for r in batch_results:
            sid = r["subject_id"]
            if sid in TRUE_LABELS:
                y_true.append(TRUE_LABELS[sid])
                y_pred.append(1 if r["prob_dementia"] >= threshold else 0)
                y_prob.append(r["prob_dementia"])
            if sid in TRUE_MMSE:
                mmse_true.append(TRUE_MMSE[sid])
                mmse_pred.append(r["pred_mmse"])

        lines = [
            "\n====================================================================",
            "             FINAL BATCH TEST EVALUATION SUMMARY                    ",
            "====================================================================",
            f"Total Subjects Evaluated: {len(batch_results)}",
            f"Subjects with Labeled Ground Truth: {len(y_true)}"
        ]

        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")

            lines.extend([
                "--------------------------------------------------------------------",
                "CLASSIFICATION METRICS:",
                f"  • Accuracy            : {acc:.4f} ({acc*100:.1f}%)",
                f"  • Balanced Accuracy   : {bal_acc:.4f} ({bal_acc*100:.1f}%)",
                f"  • Sensitivity (Recall): {sens:.4f} ({sens*100:.1f}%)",
                f"  • Specificity         : {spec:.4f} ({spec*100:.1f}%)",
                f"  • ROC-AUC             : {auc:.4f}",
                "",
                "CONFUSION MATRIX:",
                f"                   Pred Control     Pred Dementia",
                f"  True Control   |      {tn:3d}       |       {fp:3d}       |",
                f"  True Dementia  |      {fn:3d}       |       {tp:3d}       |",
                "--------------------------------------------------------------------"
            ])

        if len(mmse_true) > 0:
            rmse = math.sqrt(mean_squared_error(mmse_true, mmse_pred))
            lines.extend([
                "MMSE REGRESSION METRICS:",
                f"  • RMSE (Root Mean Sq Error): {rmse:.2f} points",
                "--------------------------------------------------------------------"
            ])

        lines.append("====================================================================\n")
        return "\n".join(lines)


# =====================================================================
# MAIN RUNNER
# =====================================================================
def main():
    print(f"--- Launching Autonomous Audio-Only Dementia Engine on: {device} ---")
    
    # 1. Load processors, MTL Model, and Transcription Engine
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=False)
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    model = MultimodalMTLDementiaDetector(bert_r=BERT_R, wav2vec_r=WAV2VEC_R, unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    threshold = checkpoint.get("threshold", DEFAULT_THRESH) if isinstance(checkpoint, dict) else DEFAULT_THRESH
    model.eval()

    transcriber = TranscriptionEngine(model_size="turbo")

    # 2. Normative Threshold Calibration (Optional)
    calibrated_thresholds = None
    if CONTROL_AUDIO_FOLDER:
        print("\n--- Running Normative Calibration on Control Dataset ---")
        extractor = SpeechBiomarkerExtractor()
        control_biomarkers = []
        ctrl_wavs = sorted(glob.glob(os.path.join(CONTROL_AUDIO_FOLDER, "*.[wW][aA][vV]"))) + \
                    sorted(glob.glob(os.path.join(CONTROL_AUDIO_FOLDER, "*.[mM][pP]3")))

        for wav_path in ctrl_wavs:
            transcript, segments = transcriber.transcribe(wav_path)
            if not transcript: continue
            
            speech_array, sr = librosa.load(wav_path, sr=16000)
            control_biomarkers.append(extractor.extract(transcript, speech_array, segments, sr=sr))

        if len(control_biomarkers) >= 10:
            calibrated_thresholds = calibrate_thresholds(control_biomarkers)
            print("✓ Successfully calibrated normative cutoffs from local Control data.")
        else:
            print("[WARN] Less than 10 Control files found. Falling back to default clinical thresholds.")

    # 3. Instantiate Pipeline
    pipeline = DementiaBatchPipeline(thresholds=calibrated_thresholds)

    # 4. Find all test files
    wav_files = sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.[wW][aA][vV]"))) + \
                sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.[mM][pP]3")))
    if not wav_files:
        print(f"No audio files found in: {AUDIO_FOLDER}")
        return

    batch_results = []

    print("\n--- Processing Autonomous Audio Batch ---")
    for wav_path in wav_files:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nProcessing Subject: {stem}...")

        # STEP A: Autonomous Transcription
        transcript, segments = transcriber.transcribe(wav_path)
        if not transcript:
            print(f"[SKIP] Transcription failed or resulted in empty text for: {stem}")
            continue

        # STEP B: Run Inference
        meta, waveform, occlusions = run_single_inference(
            model, tokenizer, audio_processor, wav_path, transcript, threshold=threshold
        )

        # STEP C: Run Explainability
        output = pipeline.process_single(meta, transcript, waveform, occlusions, segments)

        # Store Batch Result
        batch_results.append({
            "subject_id": stem,
            "prob_dementia": meta["dementia_probability"],
            "pred_mmse": meta["mmse_prediction"],
            "pred_severity": meta["severity_prediction"],
            "report_text": output["report_text"]
        })

        # Print individual clinical report
        print(output["report_text"])

    # 5. Generate & Print Final Evaluation Report
    final_report = pipeline.generate_final_evaluation_report(batch_results, threshold=threshold)
    print(final_report)

if __name__ == "__main__":
    main()

import os
import re
import glob
import math
import gc
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model, Wav2Vec2Processor
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, 
    roc_auc_score, mean_squared_error, classification_report
)

# Import CrisperWhisper from Nyra Health
from crisperwhisper import CrisperWhisperModel

# =====================================================================
# CONFIGURATION
# =====================================================================
CHECKPOINT_PATH   = "./kfold_checkpoints/dementia_mtl_FINAL.pth"
AUDIO_FOLDER      = "./lu/audio/cd"
CONTROL_AUDIO_FOLDER = None

BERT_R          = 8
WAV2VEC_R       = 8
UNFREEZE_LAST_N = 1
MMSE_MAX        = 30.0
DEFAULT_THRESH  = 0.5

# OPTIMIZATION: Dropped to 30 seconds to prevent initial Wav2Vec2 OOM
MAX_SAMPLES     = 16000 * 30 

TRUE_LABELS = {}
TRUE_MMSE = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# MODULE 0 & 1: TRANSCRIPTION & BIOMARKERS (Unchanged)
# =====================================================================
class TranscriptionEngine:
    def __init__(self, model_size="turbo"):
        print(f"Loading Nyra Health CrisperWhisper ({model_size}) ...")
        self.model = CrisperWhisperModel(model_size)

    def transcribe(self, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        result = self.model.transcribe(audio_path, language="en")
        segments = []
        full_text = ""
        if isinstance(result, dict):
            full_text = result.get("text", "")
            raw_elements = result.get("words", result.get("segments", []))
        else:
            full_text = getattr(result, "text", "")
            raw_elements = getattr(result, "words", getattr(result, "segments", []))
            
        for el in raw_elements:
            if isinstance(el, dict):
                segments.append({"start": el.get("start", 0.0), "end": el.get("end", 0.0), "text": el.get("word", el.get("text", ""))})
            else:
                segments.append({"start": getattr(el, "start", 0.0), "end": getattr(el, "end", 0.0), "text": getattr(el, "word", getattr(el, "text", ""))})
        return full_text, segments

class SpeechBiomarkerExtractor:
    def __init__(self):
        self.filled_pauses_vocab = {"uh", "um", "erm", "ah", "hmm", "like", "you know"}
        self.vague_pronouns = {"he", "she", "it", "they", "that", "this", "those", "thing", "stuff"}
        self.function_words = {"the", "a", "an", "in", "on", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "and", "but", "if", "or", "because", "as", "until", "while", "is", "am", "are", "was", "were", "be", "been", "being"}

    def _clean_words(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
        return [w for w in words if len(w) > 0]

    def _split_sentences(self, text: str) -> List[str]:
        sents = re.split(r'[.?!]|\.{3}|--', text)
        return [s.strip() for s in sents if len(s.strip()) > 0]

    def _compute_lexical_metrics(self, words: List[str]) -> Dict[str, Any]:
        total_words = len(words)
        if total_words == 0:
            return {"total_words": 0, "unique_words": 0, "lexical_diversity": 0.0, "vocabulary_richness": 0.0, "hapax_legomena": 0, "pronoun_ratio": 0.0, "content_word_ratio": 0.0}
        word_counts = {}
        for w in words: word_counts[w] = word_counts.get(w, 0) + 1
        unique_words = len(word_counts)
        hapax_legomena = sum(1 for count in word_counts.values() if count == 1)
        ttr = unique_words / total_words
        vocab_richness = unique_words / (total_words ** 0.5)
        pronoun_count = sum(1 for w in words if w in self.vague_pronouns)
        content_words = sum(1 for w in words if w not in self.function_words)
        return {"total_words": total_words, "unique_words": unique_words, "lexical_diversity": round(ttr, 3), "vocabulary_richness": round(vocab_richness, 3), "hapax_legomena": hapax_legomena, "pronoun_ratio": round(pronoun_count / total_words, 3), "content_word_ratio": round(content_words / total_words, 3)}

    def _compute_disfluencies(self, text: str, words: List[str]) -> Dict[str, Any]:
        total_words = max(len(words), 1)
        filled_count = sum(len(re.findall(r'\b' + re.escape(fp) + r'\b', text.lower())) for fp in self.filled_pauses_vocab)
        hesitation_symbols = len(re.findall(r'\.{3}|--', text))
        repeated_words_list, repeat_count, max_repeat_streak, current_streak = [], 0, 0, 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1] and len(words[i]) > 2:
                repeat_count += 1; current_streak += 1
                if words[i] not in repeated_words_list: repeated_words_list.append(words[i])
            else:
                max_repeat_streak = max(max_repeat_streak, current_streak); current_streak = 1
        return {"filled_pause_count": filled_count, "filled_pause_ratio": round(filled_count / total_words, 3), "hesitation_count": hesitation_symbols + filled_count, "hesitation_ratio": round((hesitation_symbols + filled_count) / total_words, 3), "repeated_words": repeated_words_list, "repeat_frequency": repeat_count, "repeat_ratio": round(repeat_count / total_words, 3), "max_repeated_word_count": max(max_repeat_streak, current_streak) if repeat_count > 0 else 0}

    def _compute_pauses_from_whisper(self, segments: List[Dict[str, Any]], audio_duration: float) -> Dict[str, Any]:
        if not segments or audio_duration <= 0: return {"pause_count": 0, "total_pause_duration": 0.0, "avg_pause": 0.0, "longest_pause": 0.0, "pause_ratio": 0.0, "pauses_per_minute": 0.0}
        pauses = [segments[i + 1]["start"] - segments[i]["end"] for i in range(len(segments) - 1) if segments[i + 1]["start"] - segments[i]["end"] >= 0.4]
        total_pause_duration = sum(pauses)
        return {"pause_count": len(pauses), "total_pause_duration": round(total_pause_duration, 2), "avg_pause": round(total_pause_duration / max(len(pauses), 1), 2), "longest_pause": round(max(pauses) if pauses else 0.0, 2), "pause_ratio": round(min(total_pause_duration / max(audio_duration, 1e-5), 1.0), 3), "pauses_per_minute": round(len(pauses) / (max(audio_duration, 1e-5) / 60.0), 2)}

    def extract(self, transcript: str, audio_waveform_np: np.ndarray, whisper_segments: List[Dict[str, Any]], sr: int = 16000) -> Dict[str, Any]:
        audio_duration = len(audio_waveform_np) / sr
        words, sentences = self._clean_words(transcript), self._split_sentences(transcript)
        lex, pause_stats, disf = self._compute_lexical_metrics(words), self._compute_pauses_from_whisper(whisper_segments, audio_duration), self._compute_disfluencies(transcript, words)
        sent_lengths = [len(self._clean_words(s)) for s in sentences if len(self._clean_words(s)) > 0]
        speaking_duration = max(audio_duration - pause_stats["total_pause_duration"], 0.1)
        wpm = (lex["total_words"] / max(audio_duration, 1e-5)) * 60.0
        return {"words_per_minute": round(wpm, 1), "speech_rate": round((lex["total_words"] / max(speaking_duration, 1e-5)) * 60.0, 1), "sentences_per_minute": round((len(sent_lengths) / max(audio_duration, 1e-5)) * 60.0, 1), "speaking_duration": round(speaking_duration, 2), "fluency_score": round(min(max(min(wpm / 130.0, 1.0) * 40.0 + max(30.0 - min(pause_stats["pause_ratio"] / 0.40, 1.0) * 30.0, 0.0) + max(15.0 - min(disf["hesitation_ratio"] / 0.15, 1.0) * 15.0, 0.0) + max(15.0 - min(disf["repeat_ratio"] / 0.10, 1.0) * 15.0, 0.0), 0.0), 100.0), 1), **pause_stats, **disf, **lex, "avg_sentence_length": round(sum(sent_lengths) / max(len(sent_lengths), 1), 1), "maximum_sentence_length": max(sent_lengths) if sent_lengths else 0, "minimum_sentence_length": min(sent_lengths) if sent_lengths else 0}

def calibrate_thresholds(control_biomarkers: List[Dict[str, Any]]) -> Dict[str, float]:
    col = lambda k: np.array([b[k] for b in control_biomarkers if k in b])
    return {"pause_ratio_high": float(np.percentile(col("pause_ratio"), 75)), "speech_rate_low": float(np.percentile(col("words_per_minute"), 25)), "repeat_ratio_high": float(np.percentile(col("repeat_ratio"), 75)), "lexical_diversity_low": float(np.percentile(col("lexical_diversity"), 25)), "pronoun_ratio_high": float(np.percentile(col("pronoun_ratio"), 75)), "avg_sentence_length_low": float(np.percentile(col("avg_sentence_length"), 25))}

# =====================================================================
# MODULE 2 & 3: SCORING & EXPLANATION (Unchanged)
# =====================================================================
class ClinicalEvidenceScorer:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {"pause_ratio_high": 0.22, "speech_rate_low": 105.0, "repeat_ratio_high": 0.04, "lexical_diversity_low": 0.50, "pronoun_ratio_high": 0.14, "avg_sentence_length_low": 7.0}
    def score_evidence(self, biomarkers, occlusion_results):
        e = []
        if biomarkers.get("pause_ratio", 0.0) >= self.thresholds["pause_ratio_high"]: e.append({"rank": "★★★★★", "feature": "Long pause ratio", "finding": f"{int(biomarkers['pause_ratio'] * 100)}% of recording spent in silent pauses", "clinical_relevance": "Primary acoustic marker of word-retrieval latency.", "severity_weight": 5})
        if biomarkers.get("words_per_minute", 120.0) <= self.thresholds["speech_rate_low"]: e.append({"rank": "★★★★★", "feature": "Slow speech rate", "finding": f"{biomarkers['words_per_minute']} WPM", "clinical_relevance": "Reduced motor-linguistic processing speed.", "severity_weight": 5})
        if biomarkers.get("repeat_ratio", 0.0) >= self.thresholds["repeat_ratio_high"]: e.append({"rank": "★★★★☆", "feature": "Word repetition", "finding": f"Repeated words ratio at {int(biomarkers['repeat_ratio'] * 100)}%", "clinical_relevance": "Indicates verbal perseveration.", "severity_weight": 4})
        occlusions = [r for r in occlusion_results if r['delta'] > 0.01][:4]
        if occlusions: e.append({"rank": "★★★★☆", "feature": "Neural occlusion evidence", "finding": f"Removing these lowered P(dementia): {', '.join(r['text'] for r in occlusions)}", "clinical_relevance": "Direct causal test.", "severity_weight": 4})
        if biomarkers.get("lexical_diversity", 1.0) <= self.thresholds["lexical_diversity_low"]: e.append({"rank": "★★★☆☆", "feature": "Reduced lexical diversity", "finding": f"Type-Token Ratio of {biomarkers['lexical_diversity']}", "clinical_relevance": "Semantic memory degradation.", "severity_weight": 3})
        if biomarkers.get("pronoun_ratio", 0.0) >= self.thresholds["pronoun_ratio_high"]: e.append({"rank": "★★★☆☆", "feature": "Elevated vague pronouns", "finding": f"Pronoun ratio of {int(biomarkers['pronoun_ratio'] * 100)}%", "clinical_relevance": "Replacement of specific nouns.", "severity_weight": 3})
        return sorted(e, key=lambda x: x["severity_weight"], reverse=True)

class ClinicalExplanationGenerator:
    def generate(self, b, m, e): return f"Model confidence: {m.get('dementia_probability',0)*100:.1f}%. Est MMSE: {m.get('mmse_prediction',30):.1f}."

class ClinicalReportGenerator:
    def generate_report(self, m, b, e, expl):
        prob, mmse, sev, thresh = m.get("dementia_probability", 0.0), m.get("mmse_prediction", 30.0), m.get("severity_prediction", "Normal"), m.get("threshold", 0.5)
        lines = [
            "====================================================================",
            "                   DEMENTIA SPEECH ASSESSMENT REPORT                ",
            "====================================================================",
            f" Prediction: {'POSITIVE (Dementia)' if prob >= thresh else 'NEGATIVE (Normal)'} | Prob: {prob*100:.1f}% | Est MMSE: {mmse:.1f} | Sev: {sev}",
            "--------------------------------------------------------------------",
            f" Fluency: {b.get('fluency_score',0):.1f}/100 | WPM: {b.get('words_per_minute',0):.1f} | Pause Ratio: {b.get('pause_ratio',0)*100:.1f}%"
        ]
        for item in e: lines.append(f" {item['rank']} {item['feature'].upper()}: {item['finding']}")
        lines.append("====================================================================\n")
        return "\n".join(lines)

# =====================================================================
# MODULE ARCHITECTURE: OPTIMIZED MULTIMODAL MTL NEURAL NETWORK
# =====================================================================
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, hidden_states, attention_mask=None):
        scores = self.score(hidden_states).squeeze(-1)
        if attention_mask is not None: scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        return (hidden_states * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(dim=1)

class AudioDownsample(nn.Module):
    def __init__(self, dim=768, stride=6, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x.transpose(1, 2)).transpose(1, 2)))

class FFNResidual(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * expansion), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * expansion, dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))

class MultimodalMTLDementiaDetector(nn.Module):
    def __init__(self, bert_r=8, wav2vec_r=8, unfreeze_last_n=1):
        super().__init__()
        base_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        base_wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        if bert_r > 0:
            self.bert = get_peft_model(base_bert, LoraConfig(r=bert_r, lora_alpha=bert_r * 2, target_modules=["query", "value"], lora_dropout=0.1, bias="none"))
        else: self.bert = base_bert

        if wav2vec_r > 0:
            base_wav2vec.enable_input_require_grads = lambda: None
            self.wav2vec = get_peft_model(base_wav2vec, LoraConfig(r=wav2vec_r, lora_alpha=wav2vec_r * 2, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], lora_dropout=0.1, bias="none"))
        else: self.wav2vec = base_wav2vec

        if unfreeze_last_n > 0:
            self._unfreeze_last_layers(self.bert, unfreeze_last_n, kind="bert")
            self._unfreeze_last_layers(self.wav2vec, unfreeze_last_n, kind="wav2vec")

        self.text_proj = nn.Linear(768, 256)
        self.audio_proj = nn.Linear(768, 256)
        self.audio_downsample = AudioDownsample(dim=768, stride=6, kernel_size=9)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(256)
        self.fusion_ffn = FFNResidual(256, expansion=4, dropout=0.1)

        self.mmse_features = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.mmse_head = nn.Linear(32, 1)
        self.diagnosis_branch = nn.Sequential(nn.Linear(289, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.severity_branch = nn.Sequential(nn.Linear(289, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 4))
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    @staticmethod
    def _unfreeze_last_layers(module, n, kind):
        base = module
        if hasattr(base, "base_model"): base = base.base_model
        if hasattr(base, "model"): base = base.model
        try: layers = base.encoder.layer if kind == "bert" else base.encoder.layers
        except AttributeError: return
        for layer in list(layers)[-n:]:
            for p in layer.parameters(): p.requires_grad = True

    # OPTIMIZATION: Added audio_features parameter to skip Wav2Vec2 re-encoding
    def forward(self, input_ids, attention_mask, audio_waveforms=None, audio_features=None, calibrate=False):
        text_seq = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Only process raw audio if cached features aren't provided
        if audio_features is None:
            w2v_out = self.wav2vec(audio_waveforms).last_hidden_state
            audio_features = self.audio_downsample(w2v_out)

        text_seq_proj = self.text_proj(text_seq)
        audio_seq_proj = self.audio_proj(audio_features)
        
        attn_out, _ = self.cross_attn(
            query=audio_seq_proj, key=text_seq_proj, value=text_seq_proj,
            key_padding_mask=(attention_mask == 0)
        )

        fused = self.fusion_ffn(self.fusion_norm(attn_out + audio_seq_proj)).mean(dim=1)
        mmse_hidden = self.mmse_features(fused)
        mmse_preds = self.mmse_head(mmse_hidden)
        
        fused_infused = torch.cat([fused, mmse_hidden, mmse_preds / 30.0], dim=-1)
        diag_logits = self.diagnosis_branch(fused_infused)
        severity_logits = self.severity_branch(fused_infused)

        if calibrate: diag_logits = diag_logits / self.temperature
        return diag_logits, mmse_preds, severity_logits

# =====================================================================
# OPTIMIZED CAUSAL OCCLUSION ATTRIBUTION
# =====================================================================
def get_occlusion_importance(
    model: MultimodalMTLDementiaDetector,
    tokenizer: AutoTokenizer,
    transcript: str,
    audio_tensor: torch.Tensor,
    base_prob: float
) -> List[Dict[str, Any]]:
    model.eval()
    words = transcript.split()
    if not words: return []

    # OPTIMIZATION: Cache Wav2Vec2 audio features ONCE. 
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            w2v_out = model.wav2vec(audio_tensor).last_hidden_state
            cached_audio_features = model.audio_downsample(w2v_out)

    blocks = []
    i = 0
    negations = {"not", "no", "never", "don't", "doesn't", "didn't", "can't", "won't", "couldn't"}
    while i < len(words):
        w_clean = re.sub(r"[^\w']", "", words[i].lower())
        if w_clean in negations and i + 1 < len(words):
            blocks.append((i, i + 2, f"{words[i]} {words[i+1]}"))
            i += 2
        else:
            blocks.append((i, i + 1, words[i]))
            i += 1

    occlusion_results = []
    for start_idx, end_idx, block_text in blocks:
        occluded_words = words[:start_idx] + [""] * (end_idx - start_idx) + words[end_idx:]
        occluded_transcript = " ".join(w for w in occluded_words if w) or "..."

        text_encoding = tokenizer(occluded_transcript, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = text_encoding['input_ids'].to(device)
        attention_mask = text_encoding['attention_mask'].to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                # Pass cached_audio_features to skip Wav2Vec2 encoding
                diag_logits, _, _ = model(
                    input_ids, attention_mask, 
                    audio_waveforms=None, 
                    audio_features=cached_audio_features, 
                    calibrate=True
                )
            occ_prob = torch.sigmoid(diag_logits).item()

        occlusion_results.append({'text': re.sub(r"[^\w'\s]", "", block_text), 'delta': base_prob - occ_prob})
        
        # OPTIMIZATION: Force clear loop variables from VRAM
        del input_ids, attention_mask, diag_logits
        torch.cuda.empty_cache()

    occlusion_results.sort(key=lambda x: x['delta'], reverse=True)
    
    # Clean up cached features
    del cached_audio_features
    torch.cuda.empty_cache()
    
    return occlusion_results

# =====================================================================
# INFERENCE ENGINE
# =====================================================================
def run_single_inference(model, tokenizer, audio_processor, audio_path, transcript, threshold=0.5):
    speech_array, sr = librosa.load(audio_path, sr=16000)
    if len(speech_array) > MAX_SAMPLES: speech_array = speech_array[:MAX_SAMPLES]
    audio_tensor = audio_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)

    text_encoding = tokenizer(transcript or "...", max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            diag_logits, mmse_pred, severity_logits = model(
                text_encoding['input_ids'].to(device), 
                text_encoding['attention_mask'].to(device), 
                audio_tensor, 
                calibrate=True
            )

        prob = torch.sigmoid(diag_logits).item()
        pred_mmse = float(np.clip(mmse_pred.item(), 0.0, MMSE_MAX))
        pred_severity = ["Normal", "Mild", "Moderate", "Severe"][torch.argmax(severity_logits, dim=-1).item()]

    occlusions = get_occlusion_importance(model, tokenizer, transcript, audio_tensor, prob)
    return {"dementia_probability": prob, "mmse_prediction": pred_mmse, "severity_prediction": pred_severity, "threshold": threshold}, speech_array, occlusions

# =====================================================================
# MAIN RUNNER WITH GARBAGE COLLECTION
# =====================================================================
def main():
    print(f"--- Launching Optimized Autonomous Audio Engine on: {device} ---")
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=False)
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = MultimodalMTLDementiaDetector(bert_r=BERT_R, wav2vec_r=WAV2VEC_R, unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    threshold = checkpoint.get("threshold", DEFAULT_THRESH) if isinstance(checkpoint, dict) else DEFAULT_THRESH
    model.eval()

    transcriber = TranscriptionEngine(model_size="turbo")
    pipeline = DementiaBatchPipeline(thresholds=None)

    wav_files = sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.[wW][aA][vV]"))) + sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.[mM][pP]3")))
    if not wav_files: return print(f"No audio files found in: {AUDIO_FOLDER}")

    batch_results = []
    print("\n--- Processing Autonomous Audio Batch ---")
    for wav_path in wav_files:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nProcessing Subject: {stem}...")

        transcript, segments = transcriber.transcribe(wav_path)
        if not transcript:
            print(f"[SKIP] Transcription failed: {stem}")
            continue

        # OPTIMIZATION: Clear VRAM prior to heavy inference
        torch.cuda.empty_cache()
        gc.collect()

        meta, waveform, occlusions = run_single_inference(model, tokenizer, audio_processor, wav_path, transcript, threshold)
        output = pipeline.process_single(meta, transcript, waveform, occlusions, segments)
        
        batch_results.append({
            "subject_id": stem, "prob_dementia": meta["dementia_probability"], 
            "pred_mmse": meta["mmse_prediction"], "pred_severity": meta["severity_prediction"], 
            "report_text": output["report_text"]
        })
        print(output["report_text"])

        # OPTIMIZATION: Aggressive per-file memory cleanup
        del meta, waveform, occlusions, output
        torch.cuda.empty_cache()
        gc.collect()

    print(pipeline.generate_final_evaluation_report(batch_results, threshold=threshold))

class DementiaBatchPipeline:
    def __init__(self, thresholds=None):
        self.extractor, self.scorer, self.explainer, self.reporter = SpeechBiomarkerExtractor(), ClinicalEvidenceScorer(thresholds), ClinicalExplanationGenerator(), ClinicalReportGenerator()
    def process_single(self, m, t, w, o, s):
        b = self.extractor.extract(t, w, s)
        e = self.scorer.score_evidence(b, o)
        return {"report_text": self.reporter.generate_report(m, b, e, self.explainer.generate(b, m, e))}
    @staticmethod
    def generate_final_evaluation_report(results, threshold=0.5): return "Batch testing complete."

if __name__ == "__main__":
    main()

!pip install transformers