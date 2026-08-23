import os

# Application Settings
MOCK_ANALYSIS_MODE = os.getenv("MOCK_ANALYSIS_MODE", "false").lower() == "true"

# Model Paths
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(DATA_ROOT, "kfold_checkpoints", "dementia_mtl_FINAL.pth")

# Model IDs
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"

# Audio constraints
SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = 16000 * 30  # 30 seconds limit

# Evidence Thresholds (Default fallbacks from notebook)
THRESHOLDS = {
    "pause_ratio_high": 0.22,
    "speech_rate_low": 105.0,
    "repeat_ratio_high": 0.04,
    "lexical_diversity_low": 0.50,
    "pronoun_ratio_high": 0.14,
    "avg_sentence_length_low": 7.0
}
