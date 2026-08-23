import os
import torch
from transformers import AutoTokenizer, Wav2Vec2Processor
from backend.models.dementia_model import MultimodalMTLDementiaDetector
from backend.config import CHECKPOINT_PATH, TEXT_MODEL_NAME, AUDIO_MODEL_NAME, MOCK_ANALYSIS_MODE

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return
            
        if MOCK_ANALYSIS_MODE:
            self.device = "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=False)
            self.audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_NAME)
            self.model = None
            self.threshold = 0.5
            self.initialized = True
            return

        print("Loading models into memory...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=False)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_NAME)
        
        # Load model structure
        self.model = MultimodalMTLDementiaDetector(bert_r=8, wav2vec_r=8)
        
        # Load checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.threshold = checkpoint.get('threshold', 0.5)
            print(f"Loaded checkpoint from {CHECKPOINT_PATH} (threshold: {self.threshold})")
        else:
            print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}. Using untrained weights.")
            self.threshold = 0.5
            
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

model_manager = ModelManager()
