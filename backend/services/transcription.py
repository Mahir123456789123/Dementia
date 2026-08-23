import sys
from backend.config import MOCK_ANALYSIS_MODE

class TranscriptionEngine:
    def __init__(self):
        if MOCK_ANALYSIS_MODE:
            self.model = None
            return
            
        try:
            from crisperwhisper import CrisperWhisper
            self.model = CrisperWhisper(model="turbo")
        except ImportError:
            print("WARNING: CrisperWhisper not found. Please install the required package.")
            self.model = None

    def transcribe(self, audio_array):
        if MOCK_ANALYSIS_MODE or self.model is None:
            # Mock verbatim transcript with timestamps
            return {
                "text": "well um I see a boy taking a cookie from the jar and he is about to fall.",
                "segments": [
                    {"start": 0.0, "end": 0.4, "text": "well"},
                    {"start": 0.8, "end": 1.0, "text": "um"},
                    {"start": 1.5, "end": 1.7, "text": "I"},
                    {"start": 1.7, "end": 1.9, "text": "see"},
                    {"start": 2.0, "end": 2.1, "text": "a"},
                    {"start": 2.1, "end": 2.5, "text": "boy"},
                    {"start": 2.6, "end": 2.9, "text": "taking"},
                    {"start": 3.0, "end": 3.1, "text": "a"},
                    {"start": 3.1, "end": 3.5, "text": "cookie"},
                    {"start": 3.6, "end": 3.8, "text": "from"},
                    {"start": 3.8, "end": 3.9, "text": "the"},
                    {"start": 3.9, "end": 4.2, "text": "jar"},
                    {"start": 5.0, "end": 5.2, "text": "and"},
                    {"start": 5.3, "end": 5.4, "text": "he"},
                    {"start": 5.4, "end": 5.6, "text": "is"},
                    {"start": 5.6, "end": 5.9, "text": "about"},
                    {"start": 5.9, "end": 6.0, "text": "to"},
                    {"start": 6.0, "end": 6.4, "text": "fall"}
                ]
            }
            
        # Using exact kwargs from the prompt
        result = self.model.transcribe(
            audio_array, 
            language="en", 
            mode="verbatim", 
            word_timestamps=True
        )
        
        # Format the segments structure as expected by backend
        segments = []
        for word_info in result.get("words", []):
            segments.append({
                "start": word_info["start"],
                "end": word_info["end"],
                "text": word_info["word"]
            })
            
        return {
            "text": result["text"],
            "segments": segments
        }

transcription_engine = TranscriptionEngine()
