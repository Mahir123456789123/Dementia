from backend.config import MOCK_ANALYSIS_MODE

class TranscriptionEngine:
    def __init__(self):
        if MOCK_ANALYSIS_MODE:
            self.model = None
            return
        self.model = None

    def _initialize(self):
        if self.model is not None:
            return
        try:
            from crisperwhisper import CrisperWhisperModel
            self.model = CrisperWhisperModel("turbo", backend="transformers")
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "CrisperWhisper is unavailable. Install backend requirements and "
                f"try again. Details: {exc}"
            ) from exc

    def transcribe(self, audio_array):
        if MOCK_ANALYSIS_MODE:
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
            
        self._initialize()
        result = self.model.transcribe(
            audio_array, 
            language="en", 
            mode="verbatim", 
            word_timestamps=True
        )
        
        segments = []
        for word_info in getattr(result, "words", []) or []:
            segments.append({
                "start": float(getattr(word_info, "start", 0.0)),
                "end": float(getattr(word_info, "end", 0.0)),
                "text": getattr(word_info, "word", "")
            })
            
        return {
            "text": result.text,
            "segments": segments
        }

transcription_engine = TranscriptionEngine()
