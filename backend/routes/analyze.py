import os
import tempfile
import time
import librosa
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from backend.services.transcription import transcription_engine
from backend.services.biomarker_extractor import SpeechBiomarkerExtractor, ClinicalEvidenceScorer
from backend.services.inference import inference_engine
from backend.config import THRESHOLDS

router = APIRouter()

@router.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    start_time = time.time()
    
    # 1. Read and validate audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")
        
    temp_audio_path = None
    try:
        # Preserve the container extension so audioread/ffmpeg can identify WebM/OGG.
        suffix = os.path.splitext(audio.filename or "recording.wav")[1].lower() or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        audio_array, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
        audio_duration = len(audio_array) / sr
    except Exception as e:
        content_type = audio.content_type or "unknown"
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not decode the uploaded audio ({content_type}). "
                "Please upload WAV, WebM, or OGG audio. "
                f"Decoder error: {e}"
            ),
        )
    finally:
        if temp_audio_path:
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass
        
    if audio_duration < 1.0:
        raise HTTPException(status_code=400, detail="Recording is too short.")
        
    # 2. Transcribe
    try:
        transcription_result = transcription_engine.transcribe(audio_array)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    transcript_text = transcription_result["text"]
    segments = transcription_result["segments"]
    
    if not transcript_text.strip():
        raise HTTPException(status_code=400, detail="No speech detected in recording.")
        
    # 3. Model Inference (Dementia, MMSE, Severity, Occlusion)
    inference_result = inference_engine.run_single_inference(transcript_text, audio_array)
    
    # 4. Biomarker Extraction
    biomarker_extractor = SpeechBiomarkerExtractor()
    biomarkers = biomarker_extractor.extract(transcript_text, segments, audio_duration)
    
    # 5. Evidence Scoring
    evidence_scorer = ClinicalEvidenceScorer(THRESHOLDS)
    evidence = evidence_scorer.score(biomarkers)
    
    # 6. Response Construction
    processing_time = time.time() - start_time
    
    response = {
        "status": "success",
        "prediction": {
            "label": inference_result["predicted_label"],
            "probability": inference_result["dementia_probability"],
            "threshold": inference_result["threshold"],
            "confidence": inference_result["confidence"]
        },
        "mmse": {
            "prediction": inference_result["mmse_prediction"],
            "min": 0,
            "max": 30
        },
        "severity": {
            "label": inference_result["severity_prediction"]
        },
        "transcription": {
            "text": transcript_text,
            "segments": segments
        },
        "speech_biomarkers": biomarkers,
        "evidence": evidence,
        "attribution": inference_result["attribution"],
        "metadata": {
            "audio_duration": round(audio_duration, 1),
            "model_version": "dementia_mtl_FINAL",
            "processing_time": round(processing_time, 2)
        }
    }
    
    return JSONResponse(content=response)
