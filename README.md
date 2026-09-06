# Dementia Speech Analysis

Multimodal dementia screening prototype using speech transcription, acoustic and linguistic biomarkers, and a neural model that combines text and audio.

The project contains:

- A FastAPI backend for audio upload, transcription, inference, biomarker extraction, and explainability.
- A Next.js frontend for recording or uploading audio and viewing the analysis report.
- `Major_Dementia.ipynb`, which documents data preparation, training, ablation studies, evaluation, and standalone inference.

> This is a research prototype and not a medical diagnostic device. Predictions and reports must be reviewed by a qualified professional.

## Architecture

The inference pipeline is:

```text
Audio upload
	-> CrisperWhisper transcription
	-> Bio_ClinicalBERT text encoding + Wav2Vec2 audio encoding
	-> cross-modal fusion
	-> dementia classification
	-> MMSE estimation and severity prediction
	-> speech biomarkers and evidence report
```

The trained model uses:

- Text encoder: `emilyalsentzer/Bio_ClinicalBERT`
- Audio encoder: `facebook/wav2vec2-base`
- LoRA rank: 8 for both encoders
- Audio sample rate: 16 kHz
- Backend upload limit: audio is analyzed with a 30-second model input limit
- Outputs: dementia probability, classification threshold, estimated MMSE score, severity, transcript, biomarkers, and evidence

## Project Structure

```text
Major Project/
|-- backend/                 FastAPI service
|   |-- main.py              API entry point
|   |-- config.py            Model paths and runtime settings
|   |-- models/              Model architecture
|   |-- routes/              API routes
|   `-- services/             Inference, transcription, and biomarkers
|-- frontend/                Next.js web application
|-- Major_Dementia.ipynb     Training and inference notebook
|-- kfold_checkpoints/       Trained model checkpoints
|-- train/                   Training audio and transcripts
|-- lu/                      Additional audio and transcript data
`-- MODEL_INTEGRATION.md     Model integration notes
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Node.js 18 or newer and npm
- FFmpeg available on `PATH` for WebM/OGG decoding
- NVIDIA GPU with a compatible PyTorch installation is recommended for real inference
- Approximately 8 GB or more of available system memory for model loading

## Backend Setup

From the repository root, create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

Start the API:

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Verify that the service is running:

```text
http://localhost:8000/api/health
```

The interactive API documentation is available at:

```text
http://localhost:8000/docs
```

### Backend configuration

The backend expects the final checkpoint at:

```text
kfold_checkpoints/dementia_mtl_FINAL.pth
```

To run the API without loading the model, enable mock mode before starting it:

```powershell
$env:MOCK_ANALYSIS_MODE = "true"
python -m uvicorn backend.main:app --reload --port 8000
```

Unset the variable or set it to `false` to use real inference:

```powershell
Remove-Item Env:MOCK_ANALYSIS_MODE -ErrorAction SilentlyContinue
```

## Frontend Setup

Install the frontend dependencies:

```powershell
cd frontend
npm install
```

Start the development server:

```powershell
npm run dev
```

Open:

```text
http://localhost:3000
```

The frontend defaults to `http://localhost:8000` for the API. To use another backend URL:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:8000"
npm run dev
```

Useful frontend commands:

```powershell
npm run lint
npm run build
npm start
```

## API Usage

Analyze an audio file with `curl`:

```powershell
curl.exe -X POST "http://localhost:8000/api/analyze" -F "audio=@path\to\recording.wav"
```

Supported upload formats include WAV, WebM, and OGG when the required decoder is installed. Recordings shorter than one second or recordings without detected speech are rejected.

The response includes:

- `prediction`: label, dementia probability, threshold, confidence, and decision
- `mmse`: estimated score on a 0-30 scale
- `severity`: Normal, Mild, Moderate, or Severe
- `transcription`: transcript text and timestamped segments
- `speech_biomarkers`: pause, fluency, repetition, lexical, and speech-rate features
- `evidence`: threshold-based clinical evidence items
- `attribution`: model attribution or occlusion results
- `explainability`: generated explanation and report text
- `metadata`: audio duration, model version, and processing time

## Notebook Workflow

Open `Major_Dementia.ipynb` in VS Code or Jupyter and run the sections in order:

1. Install the notebook dependencies and verify CUDA.
2. Set `DATA_ROOT` to the project directory or dataset location.
3. Parse CHAT transcripts and create combined metadata.
4. Build the multimodal dataframe and patient-level train/test split.
5. Run the model architecture and training cells.
6. Run cross-validation or the 80/20 ablation study as required.
7. Save the final checkpoint to `kfold_checkpoints/dementia_mtl_FINAL.pth`.
8. Run evaluation, explainability, or the standalone inference section.

The notebook downloads the Hugging Face model weights on first use. Training is computationally expensive and is intended for a CUDA-enabled environment.

## Troubleshooting

### Checkpoint not found

Place `dementia_mtl_FINAL.pth` in `kfold_checkpoints/`. The backend will start with untrained weights if the file is missing, but its predictions should not be used.

### CUDA or out-of-memory errors

Confirm the selected PyTorch build can see the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Use a CPU environment only for smoke tests or mock mode; real inference with both transformer encoders may be slow and memory intensive.

### Audio decoder errors

Install FFmpeg and ensure `ffmpeg -version` works in a new terminal. WAV is the simplest format for local testing.

### Frontend cannot reach the API

Start the backend on port 8000 first, then confirm `http://localhost:8000/api/health` responds. Set `NEXT_PUBLIC_API_BASE_URL` if the API runs on another host or port.

## Data and Privacy

Do not commit participant audio, transcripts, cookies, credentials, or private datasets. Keep checkpoints and personally identifiable research data in controlled storage and follow the consent and data-governance requirements for the dataset.