# Conditioning on Estimated Cognitive State: A Hierarchical Multimodal Approach to Speech-Based Dementia Screening



Hetanshi Shah, Tanay Kinariwala, Mahir Shah, Kailas Devadkar, Jignesh Sisodia
Sardar Patel Institute of Technology, Mumbai, India

> **Research prototype. Not a medical device.** This system is a screening aid for research purposes. It is not a diagnostic instrument, and its outputs must be reviewed by a qualified clinician.

---

## Table of contents

1. [Overview](#overview)
2. [Key results](#key-results)
3. [Method](#method)
4. [Repository structure](#repository-structure)
5. [Requirements](#requirements)
6. [Data and checkpoints](#data-and-checkpoints)
7. [Reproducing the paper (notebook)](#reproducing-the-paper-notebook)
8. [Running the web application](#running-the-web-application)
9. [API reference](#api-reference)
10. [Post-hoc evidence layer](#post-hoc-evidence-layer)
11. [Model variants and full results](#model-variants-and-full-results)
12. [Limitations](#limitations)
13. [Troubleshooting](#troubleshooting)
14. [Data and privacy](#data-and-privacy)
15. [Citation](#citation)
16. [License and acknowledgements](#license-and-acknowledgements)

---

## Overview

Speech-based dementia screening is usually framed as a direct mapping from a recording to a diagnostic label, or as multi-task learning where a cognitive score such as the Mini-Mental State Examination (MMSE) is predicted **in parallel** with the diagnosis. In clinical practice the order is different: cognitive state is assessed first, and that assessment informs the diagnosis.

This project tests whether a speech model benefits from the same structure. We call it **cognitive conditioning**:

1. A multimodal encoder (Wav2Vec2 for audio, Bio_ClinicalBERT for the transcript, fused by audio-to-text cross-attention) produces a participant-level representation `z`.
2. An MMSE head estimates the MMSE score `m̂` and a 32-dimensional latent cognitive embedding `h`.
3. The dementia and severity classifiers receive **`[z ; h ; m̂/30]`**, so the diagnosis is conditioned on the speech-estimated cognitive state, rather than predicted in parallel with it.

The repository contains:

| Part | Description |
|------|-------------|
| `Major_Dementia.ipynb` | Data preparation, training, ablation study (10 variants), evaluation, external evaluation and standalone inference. |
| `backend/` | FastAPI service: audio upload, CrisperWhisper transcription, model inference, biomarker extraction, explainability. |
| `frontend/` | Next.js web app for recording or uploading audio and viewing the patient-level report. |

### The three formulations compared

```
(a) Diagnosis only (F0)        x -> z -> y_hat
(b) Parallel multi-task (F1)   x -> z -> y_hat
                                    \-> m_hat            (m_hat never reaches the diagnosis head)
(c) Cognitive conditioning (F2, ours)
                               x -> z -> (m_hat, h) -> [z ; h ; m_hat/30] -> y_hat, s_hat
```

The formulation is deliberately **soft**: the fused representation `z` is retained in the classifier input, so the classifier is not forced to rely on the cognitive estimate when it is uninformative.

---

## Key results

All results below are on a participant-level held-out test set of 100 recordings (56 dementia, 44 control) from the DementiaBank Pitt Corpus (Cookie Theft task), evaluated once per variant.

| Formulation | Balanced accuracy | ROC-AUC | MMSE RMSE |
|-------------|------------------:|--------:|----------:|
| Diagnosis only (A5) | 79.87% | 0.838 | n/a |
| Parallel multi-task (A6) | 83.12% | 0.871 | 4.89 |
| No feedback path (A8) | 85.71% | 0.889 | 5.02 |
| **Cognitive conditioning (A7, ours)** | **87.91%** | **0.912** | **3.55** |

Headline numbers for the conditioned model (A7):

- Accuracy 87.00% (95% CI 79.0 to 92.2), sensitivity 80.36% (45/56), specificity 95.45% (42/44), dementia F1 0.874.
- Feeding the cognitive estimate forward lowers MMSE error from **5.02 to 3.55** (a 29% reduction relative to the same model without the feedback path).
- Across the nine variants that estimate MMSE, lower MMSE error tended to accompany better diagnosis (Spearman rho = -0.60, descriptive only).
- **External evaluation without retraining** (CrisperWhisper transcripts):
  - DementiaBank LU: 83.02% accuracy (CI 70.8 to 90.8), 85.19% sensitivity (23/27), 80.77% specificity (21/26).
  - Kempler subset: 6/6 dementia recordings correct (dementia-only, so specificity is undefined).

These are single-run results on a small test set. Differences of a few percentage points between variants are within the uncertainty of a 100-recording test set (see [Limitations](#limitations)).

---

## Method

### Data

| | DementiaBank Pitt | DementiaBank LU | Kempler subset |
|---|---|---|---|
| Role | Development and held-out test | External test | External test |
| Recordings | 552 | 53 | 6 |
| Dementia / control | 308 / 244 | 27 / 26 | 6 / 0 |
| Transcript source | Manual (CHAT) | CrisperWhisper | CrisperWhisper |
| MMSE available | Yes | No | No |

**Partitioning.** All splits are at the **participant level** (no participant appears in more than one partition, asserted in code). The 552 recordings were split, stratified by diagnosis, into a development pool of 452 recordings (252 dementia, 200 control) and a held-out test set of 100 recordings (56 dementia, 44 control). The test set was set aside before model development and was not used for configuration selection, training, early stopping, threshold selection or calibration.

### Preprocessing

- **Audio:** resampled to 16 kHz, passed to Wav2Vec2 as raw waveforms. Recordings are truncated to the first 75 s during training and standard inference; shorter ones are zero-padded within a mini-batch.
- **Transcripts:** participant utterances extracted from CHAT files, tokenised to at most 512 tokens (truncation from the end, attention-masked padding).
- **External transcripts:** generated with [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper), which retains fillers, repetitions and other disfluencies that conventional ASR tends to remove.

### Architecture

```
Raw audio (<= 75 s) ---> Wav2Vec2 (LoRA r=8) ---> conv. down-sampling, 768->256 --\
                                                                                   +--> audio-to-text cross-attention
Transcript (<= 512 tok) -> Bio_ClinicalBERT (LoRA r=8) -> projection 768->256 ----/     (4 heads; Q = audio, K,V = text)
                                                                                         |
                                  residual + layer norm -> feed-forward fusion -> mean pooling
                                                                                         v
                                                                       z in R^256 (multimodal representation)
                                                                                         |
                                                    MMSE head 256 -> 128 -> 64 -> 32 -> 1
                                                       h in R^32 (cognitive embedding), m_hat (clipped to [0, 30])
                                                                                         |
                                                              z_h = [z ; h ; m_hat/30] in R^289
                                                                  /                               \
                                               Dementia head 289->128->64->1        Severity head 289->128->4
```

### Objective

```
L = L_cls + 0.5 * L_MMSE + 0.2 * L_sev
```

- `L_cls`: class-weighted binary cross-entropy with label smoothing (diagnosis stays the primary target).
- `L_MMSE`: SmoothL1 (delta = 1) on the 0 to 30 scale, masked for recordings without an MMSE annotation.
- `L_sev`: masked cross-entropy over four MMSE-defined severity classes.

| Severity | Normal | Mild | Moderate | Severe |
|---|---|---|---|---|
| MMSE range | >= 24 | 18 to 23 | 10 to 17 | < 10 |

Severity is derived from MMSE, so it acts as a coarse, ordinal view of the same cognitive construct, not an independent clinical staging.

### Training configuration

| Component | Setting |
|---|---|
| Encoders | `facebook/wav2vec2-base`, `emilyalsentzer/Bio_ClinicalBERT` |
| Adaptation | LoRA r=8, alpha=16, dropout 0.1, no bias. Wav2Vec2: `q_proj, k_proj, v_proj, out_proj`. Bio_ClinicalBERT: `query, value`. Final block of each encoder unfrozen. |
| Parameters | 210,483,464 total, 7,801,480 trainable (3.71%) |
| Optimiser | AdamW; lr 2e-4 (LoRA), 1e-3 (task layers); weight decay 1e-4 |
| Schedule | 10% linear warm-up, linear decay to 5% of the initial rate |
| Training | Batch size 4, up to 50 epochs, early stopping on validation balanced accuracy (patience 7) |
| Stabilisation | Gradient clipping (norm 1.0), FP16 mixed precision, gradient checkpointing |
| Dropout | 0.2 in fusion FFN, 0.3 in prediction heads |

### Inference

- Dementia probabilities are calibrated with **temperature scaling** fitted on validation data.
- The decision threshold is chosen on **validation data** to maximise balanced accuracy (instead of 0.5). Each variant uses its own validation-selected threshold.
- Test-time augmentation (averaging logits over three random crops) is supported but **disabled** for all reported results.

---

## Repository structure

Items marked *(local)* are not stored in the Git repository (large files or restricted data) and must be provided by you; see [Data and checkpoints](#data-and-checkpoints).

```
Dementia/
|-- backend/                    FastAPI service
|   |-- main.py                 API entry point
|   |-- config.py               Model paths and runtime settings
|   |-- requirements.txt        Backend dependencies
|   |-- models/                 Model architecture
|   |-- routes/                 API routes
|   `-- services/               Inference, transcription, biomarkers
|-- frontend/                   Next.js web application
|-- Major_Dementia.ipynb        Training, ablation, evaluation, inference notebook
|-- kfold_checkpoints/          (local) trained checkpoints, e.g. dementia_mtl_FINAL.pth
|-- train/                      (local) DementiaBank Pitt audio + CHAT transcripts
|-- lu/                         (local) DementiaBank LU audio + transcripts
|-- .gitignore
`-- README.md
```

---

## Requirements

- Windows, macOS or Linux
- Python 3.10 or newer
- Node.js 18 or newer and npm (frontend only)
- FFmpeg available on `PATH` (needed for WebM/OGG decoding)
- NVIDIA GPU with a compatible PyTorch build is recommended for real inference and strongly recommended for training
- About 8 GB or more of free system memory for loading the model

---

## Data and checkpoints

### Data

The speech and transcript data come from **DementiaBank** (Pitt Corpus; LU and Kempler subsets) and are available to qualified researchers through the DementiaBank data-request process at <https://dementia.talkbank.org>, under its data-use agreement. **We do not have permission to redistribute the raw data, and it is not included in this repository.**

After obtaining access, place the data locally (for example under `train/` and `lu/`) and set `DATA_ROOT` in the notebook to that location.

### Model checkpoint

The backend expects the final conditioned model (variant **A7** in the paper) at:

```
kfold_checkpoints/dementia_mtl_FINAL.pth
```

> **TODO (maintainers):** state here where the checkpoint can be downloaded (release asset, Hugging Face Hub, Drive link), or that it must be regenerated by running the notebook through the "save final checkpoint" step.

If the checkpoint is missing, the backend starts with untrained weights and its predictions must **not** be used. Use mock mode for UI development (see below).

---

## Reproducing the paper (notebook)

Open `Major_Dementia.ipynb` in VS Code or Jupyter (a CUDA-enabled environment is strongly recommended; training is computationally expensive) and run the sections in order:

1. Install notebook dependencies and verify CUDA.
2. Set `DATA_ROOT` to the dataset location.
3. Parse CHAT transcripts and build the combined metadata.
4. Build the multimodal dataframe and the **participant-level** train / held-out split.
5. Define the model architecture and run training.
6. Run the ablation study (variants A1 to A10c, see [below](#model-variants-and-full-results)) and/or five-fold participant-grouped cross-validation for LoRA configuration selection.
7. Save the final checkpoint to `kfold_checkpoints/dementia_mtl_FINAL.pth`.
8. Run held-out evaluation, external evaluation (LU, Kempler), the explainability section, or standalone inference.

Model weights for the two pretrained encoders are downloaded from the Hugging Face Hub on first use.

> **Note on reproducibility.** Reported numbers come from a single run on a single held-out split, not repeated seeds. Expect small run-to-run differences.

---

## Running the web application

The application wraps the trained model in a FastAPI backend and a Next.js frontend. The live pipeline is:

```
Audio upload
  -> CrisperWhisper transcription
  -> Wav2Vec2 audio encoding + Bio_ClinicalBERT text encoding
  -> audio-to-text cross-attention fusion (z)
  -> MMSE estimation (m_hat, h)
  -> [z ; h ; m_hat/30] -> dementia classification + severity prediction
  -> speech biomarkers and evidence report
```

> The model was trained on **manual CHAT transcripts**, but the application transcribes uploads automatically with CrisperWhisper (as in the external LU/Kempler evaluation). The paper reports lower accuracy under this shift (87.00% in-corpus vs 83.02% on LU), and the effect of transcription errors on the MMSE estimate was not quantified separately.

### Backend

From the repository root, create and activate a virtual environment.

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Check that the service is running:

- Health check: <http://localhost:8000/api/health>
- Interactive API docs: <http://localhost:8000/docs>

#### Mock mode (no model required)

To run the API without loading the model (UI development, smoke tests):

```powershell
# Windows (PowerShell)
$env:MOCK_ANALYSIS_MODE = "true"
python -m uvicorn backend.main:app --reload --port 8000
```

```bash
# macOS / Linux
MOCK_ANALYSIS_MODE=true python -m uvicorn backend.main:app --reload --port 8000
```

Unset the variable (or set it to `false`) to use real inference. Mock responses are **not** model predictions.

#### Audio length

Models are trained with audio truncated to 75 s. The audio length actually used by the backend is set in `backend/config.py`.

> **TODO (maintainers):** confirm the backend's input limit (the previous README said 30 s, the paper says 75 s) and update this section so the two agree.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The frontend defaults to `http://localhost:8000` for the API. To use a different backend URL:

```powershell
# Windows (PowerShell)
$env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:8000"
npm run dev
```

```bash
# macOS / Linux
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

Other commands: `npm run lint`, `npm run build`, `npm start`.

---

## API reference

Analyse a recording with `curl`:

```bash
curl -X POST "http://localhost:8000/api/analyze" -F "audio=@path/to/recording.wav"
```

On Windows PowerShell use `curl.exe` and a backslash path.

Supported upload formats: WAV (simplest), and WebM/OGG when FFmpeg is installed. Recordings shorter than one second, or with no detected speech, are rejected.

The JSON response contains:

| Field | Contents |
|---|---|
| `prediction` | Label, calibrated dementia probability, decision threshold, confidence, decision |
| `mmse` | Estimated MMSE score on a 0 to 30 scale |
| `severity` | Normal, Mild, Moderate or Severe (MMSE-defined) |
| `transcription` | Transcript text and timestamped segments |
| `speech_biomarkers` | Pause, fluency, repetition, lexical and speech-rate features |
| `evidence` | Rule-based, threshold-based evidence items |
| `attribution` | Attention-based ranking and word-level occlusion results |
| `explainability` | Template-generated explanation and report text |
| `metadata` | Audio duration, model version, processing time |

---

## Post-hoc evidence layer

Each prediction can be accompanied by human-readable context. This layer runs **after** the forward pass and does not affect training or prediction.

- **Speech biomarkers** (rule-based, from waveform, transcript and word-level timestamps): pause ratio, speech rate, filled pauses, repetitions, vague references, and a fluency score. Items are ordered with fixed weights reflecting published associations; no composite score is computed or validated.
- **Attention ranking:** transcript words ranked by final-layer Bio_ClinicalBERT attention aggregated over heads (punctuation and special tokens excluded).
- **Word-level occlusion:** `delta p_j = p(D | X) - p(D | X without word j)`.
- **Report:** predicted class, calibrated probability, threshold, estimated MMSE, severity category and evidence items.

**Interpret with care.** Attention weights are correlational, not causal. In the paper's two worked external cases, attention concentrated on short, high-frequency words (e.g. *oh*, *too*, *I*) while occlusion found no single influential word. The generated explanations are template-based descriptions of statistical associations, **not clinical determinations**, and have not been validated against expert assessment.

The paper's patient-level cases also show that two recordings with very different speech profiles (pause ratio 79% vs 29%; 44.0 vs 20.8 words/min) received almost identical calibrated probabilities (87.8% vs 87.7%), which motivates a corpus-wide calibration analysis.

---

## Model variants and full results

Ten variants were trained under identical settings and evaluated once on the held-out test set (100 recordings; 56 dementia, 44 control). `t` is the validation-selected threshold.

| ID | Variant | t | Acc. | Bal. acc. | Sens. | Spec. | F1 | AUC | MMSE RMSE |
|----|---------|---:|-----:|----------:|------:|------:|---:|----:|----------:|
| A5 | Diagnosis only | 0.84 | 79% | 79.87% | 75.00% | 84.74% | 0.800 | 0.838 | n/a |
| A6 | Parallel multi-task | 0.58 | 82% | 83.12% | 78.57% | 87.66% | 0.830 | 0.871 | 4.89 |
| A8 | No feedback path | 0.91 | 85% | 85.71% | 82.14% | 89.29% | 0.852 | 0.889 | 5.02 |
| A9 | No severity objective | 0.58 | 81% | 81.82% | 76.79% | 86.85% | 0.814 | 0.852 | 4.47 |
| **A7** | **Full (conditioned)** | 0.90 | **87%** | **87.91%** | 80.36% | 95.45% | **0.874** | **0.912** | **3.55** |
| A1 | Audio only | 0.15 | 78% | 79.22% | 69.64% | 88.64% | 0.775 | 0.842 | 5.20 |
| A2 | Text only | 0.61 | 80% | 81.17% | 71.43% | 90.91% | 0.789 | 0.858 | 4.87 |
| A4 | Concatenation fusion | 0.18 | 85% | 85.71% | 78.57% | 92.86% | 0.846 | 0.892 | 4.24 |
| A10a | Frozen encoders | 0.90 | 84% | 85.06% | 78.57% | 91.56% | 0.838 | 0.881 | 5.56 |
| A10c | Full fine-tuning | 0.75 | 76% | 76.62% | 69.64% | 83.60% | 0.757 | 0.801 | 21.47 |

Variant groups:

- **Supervision (A5, A6, A8, A9, A7):** isolates the formulation, removing the feedback path and the severity objective one at a time.
- **Representation (A1, A2, A4):** modalities and fusion mechanism (audio only, text only, concatenation instead of cross-attention).
- **Adaptation (A10a, A10c):** frozen encoders vs full fine-tuning (LoRA with selective unfreezing is the default, A7).

Main findings from the ablation:

- **RQ1.** Conditioning outperformed diagnosis-only and parallel multi-task learning on every metric except sensitivity. The gain was concentrated in specificity (89.29% to 95.45% from A8 to A7), at the cost of one additional missed dementia case.
- **RQ2.** Adding the feedback path also improved the MMSE estimate itself (RMSE 5.02 to 3.55). Removing the severity objective hurt more than removing the feedback path.
- **RQ3.** Better fused representations gave better cognitive estimates and better diagnosis (audio only < text only < concatenation < cross-attention). Full fine-tuning of ~210M parameters on 552 recordings caused the MMSE head to collapse (RMSE 21.47), so parameter-efficient adaptation is effectively required here.
- **RQ4.** External transfer without retraining: 83.02% accuracy on DementiaBank LU.

### LoRA configuration selection (five-fold participant-grouped CV on the development pool)

| Configuration | Accuracy | Dementia F1 | MMSE RMSE |
|---|---:|---:|---:|
| No LoRA | 76.62% | 0.789 | 4.70 |
| Wav2Vec2 only, r=8 | 78.62% | 0.794 | 4.71 |
| Bio_ClinicalBERT only, r=8 | 80.97% | 0.828 | 4.33 |
| Both encoders, r=8 (selected) | 79.89% | 0.813 | 4.69 |
| Both encoders, r=32 | 78.25% | 0.786 | 4.69 |

---

## Limitations

- **Small, single test set.** All comparisons use one participant-level held-out set of 100 recordings, evaluated once per variant, without repeated seeds or splits. The sensitivity CI of the conditioned model spans roughly 68% to 89%, and paired significance tests were not performed. Differences of a few points between variants are within this uncertainty.
- **Single corpus and task.** Development used one English corpus and one elicitation task (Cookie Theft picture description). The Pitt cohort is imbalanced and is not age/sex-matched like the ADReSS subsets, so numbers are not directly comparable with challenge results.
- **Transcript mismatch.** Training and in-corpus testing use manual transcripts; deployment and external evaluation use automatic ones.
- **MMSE-derived severity.** Severity categories come from MMSE cut-offs, not independent clinical staging, and few participants fall in the lowest MMSE ranges.
- **No external validation of the MMSE estimate.** MMSE was unavailable for LU and Kempler, and Kempler has no controls.
- **Unvalidated evidence layer.** The explanations have not been checked against clinical judgement, calibration concerns rest on case studies rather than a corpus-wide analysis, and demographic subgroup performance was not examined.

This system is intended as a **screening aid**, not a diagnostic instrument.

---

## Troubleshooting

**Checkpoint not found.** Place `dementia_mtl_FINAL.pth` in `kfold_checkpoints/`. Without it the backend starts with untrained weights, and its predictions must not be used.

**CUDA or out-of-memory errors.** Confirm PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Use CPU only for smoke tests or mock mode; real inference with both transformer encoders is slow and memory-intensive on CPU.

**Audio decoder errors.** Install FFmpeg and make sure `ffmpeg -version` works in a new terminal. WAV is the simplest format for local testing.

**Frontend cannot reach the API.** Start the backend on port 8000 first, then confirm <http://localhost:8000/api/health> responds. Set `NEXT_PUBLIC_API_BASE_URL` if the API runs on another host or port.

---

## Data and privacy

- Do **not** commit participant audio, transcripts, cookies, credentials or any private dataset. Keep checkpoints and personally identifiable research data in controlled storage.
- Follow the consent and data-governance requirements of the dataset, including the [DementiaBank Ground Rules](https://talkbank.org/0share/index.html).
- This study used existing data accessed under an approved DementiaBank membership; no new data were collected.
- Recordings uploaded to the web application may contain sensitive health information. Do not deploy the app publicly without appropriate consent, security and retention controls.

---

## Citation


Please also cite the DementiaBank corpus if you use the data:

> Lanzi, A. M., Saylor, A. K., Fromm, D., Liu, H., MacWhinney, B., and Cohen, M. L. (2023). DementiaBank: Theoretical rationale, protocol, and illustrative analyses. *American Journal of Speech-Language Pathology*, 32, 426-438.


