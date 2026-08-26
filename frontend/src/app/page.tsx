"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { Mic, Square, RefreshCw, Upload, FileText, Activity, Brain, CheckCircle, AlertTriangle, ShieldCheck } from "lucide-react";
import picture from "./image.png";

type AppState = "INTRO" | "CONSENT" | "MIC_CHECK" | "READY" | "RECORDING" | "REVIEW" | "PROCESSING" | "COMPLETE" | "ERROR";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type Attribution = { word: string; delta: number };
type EvidenceItem = { rank: string; feature: string; finding: string; clinical_relevance: string; severity_weight: number };
type AnalysisResult = {
  prediction: { label: string; probability: number; threshold: number; confidence: number; positive: boolean; decision: string };
  mmse: { prediction: number; min: number; max: number };
  severity: { label: string };
  transcription: { text: string; segments: Array<{ start: number; end: number; text: string }> };
  speech_biomarkers: Record<string, number>;
  evidence: EvidenceItem[];
  attribution: Attribution[];
  explainability: { clinical_explanation: string; evidence_ranking: EvidenceItem[]; report_text: string };
};

function audioBufferToWav(buffer: AudioBuffer): Blob {
  const channelCount = Math.min(buffer.numberOfChannels, 2);
  const frameCount = buffer.length;
  const bytesPerSample = 2;
  const dataSize = frameCount * channelCount * bytesPerSample;
  const wav = new ArrayBuffer(44 + dataSize);
  const view = new DataView(wav);

  const writeText = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * channelCount * bytesPerSample, true);
  view.setUint16(32, channelCount * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[frame]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += bytesPerSample;
    }
  }

  return new Blob([wav], { type: "audio/wav" });
}

async function convertRecordingToWav(blob: Blob): Promise<Blob> {
  const audioContext = new AudioContext();
  try {
    const decoded = await audioContext.decodeAudioData(await blob.arrayBuffer());
    return audioBufferToWav(decoded);
  } finally {
    await audioContext.close();
  }
}

export default function Home() {
  const [state, setState] = useState<AppState>("INTRO");
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [processingStage, setProcessingStage] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const MAX_RECORDING_TIME = 30;

  const processingStages = [
    "Audio received",
    "Transcribing speech",
    "Extracting word timestamps",
    "Encoding transcript",
    "Encoding audio",
    "Fusing audio and language representations",
    "Running dementia prediction",
    "Estimating MMSE and severity",
    "Running model attribution",
    "Generating speech evidence",
    "Preparing final report"
  ];

  const stopTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current);
  };

  useEffect(() => {
    return () => stopTimer();
  }, []);

  const startTimer = () => {
    setRecordingTime(0);
    timerRef.current = setInterval(() => {
      setRecordingTime((prev) => {
        if (prev >= MAX_RECORDING_TIME - 1) {
          stopRecording();
          return MAX_RECORDING_TIME;
        }
        return prev + 1;
      });
    }, 1000);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"]
        .find((type) => MediaRecorder.isTypeSupported(type)) || "";
      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType || "audio/webm",
        });
        setAudioBlob(blob);
        setAudioURL(URL.createObjectURL(blob));
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setState("RECORDING");
      startTimer();
    } catch {
      setErrorMsg("Microphone permission denied or unavailable.");
      setState("ERROR");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      stopTimer();
      setState("REVIEW");
    }
  };

  const submitAnalysis = async () => {
    if (!audioBlob) return;
    setState("PROCESSING");
    
    // Simulate multi-stage progress
    let currentStage = 0;
    const progressInterval = setInterval(() => {
      currentStage++;
      if (currentStage < processingStages.length) {
        setProcessingStage(currentStage);
      }
    }, 2000);

    const formData = new FormData();
    try {
      const wavBlob = await convertRecordingToWav(audioBlob);
      formData.append("audio", wavBlob, "recording.wav");
      const res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });
      
      const data = await res.json();
      clearInterval(progressInterval);
      
      if (!res.ok || data.status === "error") {
        throw new Error(data.detail || "Analysis failed");
      }
      
      setAnalysisResult(data);
      setState("COMPLETE");
    } catch (err: unknown) {
      clearInterval(progressInterval);
      setErrorMsg(err instanceof Error ? err.message : "Failed to connect to backend");
      setState("ERROR");
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 p-8 flex flex-col items-center justify-center font-sans">
      <div className="max-w-4xl w-full">
        
        {/* Header */}
        <header className="mb-10 text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Brain className="w-10 h-10 text-blue-600" />
            <h1 className="text-3xl font-light tracking-tight">Speech Analysis Prototype</h1>
          </div>
          <p className="text-slate-500">AI-assisted research and screening system</p>
        </header>

        {/* Content Panel */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 min-h-[500px] flex flex-col transition-all duration-300">
          
          {state === "INTRO" && (
            <div className="flex flex-col items-center text-center space-y-6 flex-grow justify-center animate-in fade-in zoom-in duration-500">
              <ShieldCheck className="w-16 h-16 text-blue-500 mb-4" />
              <h2 className="text-2xl font-semibold">Welcome to the Assessment</h2>
              <p className="text-lg text-slate-600 max-w-lg">
                This prototype will ask you to describe a picture for up to 30 seconds. The system will analyze your speech patterns using our advanced research model.
              </p>
              <button 
                onClick={() => setState("CONSENT")}
                className="mt-8 px-8 py-4 bg-blue-600 text-white rounded-xl text-lg font-medium hover:bg-blue-700 transition-colors shadow-sm active:scale-95"
              >
                Begin Assessment
              </button>
            </div>
          )}

          {state === "CONSENT" && (
            <div className="flex flex-col space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
              <h2 className="text-2xl font-semibold border-b pb-4">Privacy & Consent</h2>
              <div className="bg-slate-50 p-6 rounded-xl text-slate-700 space-y-4">
                <p><strong>1. Temporary Processing:</strong> Your audio is processed temporarily and will not be permanently stored on our servers.</p>
                <p><strong>2. Research Tool:</strong> This is a research prototype. It is not a medical device and does not provide clinical diagnoses.</p>
                <p><strong>3. Audio Requirement:</strong> You will need to provide microphone access for the duration of this test.</p>
              </div>
              <div className="flex justify-end gap-4 mt-8 pt-4">
                <button onClick={() => setState("INTRO")} className="px-6 py-3 text-slate-500 hover:bg-slate-100 rounded-xl transition-colors">Cancel</button>
                <button onClick={() => setState("MIC_CHECK")} className="px-8 py-3 bg-blue-600 text-white rounded-xl font-medium shadow-sm hover:bg-blue-700 transition-colors">I Agree, Continue</button>
              </div>
            </div>
          )}

          {state === "MIC_CHECK" && (
            <div className="flex flex-col items-center justify-center space-y-8 flex-grow animate-in fade-in duration-300">
              <div className="w-24 h-24 bg-blue-50 rounded-full flex items-center justify-center">
                <Mic className="w-10 h-10 text-blue-600" />
              </div>
              <h2 className="text-2xl font-semibold">Microphone Access</h2>
              <p className="text-slate-600 text-center max-w-md">We need access to your microphone to record your response. Please click below to allow access.</p>
              <button onClick={() => setState("READY")} className="px-8 py-4 bg-blue-600 text-white rounded-xl text-lg font-medium shadow-sm hover:bg-blue-700 transition-colors">
                Enable Microphone
              </button>
            </div>
          )}

          {state === "READY" && (
            <div className="flex flex-col items-center space-y-8 flex-grow animate-in fade-in duration-300">
              <h2 className="text-2xl font-semibold">Picture Description Task</h2>
              
              <div className="w-full max-w-2xl overflow-hidden rounded-2xl border border-slate-200 bg-slate-100 shadow-sm">
                <Image
                  src={picture}
                  alt="A kitchen scene with a child reaching into a cookie jar while a woman washes dishes"
                  className="h-auto max-h-[28rem] w-full object-contain"
                  priority
                />
              </div>
              
              <div className="bg-blue-50 border border-blue-100 p-6 rounded-xl text-center max-w-2xl w-full">
                <p className="text-xl font-medium text-blue-900 mb-2">Please describe everything you see happening in this picture.</p>
                <p className="text-blue-700">Speak naturally and describe as much as you can.</p>
              </div>

              <button onClick={startRecording} className="group relative flex items-center justify-center w-24 h-24 bg-red-500 rounded-full hover:bg-red-600 transition-all shadow-lg active:scale-95">
                <Mic className="w-10 h-10 text-white" />
                <span className="absolute -bottom-10 text-red-600 font-medium opacity-0 group-hover:opacity-100 transition-opacity">Record</span>
              </button>
            </div>
          )}

          {state === "RECORDING" && (
            <div className="flex flex-col items-center justify-center space-y-12 flex-grow">
              <div className="relative flex items-center justify-center">
                <div className="absolute w-32 h-32 bg-red-100 rounded-full animate-ping"></div>
                <div className="relative w-24 h-24 bg-red-500 rounded-full flex items-center justify-center shadow-lg">
                  <Mic className="w-10 h-10 text-white animate-pulse" />
                </div>
              </div>
              
              <div className="text-center">
                <p className="text-3xl font-mono text-slate-800 tracking-wider">
                  00:{recordingTime.toString().padStart(2, '0')} <span className="text-slate-400">/ 00:30</span>
                </p>
                <p className="text-red-500 mt-2 font-medium">Recording in progress...</p>
              </div>

              <button onClick={stopRecording} className="flex items-center gap-2 px-8 py-4 bg-slate-900 text-white rounded-xl text-lg font-medium hover:bg-slate-800 transition-colors shadow-sm">
                <Square className="w-5 h-5" /> Stop Recording
              </button>
            </div>
          )}

          {state === "REVIEW" && (
            <div className="flex flex-col items-center justify-center space-y-8 flex-grow animate-in fade-in duration-300">
              <CheckCircle className="w-16 h-16 text-green-500" />
              <h2 className="text-2xl font-semibold">Recording Complete</h2>
              
              {audioURL && (
                <div className="bg-slate-50 p-6 rounded-2xl w-full max-w-md border border-slate-200 shadow-sm">
                  <audio controls src={audioURL} className="w-full" />
                </div>
              )}
              
              <div className="flex gap-4 w-full max-w-md">
                <button onClick={() => setState("READY")} className="flex-1 flex items-center justify-center gap-2 px-6 py-4 border border-slate-300 text-slate-700 rounded-xl font-medium hover:bg-slate-50 transition-colors">
                  <RefreshCw className="w-5 h-5" /> Retake
                </button>
                <button onClick={submitAnalysis} className="flex-1 flex items-center justify-center gap-2 px-6 py-4 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors shadow-sm">
                  <Upload className="w-5 h-5" /> Analyze
                </button>
              </div>
            </div>
          )}

          {state === "PROCESSING" && (
            <div className="flex flex-col items-center justify-center space-y-8 flex-grow">
              <div className="relative">
                <div className="w-20 h-20 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-blue-600" />
                </div>
              </div>
              <div className="text-center space-y-2">
                <h2 className="text-2xl font-semibold">Analyzing Speech</h2>
                <p className="text-slate-500 min-h-[1.5rem] transition-all">
                  {processingStages[processingStage]}...
                </p>
              </div>
            </div>
          )}

          {state === "ERROR" && (
            <div className="flex flex-col items-center justify-center space-y-6 flex-grow">
              <AlertTriangle className="w-16 h-16 text-red-500" />
              <h2 className="text-2xl font-semibold text-slate-800">Something went wrong</h2>
              <p className="text-red-600 bg-red-50 p-4 rounded-xl border border-red-100 max-w-lg text-center">{errorMsg}</p>
              <button onClick={() => setState("READY")} className="mt-4 px-8 py-3 bg-slate-900 text-white rounded-xl font-medium hover:bg-slate-800">
                Try Again
              </button>
            </div>
          )}

          {state === "COMPLETE" && analysisResult && (
            <div className="flex flex-col w-full animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex items-center justify-between mb-8 border-b pb-4">
                <h2 className="text-2xl font-semibold text-slate-800">Analysis Report</h2>
                <button onClick={() => setState("READY")} className="px-4 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg font-medium transition-colors">Start New</button>
              </div>

              <div className={`mb-8 rounded-2xl border p-6 ${analysisResult.prediction.positive ? "border-red-200 bg-red-50" : "border-green-200 bg-green-50"}`}>
                <p className={`text-sm font-semibold uppercase tracking-wider ${analysisResult.prediction.positive ? "text-red-700" : "text-green-700"}`}>Screening decision</p>
                <p className={`mt-2 text-2xl font-bold ${analysisResult.prediction.positive ? "text-red-900" : "text-green-900"}`}>
                  {analysisResult.prediction.positive ? "Positive" : "Negative"}
                </p>
                <p className="mt-2 text-sm text-slate-700">{analysisResult.prediction.decision}</p>
              </div>

              {/* Model Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="col-span-1 md:col-span-1 bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-2xl border border-blue-100">
                  <p className="text-sm font-semibold text-blue-600 uppercase tracking-wider mb-2">Model Result</p>
                  <div className="text-3xl font-bold text-slate-900 mb-1">
                    {(analysisResult.prediction.probability * 100).toFixed(0)}%
                  </div>
                  <p className="text-slate-600 text-sm">Model-estimated probability</p>
                </div>
                
                <div className="col-span-1 bg-slate-50 p-6 rounded-2xl border border-slate-200">
                  <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Estimated MMSE</p>
                  <div className="text-3xl font-bold text-slate-900 mb-1">
                    {analysisResult.mmse.prediction} <span className="text-lg text-slate-400 font-medium">/ 30</span>
                  </div>
                  <p className="text-slate-600 text-sm">AI estimate, not clinical</p>
                </div>

                <div className="col-span-1 bg-slate-50 p-6 rounded-2xl border border-slate-200">
                  <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Severity</p>
                  <div className="text-3xl font-bold text-slate-900 mb-1">
                    {analysisResult.severity.label}
                  </div>
                  <p className="text-slate-600 text-sm">Model-estimated category</p>
                </div>
              </div>

              {/* Biomarkers */}
              <div className="mb-8">
                <h3 className="text-lg font-semibold mb-4 text-slate-800">Speech Characteristics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <p className="text-xs text-slate-500 uppercase font-medium">Speech Rate</p>
                    <p className="text-xl font-semibold mt-1">{analysisResult.speech_biomarkers.speech_rate} <span className="text-sm font-normal text-slate-400">WPM</span></p>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <p className="text-xs text-slate-500 uppercase font-medium">Pause Ratio</p>
                    <p className="text-xl font-semibold mt-1">{(analysisResult.speech_biomarkers.pause_ratio * 100).toFixed(0)}%</p>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <p className="text-xs text-slate-500 uppercase font-medium">Fluency Score</p>
                    <p className="text-xl font-semibold mt-1">{analysisResult.speech_biomarkers.fluency_score} <span className="text-sm font-normal text-slate-400">/100</span></p>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <p className="text-xs text-slate-500 uppercase font-medium">Lexical Diversity</p>
                    <p className="text-xl font-semibold mt-1">{analysisResult.speech_biomarkers.lexical_diversity}</p>
                  </div>
                </div>
              </div>

              {/* Attribution Chart */}
              {analysisResult.attribution && analysisResult.attribution.length > 0 && (
                <div className="mb-8 p-6 bg-slate-50 rounded-2xl border border-slate-200">
                  <h3 className="text-lg font-semibold mb-4 text-slate-800">Model Attribution</h3>
                  <p className="text-sm text-slate-500 mb-6">The model&apos;s prediction changed most when these transcript elements were removed:</p>
                  
                  <div className="space-y-3">
                    {analysisResult.attribution.map((attr: Attribution, i: number) => (
                      <div key={i} className="flex items-center">
                        <div className="w-32 text-right pr-4 font-medium text-slate-700">&quot;{attr.word}&quot;</div>
                        <div className="flex-1 bg-slate-200 rounded-full h-4 overflow-hidden">
                          <div 
                            className="bg-blue-500 h-full rounded-full" 
                            style={{ width: `${Math.min(100, Math.max(0, attr.delta * 500))}%` }} 
                          ></div>
                        </div>
                        <div className="w-24 text-right pl-4 text-sm font-medium text-blue-600">
                          +{attr.delta.toFixed(3)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Evidence Cards */}
              {analysisResult.evidence && analysisResult.evidence.length > 0 && (
                <div className="mb-8">
                  <h3 className="text-lg font-semibold mb-4 text-slate-800">Why the model produced this result</h3>
                  <div className="flex flex-wrap gap-3">
                    {analysisResult.evidence.map((ev: EvidenceItem, i: number) => (
                      <div key={i} className="w-full rounded-xl border border-indigo-100 bg-indigo-50 p-4 text-indigo-950">
                        <div className="flex flex-wrap items-center gap-2 font-semibold"><span>{ev.rank}</span><span>{ev.feature}</span></div>
                        <p className="mt-1 text-sm">{ev.finding}</p>
                        <p className="mt-1 text-xs text-indigo-700">{ev.clinical_relevance}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {analysisResult.explainability && (
                <div className="mb-8 space-y-4">
                  <h3 className="text-lg font-semibold text-slate-800">Clinical Explainability</h3>
                  <div className="rounded-xl border border-slate-200 bg-white p-6 text-slate-700 leading-relaxed">
                    {analysisResult.explainability.clinical_explanation}
                  </div>
                  <details className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                    <summary className="cursor-pointer font-semibold text-slate-800">View full assessment report</summary>
                    <pre className="mt-4 whitespace-pre-wrap text-sm leading-relaxed text-slate-700">{analysisResult.explainability.report_text}</pre>
                  </details>
                </div>
              )}

              {/* Transcript Viewer */}
              <div className="mb-10">
                <h3 className="text-lg font-semibold mb-4 text-slate-800">Automatically Generated Transcript</h3>
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 text-slate-700 leading-relaxed font-serif">
                  &quot;{analysisResult.transcription.text}&quot;
                </div>
              </div>

              {/* Disclaimer */}
              <div className="mt-8 pt-6 border-t border-slate-200">
                <div className="bg-amber-50 border border-amber-200 p-4 rounded-xl flex gap-4">
                  <AlertTriangle className="w-6 h-6 text-amber-500 flex-shrink-0" />
                  <p className="text-sm text-amber-800">
                    <strong>Important Notice:</strong> This is an AI-assisted research/screening system and does not provide a medical diagnosis. This result should not replace evaluation by a qualified healthcare professional.
                  </p>
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </main>
  );
}
