import { useEffect, useRef, useState } from "react";
import { analyzeVoice } from "../services/api";

function VoiceInput({ onAnalyzed, conversationId, compact = false }) {
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [transcript, setTranscript] = useState("");
  const [levels, setLevels] = useState(Array(12).fill(8));
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const animationRef = useRef(null);
  const analyserRef = useRef(null);

  useEffect(() => () => cancelAnimationFrame(animationRef.current), []);

  function tick() {
    const analyser = analyserRef.current;
    if (analyser) {
      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(data);
      const step = Math.floor(data.length / 12) || 1;
      setLevels(
        Array.from({ length: 12 }, (_, index) =>
          Math.max(6, Math.round((data[index * step] || 0) / 6))
        )
      );
    }
    animationRef.current = requestAnimationFrame(tick);
  }

  async function startRecording() {
    setError("");
    setTranscript("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);
      analyserRef.current = analyser;
      animationRef.current = requestAnimationFrame(tick);

      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        cancelAnimationFrame(animationRef.current);
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        setLoading(true);
        try {
          const result = await analyzeVoice(blob, "recording.webm", conversationId);
          const text =
            result.original_text ||
            result.transcription?.text ||
            result.text ||
            "";
          setTranscript(text);
          onAnalyzed?.(result, text, "voice");
        } catch (err) {
          setError(err.response?.data?.message || "Could not analyze voice.");
        } finally {
          setLoading(false);
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch {
      setError("Microphone permission is required.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  if (compact) {
    return (
      <div className="relative">
        <button
          type="button"
          onClick={recording ? stopRecording : startRecording}
          disabled={loading}
          aria-label={recording ? "Stop recording" : "Record voice"}
          className={`grid h-10 w-10 place-items-center rounded-full ${recording ? "bg-rose-100 text-rose-500" : "text-slate-400"}`}
        >
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="9" y="3" width="6" height="11" rx="3" />
            <path d="M6 11a6 6 0 0 0 12 0M12 17v3" strokeLinecap="round" />
          </svg>
        </button>
        {error && <p className="absolute right-0 top-11 w-48 text-xs text-rose-500">{error}</p>}
      </div>
    );
  }

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 p-3">
      <div className="flex items-center gap-3">
        {!recording ? (
          <button
            type="button"
            onClick={startRecording}
            disabled={loading}
            className="rounded-full bg-delay px-4 py-2 text-sm font-medium"
          >
            Record
          </button>
        ) : (
          <button
            type="button"
            onClick={stopRecording}
            className="rounded-full bg-warn px-4 py-2 text-sm font-medium text-ink"
          >
            Stop
          </button>
        )}
        {recording && <span className="record-pulse h-3 w-3 rounded-full bg-delay" />}
        <span className="text-xs text-slate-400">
          {loading ? "Transcribing…" : recording ? "Recording" : "Voice input"}
        </span>
      </div>
      <div className="flex h-10 items-end gap-1">
        {levels.map((height, index) => (
          <span
            key={index}
            className="w-1 rounded-sm bg-emerald-400/80"
            style={{ height: `${height}px` }}
          />
        ))}
      </div>
      {transcript && <p className="text-sm text-slate-300">{transcript}</p>}
      {error && <p className="text-sm text-delay">{error}</p>}
    </div>
  );
}

export default VoiceInput;
