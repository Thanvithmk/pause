const VOICE_DEFERRED = "Voice transcription is currently deferred.";

function VoiceInput({ compact = false }) {
  if (compact) {
    return (
      <div className="relative">
        <button
          type="button"
          disabled
          aria-label={VOICE_DEFERRED}
          title={VOICE_DEFERRED}
          className="grid h-10 w-10 place-items-center rounded-full text-slate-300"
        >
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="9" y="3" width="6" height="11" rx="3" />
            <path d="M6 11a6 6 0 0 0 12 0M12 17v3" strokeLinecap="round" />
          </svg>
        </button>
        <p className="sr-only">{VOICE_DEFERRED}</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-200 p-3">
      <p className="text-sm font-medium text-slate-600">Voice input</p>
      <p className="mt-1 text-sm text-slate-500">{VOICE_DEFERRED}</p>
    </div>
  );
}

export default VoiceInput;
