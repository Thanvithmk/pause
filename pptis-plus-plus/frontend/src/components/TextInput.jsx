function TextInput({
  text,
  onTextChange,
  onSend,
  onAnalyze,
  flowState,
  sendDisabled,
  sendLabel,
  analyzeDisabled,
  error,
}) {
  const analyzing = flowState === "ANALYZING";
  const sending = flowState === "SENDING";

  function handleChange(event) {
    onTextChange?.(event.target.value);
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSend?.();
      }}
      className="space-y-2"
    >
      <input
        value={text}
        onChange={handleChange}
        placeholder="Type your message..."
        className="w-full rounded-xl bg-ink px-4 py-3 outline-none ring-1 ring-slate-700 focus:ring-emerald-500"
      />
      <div className="flex gap-2">
        <button
          type="submit"
          disabled={sendDisabled || analyzing || sending}
          className="rounded-lg bg-slate-100 px-4 py-2 text-sm font-medium text-ink disabled:opacity-60"
        >
          {sending ? "Sending…" : sendLabel || "Send"}
        </button>
        <button
          type="button"
          onClick={onAnalyze}
          disabled={analyzeDisabled || analyzing || sending}
          className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-medium text-ink disabled:opacity-60"
        >
          {analyzing ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      {error && <p className="text-sm text-delay">{error}</p>}
    </form>
  );
}

export default TextInput;
