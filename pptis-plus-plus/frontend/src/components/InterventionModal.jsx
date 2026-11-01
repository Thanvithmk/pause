const EXERCISES = [
  {
    exercise: "Slow Breathing",
    duration: "20–30 seconds",
    instructions: "Inhale through your nose for 4 seconds, hold for 2, exhale through your mouth for 6.",
  },
  {
    exercise: "5-4-3-2-1 Grounding",
    duration: "30–45 seconds",
    instructions: "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste or remember.",
  },
  {
    exercise: "Shoulder Release",
    duration: "15–20 seconds",
    instructions: "Shrug your shoulders up to your ears, hold, then drop them and unclench your jaw and hands.",
  },
  {
    exercise: "Pause and Reread",
    duration: "20 seconds",
    instructions: "Read the message once as if you were the recipient.",
  },
];

function InterventionModal({
  kind,
  analysis,
  remainingSeconds,
  onSendAnyway,
  onEdit,
  onCancel,
  onUseRewrite,
}) {
  if (!kind) {
    return null;
  }

  const suggested =
    analysis?.suggested_text || analysis?.rewrite?.suggested_text || analysis?.action?.suggested_text;
  const warning =
    analysis?.warning_message || analysis?.action?.warning_message || analysis?.action?.ui?.prompt;
  const exercises = analysis?.delayed_send?.exercises || EXERCISES;

  return (
    <div className="fixed inset-0 z-40 grid place-items-center bg-black/60 px-4">
      <div className="w-full max-w-lg rounded-2xl border border-slate-700 bg-panel p-6 space-y-4">
        {kind === "WARN" && (
          <>
            <h3 className="text-lg font-semibold text-warn">Warning</h3>
            <p className="text-sm text-slate-300">{warning}</p>
            <p className="text-xs text-slate-500">
              Final risk {Number(analysis?.final_risk || 0).toFixed(2)} · you can edit, cancel, or send anyway.
            </p>
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={onEdit} className="rounded-lg bg-slate-100 px-3 py-2 text-sm text-ink">
                Edit Message
              </button>
              <button type="button" onClick={onCancel} className="rounded-lg bg-slate-700 px-3 py-2 text-sm">
                Cancel
              </button>
              <button type="button" onClick={onSendAnyway} className="rounded-lg bg-warn px-3 py-2 text-sm text-ink">
                Send Anyway
              </button>
            </div>
          </>
        )}

        {kind === "REWRITE" && (
          <>
            <h3 className="text-lg font-semibold text-rewrite">Safer rewrite required</h3>
            <p className="text-sm text-slate-300">The original message cannot be sent.</p>
            {suggested ? (
              <blockquote className="rounded-xl bg-ink p-3 text-sm text-slate-200">{suggested}</blockquote>
            ) : (
              <p className="text-sm text-slate-400">No approved suggestion yet. Please edit the message yourself.</p>
            )}
            <div className="flex flex-wrap gap-2">
              {suggested && (
                <button
                  type="button"
                  onClick={() => onUseRewrite(suggested)}
                  className="rounded-lg bg-emerald-500 px-3 py-2 text-sm text-ink"
                >
                  Use Suggested Rewrite
                </button>
              )}
              <button type="button" onClick={onEdit} className="rounded-lg bg-slate-100 px-3 py-2 text-sm text-ink">
                Edit Myself
              </button>
              <button type="button" onClick={onCancel} className="rounded-lg bg-slate-700 px-3 py-2 text-sm">
                Cancel
              </button>
            </div>
          </>
        )}

        {kind === "DELAYED-SEND" && (
          <>
            <h3 className="text-lg font-semibold text-delay">Sending is locked</h3>
            <p className="text-sm text-slate-300">
              High-risk delay is active. There is no send-anyway option.
            </p>
            <p className="font-mono text-2xl text-delay">{remainingSeconds}s</p>
            <ul className="space-y-2 text-sm text-slate-300">
              {exercises.map((item) => (
                <li key={item.exercise} className="rounded-xl bg-ink p-3">
                  <p className="font-medium">
                    {item.exercise}{" "}
                    <span className="text-xs text-slate-500">
                      {item.duration || `${item.duration_seconds}s`}
                    </span>
                  </p>
                  <p className="text-slate-400">{item.instructions}</p>
                </li>
              ))}
            </ul>
            <p className="text-xs text-slate-500">
              After the countdown, edit the message and re-analyze. Send stays disabled until a new analysis permits it.
            </p>
            <button type="button" onClick={onEdit} className="rounded-lg bg-slate-100 px-3 py-2 text-sm text-ink">
              Edit Message
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default InterventionModal;
