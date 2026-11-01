import { useEffect, useRef, useState } from "react";
import VoiceInput from "./VoiceInput";
import AnalysisResults from "./AnalysisResults";
import { useShellBand } from "./AppShell";
import { OPEN_CONVERSATION_KEY } from "./MessageHistory";
import {
  analyzeText,
  createConversation,
  getConversation,
  getDelayedSendStatus,
  listConversations,
  sendMessage,
} from "../services/api";
import { formatClock } from "../utils/time";

const FLOW = {
  IDLE: "IDLE",
  ANALYZING: "ANALYZING",
  ALLOW: "ALLOW",
  WARNING: "WARNING",
  REWRITE_REQUIRED: "REWRITE_REQUIRED",
  DELAYED_SEND: "DELAYED_SEND",
  READY_TO_SEND: "READY_TO_SEND",
  SENDING: "SENDING",
  SENT: "SENT",
  ERROR: "ERROR",
};

function flowFromDecision(decision) {
  if (decision === "ALLOW") return FLOW.ALLOW;
  if (decision === "WARN") return FLOW.WARNING;
  if (decision === "REWRITE") return FLOW.REWRITE_REQUIRED;
  if (decision === "DELAYED-SEND" || decision === "DELAYED_SEND") return FLOW.DELAYED_SEND;
  return FLOW.IDLE;
}

function Chat({ user }) {
  const [messages, setMessages] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [text, setText] = useState("");
  const [flowState, setFlowState] = useState(FLOW.IDLE);
  const [conversationId, setConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [error, setError] = useState("");
  const [warnHidden, setWarnHidden] = useState(false);
  const [activeExercise, setActiveExercise] = useState(null);
  const [copied, setCopied] = useState(false);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  const sendingRef = useRef(false);
  const analyzedTextRef = useRef("");
  const { setBand } = useShellBand();

  function mapThread(thread) {
    return (thread.messages || []).map((item) => {
      const stored = item.message || {};
      const risk = item.decision || {};
      const score = typeof risk.final_risk === "number" ? risk.final_risk : null;
      return {
        id: `hist-${stored.message_id}`,
        role: "user",
        text: stored.current_text || stored.original_text || "",
        at: formatClock(stored.timestamp),
        decision: risk.decision || null,
        riskScore: score,
        scoreFailed: !risk.decision && score === null,
        messageId: stored.message_id,
        sendStatus: stored.send_status || "draft",
      };
    });
  }

  async function loadThread(id) {
    if (!id) {
      setMessages([]);
      return;
    }
    const thread = await getConversation(id);
    setMessages(mapThread(thread));
  }

  useEffect(() => {
    let cancelled = false;
    async function restore() {
      try {
        const listed = await listConversations();
        const rows = listed.conversations || [];
        if (cancelled) {
          return;
        }
        setConversations(rows);
        const requested = Number(sessionStorage.getItem(OPEN_CONVERSATION_KEY));
        sessionStorage.removeItem(OPEN_CONVERSATION_KEY);
        const latest = rows.find((row) => row.conversation_id === requested) || rows[0];
        if (!latest) {
          return;
        }
        setConversationId(latest.conversation_id);
        await loadThread(latest.conversation_id);
      } catch {
        /* keep empty until the user analyzes */
      }
    }
    restore();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSelectConversation(event) {
    const nextId = Number(event.target.value);
    setConversationId(nextId);
    setAnalysis(null);
    setFlowState(FLOW.IDLE);
    setWarnHidden(false);
    setText("");
    analyzedTextRef.current = "";
    try {
      await loadThread(nextId);
    } catch {
      setError("Could not load that conversation.");
    }
  }

  async function handleNewConversation() {
    try {
      const created = await createConversation();
      const row = created.conversation || created;
      setConversations((current) => [row, ...current]);
      setConversationId(row.conversation_id);
      setMessages([]);
      setAnalysis(null);
      setFlowState(FLOW.IDLE);
      setWarnHidden(false);
      setText("");
      analyzedTextRef.current = "";
    } catch {
      setError("Could not start a new conversation.");
    }
  }

  useEffect(() => {
    if (flowState !== FLOW.DELAYED_SEND || !analysis?.message_id) {
      return undefined;
    }
    let cancelled = false;
    async function poll() {
      try {
        const status = await getDelayedSendStatus(analysis.message_id);
        if (!cancelled) {
          setRemainingSeconds(status.remaining_seconds || 0);
        }
      } catch {
        /* keep last countdown */
      }
    }
    poll();
    const timer = setInterval(() => {
      setRemainingSeconds((value) => Math.max(0, value - 1));
      poll();
    }, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [flowState, analysis?.message_id]);

  useEffect(() => {
    const decision = analysis?.decision;
    if (decision === "ALLOW" || decision === "WARN" || decision === "REWRITE" || decision === "DELAYED-SEND") {
      setBand(decision);
    } else {
      setBand(null);
    }
    return () => setBand(null);
  }, [analysis?.decision, setBand]);

  function pushMessage(role, body, extra = {}) {
    setMessages((current) => [
      ...current,
      {
        id: `${Date.now()}-${role}-${Math.random()}`,
        role,
        text: body,
        at: formatClock(new Date()),
        ...extra,
      },
    ]);
  }

  function applyAnalysis(result, sourceText) {
    setAnalysis(result);
    setConversationId(result.conversation_id);
    analyzedTextRef.current = sourceText;
    const decision = result?.decision || result?.risk?.decision;
    const next = flowFromDecision(decision);
    setFlowState(next);
    setWarnHidden(false);
    setActiveExercise(null);
    if (decision === "DELAYED-SEND") {
      setRemainingSeconds(result?.delayed_send?.countdown_seconds || 60);
    }
    return result;
  }

  function handleTextChange(value) {
    setText(value);
    if (analyzedTextRef.current && value.trim() !== analyzedTextRef.current) {
      analyzedTextRef.current = "";
      setAnalysis(null);
      setFlowState(FLOW.IDLE);
      setWarnHidden(false);
      setError("");
    }
  }

  async function runAnalyze(sourceText = text.trim()) {
    if (!sourceText) {
      setError("Type a message first.");
      setFlowState(FLOW.ERROR);
      return null;
    }
    setFlowState(FLOW.ANALYZING);
    setError("");
    const pendingId = `pending-${Date.now()}`;
    pushMessage("user", sourceText, { source: "text", pending: true, loadingScore: true, id: pendingId });
    try {
      const result = await analyzeText(sourceText, conversationId);
      const score = typeof result.risk_score === "number" ? result.risk_score : result.final_risk;
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingId || (item.pending && item.text === sourceText)
            ? {
                ...item,
                pending: false,
                loadingScore: false,
                messageId: result.message_id,
                decision: result.decision,
                riskScore: typeof score === "number" ? score : null,
                scoreFailed: typeof score !== "number",
              }
            : item
        )
      );
      return applyAnalysis(result, sourceText);
    } catch (err) {
      setFlowState(FLOW.ERROR);
      setError(err.response?.data?.message || "Could not analyze text.");
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingId ? { ...item, pending: false, loadingScore: false, scoreFailed: true } : item
        )
      );
      return null;
    }
  }

  async function runSend(userAction = "send", result = analysis) {
    if (sendingRef.current) {
      return;
    }
    const payload = result || analysis;
    const current = text.trim();
    if (!payload?.message_id || current !== (analyzedTextRef.current || current)) {
      const fresh = await runAnalyze(current);
      if (!fresh) {
        return;
      }
      if (fresh.decision !== "ALLOW" && userAction !== "send_anyway") {
        return;
      }
      return runSend(userAction, fresh);
    }
    sendingRef.current = true;
    setFlowState(FLOW.SENDING);
    setError("");
    try {
      const sent = await sendMessage({
        messageId: payload.message_id,
        conversationId: payload.conversation_id || conversationId,
        text: current,
        userAction,
      });
      setFlowState(FLOW.SENT);
      setMessages((current) =>
        current.map((item) =>
          item.messageId === payload.message_id ? { ...item, sendStatus: "sent" } : item
        )
      );
      setText("");
      analyzedTextRef.current = "";
      setAnalysis(payload);
      setTimeout(() => setFlowState(FLOW.IDLE), 400);
    } catch (err) {
      const message = err.response?.data?.message || "Could not send message.";
      if (err.response?.status === 403 && /wait|lock|delay/i.test(message)) {
        setFlowState(FLOW.DELAYED_SEND);
      } else {
        setFlowState(FLOW.ERROR);
      }
      setError(message);
    } finally {
      sendingRef.current = false;
    }
  }

  async function handleSend() {
    if (flowState === FLOW.DELAYED_SEND || flowState === FLOW.REWRITE_REQUIRED) {
      return;
    }
    if (flowState === FLOW.WARNING) {
      await runSend("send_anyway");
      return;
    }
    if (flowState === FLOW.ALLOW || flowState === FLOW.READY_TO_SEND) {
      await runSend("send");
      return;
    }
    const result = await runAnalyze();
    if (result?.decision === "ALLOW") {
      await runSend("send", result);
    }
  }

  function handleVoiceAnalysis(result, sourceText) {
    const spoken = typeof sourceText === "string" ? sourceText : result.original_text;
    const score = typeof result.risk_score === "number" ? result.risk_score : result.final_risk;
    setText(spoken);
    pushMessage("user", spoken, {
      source: "voice",
      messageId: result.message_id,
      decision: result.decision,
      riskScore: typeof score === "number" ? score : null,
      scoreFailed: typeof score !== "number",
    });
    applyAnalysis(result, spoken);
  }

  const sendDisabled =
    flowState === FLOW.ANALYZING ||
    flowState === FLOW.SENDING ||
    flowState === FLOW.DELAYED_SEND ||
    flowState === FLOW.REWRITE_REQUIRED ||
    !text.trim();

  const suggested =
    analysis?.suggested_text || analysis?.rewrite?.suggested_text || analysis?.action?.suggested_text;
  const exercises = analysis?.delayed_send?.exercises || FALLBACK_EXERCISES;
  const initial = (user?.username || user?.email || "Y").trim().charAt(0).toUpperCase();
  const clock = formatRemaining(remainingSeconds);

  async function copySuggestion() {
    if (!suggested) {
      return;
    }
    try {
      await navigator.clipboard.writeText(suggested);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  function beginEdit() {
    analyzedTextRef.current = "";
    setFlowState(FLOW.IDLE);
    setWarnHidden(true);
  }

  return (
    <section className="flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex shrink-0 items-center justify-between px-5 py-4">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-[#3d3a8c]">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M7 17.5 4 20V6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v8A2.5 2.5 0 0 1 17.5 17H7Z" />
          </svg>
          Conversation
        </h2>
        <div className="flex items-center gap-2">
          {conversations.length > 1 && (
            <select
              value={conversationId || ""}
              onChange={handleSelectConversation}
              aria-label="Choose conversation"
              className="rounded-xl bg-[#f6f7fb] px-2 py-1 text-xs text-slate-500"
            >
              {conversations.map((row) => (
                <option key={row.conversation_id} value={row.conversation_id}>
                  Chat {row.conversation_id}
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            onClick={handleNewConversation}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-3 py-1.5 text-sm text-slate-500"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M6 7h12M9 7V5h6v2M8 7l1 12h6l1-12" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Clear Chat
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 pb-4">
        {messages.length === 0 && (
          <p className="pt-8 text-center text-sm text-slate-400">Type a message to analyze it before it is sent.</p>
        )}
        {messages
          .filter((message) => message.role === "user")
          .map((message) => (
            <div key={message.id} className="flex items-start gap-3">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-slate-200 text-sm font-semibold text-slate-600">
                {initial}
              </span>
              <div>
                <p className="text-sm font-semibold text-slate-700">You</p>
                <div className="mt-1 max-w-md rounded-2xl rounded-tl-md bg-[#f3f4f8] px-4 py-3 text-sm text-slate-700">
                  {message.text}
                  {message.loadingScore && <p className="mt-1 text-xs text-slate-400">Analyzing…</p>}
                  {message.scoreFailed && <p className="mt-1 text-xs text-rose-500">Risk score unavailable</p>}
                </div>
                <p className="mt-1 text-xs text-slate-400">{message.at}</p>
              </div>
            </div>
          ))}

        {flowState === FLOW.WARNING && !warnHidden && (
          <div className="flex items-center justify-between rounded-2xl bg-[#fff6dd] px-4 py-3 text-sm text-[#c98409]">
            <p className="flex items-center gap-2">
              <span className="grid h-5 w-5 place-items-center rounded-full bg-[#f5b942] text-xs font-bold text-white">!</span>
              This message might be interpreted as negative.
            </p>
            <button type="button" onClick={() => setWarnHidden(true)} aria-label="Dismiss warning" className="text-lg leading-none">
              ×
            </button>
          </div>
        )}

        {flowState === FLOW.REWRITE_REQUIRED && (
          <div className="rounded-3xl border border-[#ffd0d4] bg-[#fff5f6] p-4">
            <p className="flex items-center gap-2 font-semibold text-[#ef4d5c]">
              <span aria-hidden="true">✨</span> Suggested Safer Message
            </p>
            <div className="mt-3 flex items-start justify-between gap-3 rounded-2xl bg-white px-4 py-3 text-sm text-slate-700 shadow-sm">
              <p className="whitespace-pre-wrap">{suggested || "Edit the message into a more respectful version."}</p>
              {suggested && (
                <button type="button" onClick={copySuggestion} aria-label="Copy suggestion" className="text-slate-400">
                  {copied ? "Copied" : (
                    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
                      <rect x="8" y="8" width="11" height="11" rx="2" />
                      <path d="M5 15V5h10" />
                    </svg>
                  )}
                </button>
              )}
            </div>
            <p className="mt-3 text-sm text-[#e07a84]">
              This message might be interpreted as insulting. Consider using a more respectful and positive tone.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {suggested && (
                <button
                  type="button"
                  onClick={async () => {
                    setText(suggested);
                    const result = await runAnalyze(suggested);
                    if (result?.decision === "ALLOW") {
                      setFlowState(FLOW.READY_TO_SEND);
                    }
                  }}
                  className="rounded-xl bg-[#ff4d5e] px-4 py-2 text-sm font-semibold text-white"
                >
                  Use Suggested
                </button>
              )}
              <button type="button" onClick={beginEdit} className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-600">
                Edit
              </button>
              <button
                type="button"
                onClick={() => setError("Sending the original message is not permitted. Use the suggestion or edit it.")}
                className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-600"
              >
                Keep Original
              </button>
            </div>
          </div>
        )}

        {flowState === FLOW.DELAYED_SEND && (
          <div className="rounded-3xl border border-[#e4defe] bg-[#f7f5ff] p-5">
            <p className="flex items-center gap-2 font-semibold text-[#5b4dff]">
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
                <circle cx="12" cy="13" r="7" />
                <path d="M12 10v3.5l2 1.5M9 4h6" strokeLinecap="round" />
              </svg>
              Take a moment before sending
            </p>
            <p className="mt-2 text-sm text-slate-500">
              This message has been placed on a short pause because its current risk is high.
            </p>
            <p className="mx-auto mt-4 w-fit rounded-full bg-[#ece7ff] px-8 py-3 font-mono text-3xl font-semibold tracking-wider text-[#5b4dff]">
              {clock}
            </p>
            <p className="mt-4 text-center text-sm text-slate-500">Try a short calming exercise below.</p>
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {exercises.slice(0, 4).map((item, index) => {
                const preset = EXERCISE_PRESETS[index] || EXERCISE_PRESETS[3];
                const selected = activeExercise === item.exercise;
                return (
                  <button
                    key={item.exercise}
                    type="button"
                    onClick={() => setActiveExercise(selected ? null : item.exercise)}
                    className={`rounded-2xl border px-2 py-3 text-center text-xs font-medium ${
                      selected ? "border-[#5b4dff] bg-white text-[#5b4dff]" : "border-transparent bg-white text-slate-600"
                    }`}
                  >
                    <span className="mb-1 block text-lg" aria-hidden="true">{preset.icon}</span>
                    {preset.label}
                  </button>
                );
              })}
            </div>
            {activeExercise && (
              <p className="mt-3 text-sm text-slate-500">
                {exercises.find((item) => item.exercise === activeExercise)?.instructions}
              </p>
            )}
            {remainingSeconds === 0 && (
              <p className="mt-3 text-center text-xs text-slate-500">
                The pause is over. Edit the message and analyze it again before sending.
              </p>
            )}
          </div>
        )}

        {analysis && (
          <details className="rounded-2xl bg-[#f6f7fb] p-3 text-sm">
            <summary className="cursor-pointer font-medium text-slate-500">Analysis details</summary>
            <div className="mt-3">
              <AnalysisResults analysis={analysis} analyzing={flowState === FLOW.ANALYZING} />
            </div>
          </details>
        )}
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (!sendDisabled) {
            handleSend();
          }
        }}
        className="shrink-0 border-t border-slate-100 px-4 py-4"
      >
        <div className="flex items-center gap-2">
          <span className="grid h-10 w-10 place-items-center text-slate-300" title="Attachments are not available" aria-hidden="true">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="m8 12 4.5-4.5a3 3 0 1 1 4.2 4.2L10 18.4a4.2 4.2 0 0 1-6-6L11.5 5" strokeLinecap="round" />
            </svg>
          </span>
          <input
            value={text}
            onChange={(event) => handleTextChange(event.target.value)}
            placeholder="Type your message..."
            className="h-12 flex-1 rounded-full bg-[#f4f5f9] px-5 text-sm outline-none"
          />
          <VoiceInput compact onAnalyzed={handleVoiceAnalysis} conversationId={conversationId} />
          {flowState === FLOW.WARNING && (
            <span className="hidden items-center gap-1 rounded-full bg-[#fff3d6] px-3 py-2 text-xs font-bold text-[#e39a12] sm:inline-flex">
              <span aria-hidden="true">⚠</span> WARN
            </span>
          )}
          <button
            type="submit"
            disabled={sendDisabled}
            className="inline-flex h-12 items-center gap-2 rounded-full bg-[#5b4dff] px-5 text-sm font-semibold text-white disabled:bg-slate-300"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
              <path d="M3 11.5 21 3l-7.2 18-2.3-7.2L3 11.5Z" />
            </svg>
            {flowState === FLOW.ANALYZING ? "Analyzing…" : flowState === FLOW.SENDING ? "Sending…" : "Send"}
          </button>
        </div>
        {error && <p className="mt-2 px-2 text-sm text-rose-500">{error}</p>}
      </form>
    </section>
  );
}

const EXERCISE_PRESETS = [
  { label: "Breathing", icon: "🫁" },
  { label: "Grounding", icon: "🍃" },
  { label: "Relax shoulders", icon: "🧘" },
  { label: "Pause & reread", icon: "📖" },
];

const FALLBACK_EXERCISES = [
  { exercise: "Breathing", instructions: "Inhale through your nose for 4 seconds, hold for 2, exhale through your mouth for 6." },
  { exercise: "Grounding", instructions: "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you remember." },
  { exercise: "Relax shoulders", instructions: "Shrug your shoulders up, hold, then drop them and unclench your jaw and hands." },
  { exercise: "Pause & reread", instructions: "Read the message once as if you were the recipient." },
];

function formatRemaining(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const rest = safe % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

export default Chat;
