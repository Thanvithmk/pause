import { useEffect, useRef, useState } from "react";
import TextInput from "./TextInput";
import VoiceInput from "./VoiceInput";
import AnalysisResults from "./AnalysisResults";
import InterventionModal from "./InterventionModal";
import RiskScore from "./RiskScore";
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
  const [modal, setModal] = useState(null);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  const sendingRef = useRef(false);
  const analyzedTextRef = useRef("");

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
        const latest = rows[0];
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
    setModal(null);
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
      setModal(null);
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
    if (decision === "WARN") {
      setModal("WARN");
    } else if (decision === "REWRITE") {
      setModal("REWRITE");
    } else if (decision === "DELAYED-SEND") {
      setModal("DELAYED-SEND");
      setRemainingSeconds(result?.delayed_send?.countdown_seconds || 60);
    } else {
      setModal(null);
    }
    return result;
  }

  function handleTextChange(value) {
    setText(value);
    if (analyzedTextRef.current && value.trim() !== analyzedTextRef.current) {
      analyzedTextRef.current = "";
      setAnalysis(null);
      setFlowState(FLOW.IDLE);
      setModal(null);
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
      setModal(null);
      pushMessage("system", `Sent (${sent.decision})`, { inheritScore: false });
      setText("");
      analyzedTextRef.current = "";
      setAnalysis(payload);
      setTimeout(() => setFlowState(FLOW.IDLE), 400);
    } catch (err) {
      const message = err.response?.data?.message || "Could not send message.";
      if (err.response?.status === 403 && /wait|lock|delay/i.test(message)) {
        setFlowState(FLOW.DELAYED_SEND);
        setModal("DELAYED-SEND");
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
      setModal(flowState === FLOW.DELAYED_SEND ? "DELAYED-SEND" : "REWRITE");
      return;
    }
    if (flowState === FLOW.WARNING) {
      setModal("WARN");
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

  const sendLabel =
    flowState === FLOW.DELAYED_SEND
      ? remainingSeconds > 0
        ? `Locked ${remainingSeconds}s`
        : "Locked — edit to continue"
      : flowState === FLOW.REWRITE_REQUIRED
        ? "Rewrite required"
        : "Send";

  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <section className="rounded-2xl border border-slate-800 bg-panel p-4 flex flex-col min-h-[70vh]">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold">Conversation</h2>
            <p className="text-xs text-slate-400">
              Signed in as {user?.username || user?.email || "you"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={conversationId || ""}
              onChange={handleSelectConversation}
              className="rounded-lg bg-ink px-2 py-1 text-xs ring-1 ring-slate-700"
            >
              {conversations.length === 0 && <option value="">No conversations</option>}
              {conversations.map((row) => (
                <option key={row.conversation_id} value={row.conversation_id}>
                  Chat {row.conversation_id}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={handleNewConversation}
              className="rounded-lg bg-slate-800 px-2 py-1 text-xs text-slate-200"
            >
              New chat
            </button>
          </div>
        </div>
        <div className="flex-1 space-y-3 overflow-y-auto pr-1">
          {messages.length === 0 && (
            <p className="text-slate-500 text-sm">No messages yet. Type or record to analyze.</p>
          )}
          {messages.map((message) => (
            <div
              key={message.id}
              className={`max-w-[85%] rounded-xl px-3 py-2 text-sm ${
                message.role === "user"
                  ? "ml-auto bg-emerald-500/15 text-emerald-100"
                  : "bg-slate-800 text-slate-200"
              }`}
            >
              <p>{message.text}</p>
              {message.role === "user" && message.decision && (
                <p className="mt-1 text-xs font-semibold">
                  Decision: {message.decision}
                  {message.sendStatus ? ` · ${message.sendStatus}` : ""}
                </p>
              )}
              {message.role === "user" && (
                <RiskScore
                  score={message.riskScore}
                  loading={message.loadingScore}
                  failed={message.scoreFailed}
                />
              )}
              <p className="mt-1 text-[10px] uppercase tracking-wide text-slate-500">
                {message.role} · {message.at}
              </p>
            </div>
          ))}
        </div>
        <div className="mt-4 space-y-3 border-t border-slate-800 pt-4">
          <TextInput
            text={text}
            onTextChange={handleTextChange}
            onSend={handleSend}
            onAnalyze={() => runAnalyze()}
            flowState={flowState}
            sendDisabled={sendDisabled}
            sendLabel={sendLabel}
            analyzeDisabled={flowState === FLOW.DELAYED_SEND && remainingSeconds > 0}
            error={error}
          />
          <VoiceInput onAnalyzed={handleVoiceAnalysis} conversationId={conversationId} />
        </div>
      </section>
      <AnalysisResults analysis={analysis} analyzing={flowState === FLOW.ANALYZING} />
      <InterventionModal
        kind={modal}
        analysis={analysis}
        remainingSeconds={remainingSeconds}
        onSendAnyway={() => runSend("send_anyway")}
        onCancel={() => setModal(null)}
        onEdit={() => {
          setModal(null);
          analyzedTextRef.current = "";
          setFlowState(FLOW.IDLE);
        }}
        onUseRewrite={async (suggested) => {
          setText(suggested);
          setModal(null);
          const result = await runAnalyze(suggested);
          if (result?.decision === "ALLOW") {
            setFlowState(FLOW.READY_TO_SEND);
          }
        }}
      />
    </div>
  );
}

export default Chat;
