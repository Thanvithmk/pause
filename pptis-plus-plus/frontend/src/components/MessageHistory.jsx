import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listConversations } from "../services/api";
import { formatClock } from "../utils/time";

export const OPEN_CONVERSATION_KEY = "pptis_open_conversation";

function MessageHistory() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    listConversations()
      .then((payload) => {
        if (!cancelled) {
          setRows(payload.conversations || []);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError("Could not load message history.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function openRow(id) {
    sessionStorage.setItem(OPEN_CONVERSATION_KEY, String(id));
    navigate("/chat");
  }

  return (
    <div className="p-5 md:p-8">
      <h2 className="text-xl font-semibold text-slate-900">Message History</h2>
      <p className="mt-1 text-sm text-slate-400">Open a conversation to continue it.</p>
      {error && <p className="mt-4 text-sm text-rose-500">{error}</p>}
      <ul className="mt-6 space-y-3">
        {rows.length === 0 && !error && (
          <li className="rounded-2xl bg-[#f6f7fb] px-4 py-6 text-sm text-slate-400">No conversations yet.</li>
        )}
        {rows.map((row) => (
          <li key={row.conversation_id}>
            <button
              type="button"
              onClick={() => openRow(row.conversation_id)}
              className="flex w-full items-center justify-between rounded-2xl bg-[#f6f7fb] px-4 py-3 text-left hover:bg-[#efeaff]"
            >
              <span className="font-medium text-slate-800">Conversation {row.conversation_id}</span>
              <span className="text-xs text-slate-400">{formatClock(row.updated_at || row.created_at)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default MessageHistory;
