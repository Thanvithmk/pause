import { useEffect, useState } from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getDashboard } from "../services/api";
const TIMELINE_FILTERS = ["ALL", "ALLOW", "WARN", "REWRITE", "DELAYED-SEND"];

function parsedDate(value) {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatWhen(value) {
  const date = parsedDate(value);
  if (!date) {
    return value ? String(value) : "";
  }
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDate(value) {
  const date = parsedDate(value);
  if (!date) {
    return value ? String(value) : "";
  }
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function formatTime(value) {
  const date = parsedDate(value);
  if (!date) {
    return "";
  }
  return date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

const EMPTY_DASHBOARD = {
  toxicity_trends: [],
  message_outcomes: [],
  escalation_history: [],
  insights: ["Not enough data in this window yet."],
};

function BehaviorDashboard() {
  const [days, setDays] = useState(30);
  const [timelineFilter, setTimelineFilter] = useState("ALL");
  const [data, setData] = useState(EMPTY_DASHBOARD);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const payload = await getDashboard(days);
        if (!cancelled) {
          setData({ ...EMPTY_DASHBOARD, ...payload });
        }
      } catch {
        if (!cancelled) {
          setError("Dashboard data is unavailable.");
          setData(EMPTY_DASHBOARD);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [days]);

  const timeline = (data.recent_activity || []).filter(
    (item) => timelineFilter === "ALL" || item.decision === timelineFilter,
  );

  return (
    <div className="space-y-6 p-5 md:p-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Behavior dashboard</h2>
          <p className="text-sm text-slate-400">Trends, outcomes, and interventions</p>
        </div>
        <label className="text-sm text-slate-400">
          Window
          <select
            value={days}
            onChange={(event) => setDays(Number(event.target.value))}
            className="ml-2 rounded-lg bg-[#f6f7fb] px-2 py-1"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
        </label>
      </div>
      {loading && <p className="text-slate-500">Loading dashboard…</p>}
      {error && <p className="text-delay text-sm">{error}</p>}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Analyzed", data.messages_analyzed ?? 0],
          ["Sent", data.messages_sent ?? 0],
          ["Average risk", Number(data.average_risk || 0).toFixed(3)],
          ["Unknown slang", data.slang_statistics?.unknown_terms_logged ?? 0],
        ].map(([label, value]) => (
          <section key={label} className="rounded-2xl bg-[#f6f7fb] p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
            <p className="mt-1 text-2xl font-semibold">{value}</p>
          </section>
        ))}
      </div>
      {data.intervention_effectiveness?.counts && (
        <section className="rounded-2xl bg-[#f6f7fb] p-4">
          <h3 className="text-sm font-medium">Intervention effectiveness</h3>
          <p className="mt-1 text-xs text-slate-500">
            Counted from recorded responses. These scores do not change the risk formula.
          </p>
          <dl className="mt-3 grid gap-2 sm:grid-cols-3 text-sm">
            {[
              ["WARN", data.intervention_effectiveness.scores?.WARN_eff, data.intervention_effectiveness.counts.warn_success, data.intervention_effectiveness.counts.warn_fail],
              ["REWRITE", data.intervention_effectiveness.scores?.REWRITE_eff, data.intervention_effectiveness.counts.rewrite_success, data.intervention_effectiveness.counts.rewrite_fail],
              ["DELAYED-SEND", data.intervention_effectiveness.scores?.DELAYED_SEND_eff, data.intervention_effectiveness.counts.delayed_success, data.intervention_effectiveness.counts.delayed_fail],
            ].map(([name, score, success, fail]) => (
              <div key={name}>
                <dt className="text-slate-500">{name}</dt>
                <dd className="font-mono">
                  {Number(score || 0).toFixed(2)} · {Number(success || 0)} accepted · {Number(fail || 0)} other
                </dd>
              </div>
            ))}
          </dl>
        </section>
      )}
      <div className="grid gap-3 sm:grid-cols-4">
        {(data.message_outcomes || []).map((item) => (
          <section key={item.name} className="rounded-2xl bg-[#f6f7fb] p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500">{item.name}</p>
            <p className="mt-1 text-xl font-semibold">{item.value}</p>
          </section>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-2xl bg-[#f6f7fb] p-4">
          <h3 className="mb-3 text-sm font-medium">Toxicity trends</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.toxicity_trends}>
                <XAxis dataKey="date" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" domain={[0, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="toxic" stroke="#22c55e" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-2xl bg-[#f6f7fb] p-4">
          <h3 className="mb-3 text-sm font-medium">Escalation history</h3>
          <ul className="max-h-56 space-y-2 overflow-y-auto text-sm">
            {(data.escalation_history || []).length === 0 && (
              <li className="text-slate-500">No escalations in this window.</li>
            )}
            {(data.escalation_history || []).map((item, index) => (
              <li key={`${item.timestamp}-${index}`} className="rounded-lg bg-white px-3 py-2">
                <p className="text-slate-700">{formatDate(item.timestamp)}</p>
                <p className="text-slate-500">{formatTime(item.timestamp)}</p>
                <p className="mt-1 text-slate-800">
                  <span className="font-medium">Trigger topic: </span>
                  {item.trigger_topic || "Unavailable"}
                </p>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="rounded-2xl bg-[#f6f7fb] p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-sm font-medium">Timeline</h3>
          <div className="flex flex-wrap gap-2" role="group" aria-label="Filter timeline">
            {TIMELINE_FILTERS.map((filter) => (
              <button
                key={filter}
                type="button"
                onClick={() => setTimelineFilter(filter)}
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  timelineFilter === filter ? "bg-[#5b4dff] text-white" : "bg-white text-slate-500"
                }`}
                aria-pressed={timelineFilter === filter}
              >
                {filter === "ALL" ? "All" : filter}
              </button>
            ))}
          </div>
        </div>
        <ul className="mt-3 max-h-72 space-y-2 overflow-y-auto text-sm">
          {timeline.length === 0 && (
            <li className="text-slate-500">No messages in this window for that filter.</li>
          )}
          {timeline.map((item, index) => (
            <li key={`${item.timestamp}-${index}`} className="rounded-lg bg-white px-3 py-2">
              <div className="flex items-center justify-between gap-3">
                <span className="font-semibold text-slate-700">{item.decision || "UNKNOWN"}</span>
                <span className="text-xs text-slate-400">{formatWhen(item.timestamp)}</span>
              </div>
              <p className="mt-1 text-slate-600">{item.text}</p>
              <p className="mt-1 text-xs text-slate-400">
                Risk {Number(item.final_risk || 0).toFixed(4)}
                {item.send_status === "sent" ? " · Sent" : " · Not sent"}
              </p>
            </li>
          ))}
        </ul>
      </section>

      <section className="rounded-2xl bg-[#f6f7fb] p-4">
        <h3 className="mb-2 text-sm font-medium">Insights</h3>
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
          {(data.insights || []).map((insight) => (
            <li key={insight}>{insight}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

export default BehaviorDashboard;
