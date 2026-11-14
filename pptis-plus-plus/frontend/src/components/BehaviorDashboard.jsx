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

function interventionRows(effectiveness) {
  const counts = effectiveness?.counts || {};
  const scores = effectiveness?.scores || {};
  return [
    ["WARN", scores.WARN_eff, counts.warn_success, counts.warn_fail],
    ["REWRITE", scores.REWRITE_eff, counts.rewrite_success, counts.rewrite_fail],
    ["DELAYED-SEND", scores.DELAYED_SEND_eff, counts.delayed_success, counts.delayed_fail],
  ].map(([name, score, heeded, sent]) => ({
    name,
    score: Number(score || 0),
    heeded: Number(heeded || 0),
    sent: Number(sent || 0),
  }));
}

const TOPIC_STYLES = {
  Community: "border-warn bg-warn/15 text-warn",
  Profanity: "border-warn bg-warn/15 text-warn",
  Insult: "border-rewrite bg-rewrite/15 text-rewrite",
  Harassment: "border-rewrite bg-rewrite/15 text-rewrite",
  Harm: "border-delay bg-delay/15 text-delay",
  "Self-harm": "border-delay bg-delay/15 text-delay",
};

function followedIntervention(rows) {
  const ranked = rows
    .filter((row) => row.heeded + row.sent > 0)
    .sort((left, right) => right.score - left.score || right.heeded - left.heeded);
  return ranked[0] && ranked[0].score > 0 ? ranked[0] : null;
}

const EMPTY_DASHBOARD = {
  toxicity_trends: [],
  message_outcomes: [],
  trigger_categories: [],
  insights: ["Not enough data in this window yet."],
};

function BehaviorDashboard() {
  const [days, setDays] = useState(30);
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

  const triggers = data.trigger_categories || [];
  const interventions = interventionRows(data.intervention_effectiveness);
  const followed = followedIntervention(interventions);
  const hasInterventionResponses = interventions.some((row) => row.heeded + row.sent > 0);

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
          <h3 className="text-sm font-medium">How this user responds</h3>
          <p className="mt-1 text-sm text-slate-500">
            {followed
              ? `${followed.name} is the intervention this user follows most often.`
              : "No intervention has been followed yet."}
            {" "}
            A higher share means they edited or cancelled instead of sending the original. These scores do not change the risk formula.
          </p>
          {hasInterventionResponses ? (
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              {interventions.map((row) => {
                const total = row.heeded + row.sent;
                const percent = Math.round(row.score * 100);
                const isFollowed = followed?.name === row.name;
                return (
                  <article key={row.name} className="rounded-2xl bg-white p-4">
                    <div className="flex items-start justify-between gap-3">
                      <h4 className="text-sm font-semibold text-slate-700">{row.name}</h4>
                      {isFollowed && (
                        <span className="rounded-full bg-[#efeefe] px-2 py-1 text-xs font-semibold text-[#5b4dff]">
                          Followed most
                        </span>
                      )}
                    </div>
                    <p className="mt-3 text-3xl font-semibold text-slate-900">{percent}%</p>
                    <p className="text-xs text-slate-500">followed the intervention</p>
                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#f4f5f9]" aria-hidden="true">
                      <div className="h-full rounded-full bg-[#5b4dff]" style={{ width: `${percent}%` }} />
                    </div>
                    <p className="mt-3 text-sm text-slate-600">
                      {total === 0
                        ? "No responses yet"
                        : `${row.heeded} edited or cancelled · ${row.sent} sent the original`}
                    </p>
                  </article>
                );
              })}
            </div>
          ) : (
            <p className="mt-3 text-sm text-slate-500">No intervention responses yet.</p>
          )}
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

      <section className="rounded-2xl bg-[#f6f7fb] p-4">
        <h3 className="text-sm font-medium">Trigger categories</h3>
        <p className="mt-1 text-sm text-slate-500">
          Topics of messages that were warned, rewritten, or delayed in this window.
        </p>
        {triggers.length === 0 ? (
          <p className="mt-3 text-sm text-slate-500">No topics in this window.</p>
        ) : (
          <div className="mt-3 flex flex-wrap gap-2">
            {triggers.map((item) => (
              <article
                key={item.category}
                className={`rounded-2xl border px-3 py-2 text-sm font-semibold ${TOPIC_STYLES[item.category] || "border-slate-200 bg-white text-slate-600"}`}
              >
                {item.category}
                {item.count ? ` · ${item.count}` : ""}
              </article>
            ))}
          </div>
        )}
      </section>

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
