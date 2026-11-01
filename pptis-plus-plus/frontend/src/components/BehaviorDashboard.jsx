import { useEffect, useState } from "react";
import {
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getDashboard } from "../services/api";

const PIE_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#64748b"];

const EMPTY_DASHBOARD = {
  toxicity_trends: [],
  message_outcomes: [],
  escalation_history: [],
  decision_breakdown: [],
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

  return (
    <div className="space-y-6">
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
            className="ml-2 rounded-lg bg-panel px-2 py-1 ring-1 ring-slate-700"
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
          <section key={label} className="rounded-2xl border border-slate-800 bg-panel p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
            <p className="mt-1 text-2xl font-semibold">{value}</p>
          </section>
        ))}
      </div>
      <div className="grid gap-3 sm:grid-cols-4">
        {(data.message_outcomes || []).map((item) => (
          <section key={item.name} className="rounded-2xl border border-slate-800 bg-panel p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500">{item.name}</p>
            <p className="mt-1 text-xl font-semibold">{item.value}</p>
          </section>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-2xl border border-slate-800 bg-panel p-4">
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

        <section className="rounded-2xl border border-slate-800 bg-panel p-4">
          <h3 className="mb-3 text-sm font-medium">Message outcomes</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data.message_outcomes}
                  dataKey="value"
                  nameKey="name"
                  outerRadius={80}
                >
                  {(data.message_outcomes || []).map((entry, index) => (
                    <Cell key={entry.name} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-panel p-4">
          <h3 className="mb-3 text-sm font-medium">Decision breakdown</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data.decision_breakdown}
                  dataKey="value"
                  nameKey="name"
                  outerRadius={80}
                >
                  {(data.decision_breakdown || []).map((entry, index) => (
                    <Cell key={entry.name} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-panel p-4">
          <h3 className="mb-3 text-sm font-medium">Escalation history</h3>
          <ul className="max-h-56 space-y-2 overflow-y-auto text-sm">
            {(data.escalation_history || []).length === 0 && (
              <li className="text-slate-500">No escalations in this window.</li>
            )}
            {(data.escalation_history || []).map((item, index) => (
              <li key={index} className="rounded-lg bg-ink px-3 py-2">
                <span className="text-rewrite">{item.decision || "ESCALATION"}</span>
                <span className="ml-2 text-slate-400">{item.timestamp || item.date}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="rounded-2xl border border-slate-800 bg-panel p-4">
        <h3 className="mb-2 text-sm font-medium">Insights</h3>
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-300">
          {(data.insights || []).map((insight) => (
            <li key={insight}>{insight}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

export default BehaviorDashboard;
