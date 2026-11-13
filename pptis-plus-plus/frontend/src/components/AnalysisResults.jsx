import RiskScore from "./RiskScore";

const TOXICITY_LABELS = [
  ["toxic", "Toxic"],
  ["threat", "Threat"],
  ["insult", "Insult"],
  ["severe_toxic", "Severe"],
  ["obscene", "Obscene"],
  ["identity_hate", "Identity hate"],
];

const DECISION_STYLES = {
  ALLOW: "bg-allow/20 text-allow border-allow",
  WARN: "bg-warn/20 text-warn border-warn",
  REWRITE: "bg-rewrite/20 text-rewrite border-rewrite",
  "DELAYED-SEND": "bg-delay/20 text-delay border-delay",
};

function scoreOf(analysis, keys) {
  for (const key of keys) {
    const value = key.split(".").reduce((object, part) => object?.[part], analysis);
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function AnalysisResults({ analysis, analyzing = false }) {
  if (analyzing && !analysis) {
    return (
      <aside className="rounded-2xl border border-dashed border-slate-200 p-6 text-slate-400">
        Analyzing risk…
      </aside>
    );
  }
  if (!analysis) {
    return (
      <aside className="rounded-2xl border border-dashed border-slate-200 p-6 text-slate-400">
        Analysis will appear here after you send or record a message.
      </aside>
    );
  }

  const toxicity = analysis.toxicity || analysis.toxicity_scores || analysis;
  const decision = analysis.decision || analysis.risk?.decision || "ALLOW";
  const slang = Array.isArray(analysis.slang) ? analysis.slang : [];
  const recognized = analysis.linguistic_features?.slang?.terms || [];
  const emojis = analysis.linguistic_features?.emoji?.emojis || [];
  const combinations = analysis.linguistic_features?.emoji?.combinations || [];
  const slangItems = recognized.length > 0 ? recognized : slang;
  const baseline = analysis.behavioral_baseline;
  const delta = analysis.delta_vector || analysis.delta_vector_6d;
  const escalationFeature = scoreOf(analysis, ["escalation_feature"]);
  const pacl = scoreOf(analysis, ["pacl", "risk.pacl", "personalized_risk"]);
  const pEsc = scoreOf(analysis, ["escalation_probability", "p_esc", "risk.escalation_probability"]);
  const finalRisk = scoreOf(analysis, ["risk_score", "final_risk", "risk.final_risk"]);
  const style = DECISION_STYLES[decision] || DECISION_STYLES.ALLOW;

  return (
    <aside className="space-y-5 rounded-2xl bg-white p-2">
      <div>
        <p className="text-xs uppercase tracking-widest text-slate-500">Decision</p>
        <div className={`mt-2 inline-block rounded-full border px-3 py-1 text-sm font-semibold ${style}`}>
          {decision}
        </div>
        {analysis.mathematical_decision && analysis.mathematical_decision !== decision && (
          <p className="mt-2 text-xs text-slate-500">
            Mathematical risk is {analysis.mathematical_decision}.{" "}
            {String(analysis.guardrail || "").includes("contextual_threat")
              ? "The contextual threat assessment lowered the intervention. Toxic-BERT scores and final risk were not changed."
              : String(analysis.guardrail || "").includes("playful_emoji")
              ? "Laughter emoji lowered the intervention. Final risk was not changed."
              : `A separate safety guardrail raised the intervention${
                  analysis.guardrail ? ` (${analysis.guardrail})` : ""
                }. Final risk was not changed.`}
          </p>
        )}
        {analysis.guardrail && analysis.mathematical_decision === decision && (
          <p className="mt-2 text-xs text-slate-500">Safety guardrail: {analysis.guardrail}</p>
        )}
        {analysis.contextual_threat &&
          analysis.contextual_threat.category &&
          analysis.contextual_threat.category !== "none" && (
          <p className="mt-2 text-xs text-slate-500">
            Contextual assessment: {String(analysis.contextual_threat.category).replaceAll("_", " ")}
            {analysis.contextual_threat.object ? ` · ${analysis.contextual_threat.object}` : ""}.
            This is separate from the Toxic-BERT scores.
          </p>
        )}
        {analysis.rewrite?.status === "unavailable" && (
          <p className="mt-2 text-xs text-slate-500">
            {analysis.rewrite.error || "Rewrite generation is temporarily unavailable."}
          </p>
        )}
        <div className="mt-3">
          <RiskScore score={finalRisk} loading={analyzing} failed={finalRisk === null} />
        </div>
      </div>

      <div>
        <p className="mb-2 text-sm font-medium">Toxicity dimensions</p>
        <div className="space-y-2">
          {TOXICITY_LABELS.map(([key, label]) => {
            const value = Number(toxicity[key] ?? 0);
            return (
              <div key={key}>
                <div className="mb-1 flex justify-between text-xs text-slate-400">
                  <span>{label}</span>
                  <span>{value.toFixed(4)}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full bg-emerald-400"
                    style={{ width: `${Math.min(100, value * 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {slangItems.length > 0 && (
        <div>
          <p className="mb-2 text-sm font-medium">Slang</p>
          <ul className="space-y-2 text-sm">
            {slangItems.map((item) => (
              <li key={`${item.term}-${item.start ?? item.usage}`} className="rounded-lg bg-[#f6f7fb] p-3">
                <p className="font-medium">
                  {item.term}
                  {item.ambiguous ? " · ambiguous" : ""}
                </p>
                <p className="text-slate-400">
                  {item.usage || "recognized"}
                  {typeof item.confidence === "number" ? ` · ${item.confidence.toFixed(4)}` : ""}
                </p>
                {item.meaning && <p className="mt-1 text-slate-600">{item.meaning}</p>}
                {Array.isArray(item.senses) && item.senses.map((sense) => (
                  <p key={`${sense.definition}-${sense.category}`} className="mt-1 text-slate-600">
                    {sense.definition}
                    {sense.category ? ` · ${sense.category}` : ""}
                  </p>
                ))}
              </li>
            ))}
          </ul>
        </div>
      )}

      {emojis.length > 0 && (
        <div>
          <p className="mb-2 text-sm font-medium">Emoji</p>
          <ul className="space-y-2 text-sm">
            {emojis.map((item) => (
              <li key={`${item.emoji}-${item.start}`} className="rounded-lg bg-[#f6f7fb] p-3">
                <p className="font-medium">{item.emoji}</p>
                <p className="text-slate-600">{item.description || item.meaning || "No saved description"}</p>
                {item.meaning && item.description && (
                  <p className="text-slate-400">{item.meaning}</p>
                )}
              </li>
            ))}
          </ul>
          {combinations.length > 0 && (
            <p className="mt-2 text-xs text-slate-400">
              Combinations: {combinations.map((item) => item.emoji).join("  ")}
            </p>
          )}
        </div>
      )}

      <dl className="grid grid-cols-1 gap-3 text-sm">
        <div className="rounded-lg bg-[#f6f7fb] p-3">
          <dt className="text-slate-400">Personalized risk (PACL)</dt>
          <dd className="font-mono text-lg">{pacl === null ? "—" : pacl.toFixed(4)}</dd>
        </div>
        <div className="rounded-lg bg-[#f6f7fb] p-3">
          <dt className="text-slate-400">Escalation feature</dt>
          <dd className="font-mono text-lg">{escalationFeature === null ? "—" : escalationFeature.toFixed(4)}</dd>
        </div>
        <div className="rounded-lg bg-[#f6f7fb] p-3">
          <dt className="text-slate-400">Escalation probability</dt>
          <dd className="font-mono text-lg">{pEsc === null ? "—" : pEsc.toFixed(4)}</dd>
        </div>
        {baseline && (
          <div className="rounded-lg bg-[#f6f7fb] p-3">
            <dt className="text-slate-400">Behavioral baseline (toxic)</dt>
            <dd className="font-mono text-lg">{Number(baseline.toxic || 0).toFixed(4)}</dd>
          </div>
        )}
        {Array.isArray(delta) && (
          <div className="rounded-lg bg-[#f6f7fb] p-3">
            <dt className="text-slate-400">Delta vector</dt>
            <dd className="font-mono text-xs">{delta.map((value) => Number(value).toFixed(4)).join(", ")}</dd>
          </div>
        )}
        <div className="rounded-lg bg-[#f6f7fb] p-3">
          <dt className="text-slate-400">Final risk</dt>
          <dd className="font-mono text-lg">{finalRisk === null ? "—" : finalRisk.toFixed(4)}</dd>
        </div>
      </dl>
    </aside>
  );
}

export default AnalysisResults;
