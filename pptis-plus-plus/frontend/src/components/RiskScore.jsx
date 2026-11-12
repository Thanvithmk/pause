function RiskScore({ score, loading = false, failed = false, className = "" }) {
  if (loading) {
    return <p className={`text-xs text-slate-400 ${className}`}>Analyzing risk…</p>;
  }
  if (failed || score === null || score === undefined || Number.isNaN(Number(score))) {
    return <p className={`text-xs text-slate-400 ${className}`}>Risk score unavailable</p>;
  }
  return (
    <p className={`text-xs font-mono text-slate-300 ${className}`}>
      Risk Score: {Number(score).toFixed(4)} / 1.00
    </p>
  );
}

export default RiskScore;
