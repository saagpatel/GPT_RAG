export function StatusPill({ status }: { status: string | boolean }) {
  const label = typeof status === "boolean" ? (status ? "ready" : "blocked") : status;
  const tone =
    label === "ready" || label === "completed"
      ? "positive"
      : label === "running" || label === "pending"
        ? "neutral"
        : label === "blocked" || label === "failed"
          ? "negative"
          : "neutral";
  return <span className={`status-pill ${tone}`}>{label}</span>;
}
