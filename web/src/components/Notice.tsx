/** A soft banner (notice_html in the Streamlit UI). React renders every line as text. */
export default function Notice({
  kind,
  title,
  lines = [],
}: {
  kind: "warn" | "error";
  title: string;
  lines?: string[];
}) {
  return (
    <div className={`cc-notice cc-notice-${kind}`} role={kind === "error" ? "alert" : "status"}>
      <b>{title}</b>
      {lines.length > 1 ? (
        <ul>
          {lines.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      ) : (
        lines[0]
      )}
    </div>
  );
}

export function Spinner({ text }: { text: string }) {
  return (
    <div className="cc-spinner" role="status">
      <span className="cc-spinner-dot" aria-hidden="true" />
      {text}
    </div>
  );
}
