import { MAX_CHIPS, formatCost, formatRating } from "../format";
import type { Recommendation } from "../types";

/** One restaurant: rank, name, booking tags, rating, cost, place, cuisines, why, dishes.
 * Every value from the data or the LLM is rendered as text, never as markup (EC-UI-20). */
export default function ResultCard({ item }: { item: Recommendation }) {
  const extraCuisines = item.cuisines.length - MAX_CHIPS;
  return (
    <article className="cc-card">
      <div className="cc-rank">{String(item.rank).padStart(2, "0")}</div>
      <div className="cc-body">
        <div className="cc-head">
          <h3 className="cc-name">{item.name}</h3>
          <div className="cc-tags">
            {item.book_table && <span className="cc-tag cc-tag-green">Table booking</span>}
            {item.online_order && <span className="cc-tag">Online ordering</span>}
          </div>
        </div>
        <div className="cc-row">
          <span className={`cc-pill ${item.rating === null ? "cc-pill-new" : "cc-pill-rating"}`}>
            {formatRating(item.rating, item.votes)}
          </span>
          <span className="cc-pill">{formatCost(item.cost_for_two)}</span>
          <span className="cc-pill">📍 {item.location}</span>
          {item.rest_type.length > 0 && <span className="cc-pill">{item.rest_type.join(", ")}</span>}
        </div>
        {item.cuisines.length > 0 && (
          <div className="cc-row">
            {item.cuisines.slice(0, MAX_CHIPS).map((c) => (
              <span key={c} className="cc-chip">
                {c}
              </span>
            ))}
            {extraCuisines > 0 && (
              <span className="cc-chip cc-chip-more">+{extraCuisines} more</span>
            )}
          </div>
        )}
        <p className="cc-why">
          <span className="cc-why-label">
            {item.source === "llm" ? "✦ Why it fits" : "Ranked by rating and popularity"}
          </span>
          {item.explanation}
        </p>
        {item.popular_dishes.length > 0 && (
          <div className="cc-known">
            <b>Known for</b> {item.popular_dishes.join(" · ")}
          </div>
        )}
      </div>
    </article>
  );
}
