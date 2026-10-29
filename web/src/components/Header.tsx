export default function Header({ localities }: { localities: number | null }) {
  return (
    <>
      <nav className="cc-nav">
        <div className="cc-nav-left">
          <span className="cc-logo">
            ✦ <b>cc</b>
          </span>
          <span className="cc-pill-nav">📍 Bangalore</span>
        </div>
        <span className="cc-nav-mid">Real Zomato data · AI-ranked picks</span>
        {localities ? (
          <span className="cc-pill-nav cc-pill-live">
            <span className="cc-dot" />
            {localities} neighbourhoods
          </span>
        ) : null}
      </nav>
      <section className="cc-hero">
        <span className="cc-badge">✦ Your AI dining concierge</span>
        <h1 className="cc-title">
          Curated <span className="cc-accent">Cravings</span>
        </h1>
        <p className="cc-sub">
          Tell us what you are in the mood for. We narrow Bangalore's restaurants to your
          must-haves, then AI ranks the shortlist and tells you why each place fits.
        </p>
      </section>
    </>
  );
}
