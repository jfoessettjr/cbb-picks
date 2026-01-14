import { useEffect, useMemo, useState } from "react";
import "./App.css";

function isoTodayLocal() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export default function App() {
  const urlParams = new URLSearchParams(window.location.search);
  const dateFromQuery = urlParams.get("date");
  const [selectedDate, setSelectedDate] = useState(dateFromQuery || isoTodayLocal());

  const [manifest, setManifest] = useState(null);
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");

  // Load manifest for available dates
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}picks/manifest.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setManifest)
      .catch(() => setManifest(null));
  }, []);

  // Load data for selected date
  useEffect(() => {
    setErr("");
    setData(null);

    const url = `${import.meta.env.BASE_URL}picks/${selectedDate}.json`;

    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`No file for ${selectedDate} (HTTP ${r.status})`))))
      .then(setData)
      .catch(async () => {
        try {
          const r2 = await fetch(`${import.meta.env.BASE_URL}picks/latest.json`);
          if (!r2.ok) throw new Error(`HTTP ${r2.status}`);
          const j2 = await r2.json();
          setData(j2);
          setErr(`No data file for ${selectedDate}. Showing latest available instead.`);
        } catch (e2) {
          setErr(`Could not load picks. ${String(e2)}`);
        }
      });
  }, [selectedDate]);

  const top5 = useMemo(() => (data?.picks ?? []).slice(0, 5), [data]);

  const onChangeDate = (e) => {
    const d = e.target.value;
    setSelectedDate(d);

    const u = new URL(window.location.href);
    u.searchParams.set("date", d);
    window.history.replaceState({}, "", u.toString());
  };

  const cardStyle = {
    border: "1px solid var(--border)",
    borderRadius: 12,
    padding: 14,
    background: "var(--card-bg)",
  };

  return (
    <div style={{ maxWidth: 980, margin: "0 auto", padding: 16, fontFamily: "system-ui, Arial" }}>
      <h1 style={{ marginBottom: 8 }}>NCAAB Top 5 Picks</h1>

      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <label style={{ fontWeight: 700 }}>
          Date:&nbsp;
          <input
            type="date"
            value={selectedDate}
            onChange={onChangeDate}
            min={manifest?.min_date || undefined}
            max={manifest?.max_date || undefined}
            style={{ padding: 6, borderRadius: 8 }}
          />
        </label>

        <div style={{ opacity: 0.8 }}>
          Showing: <b>{data?.date ?? "—"}</b> • Season: <b>{data?.season ?? "—"}</b> • Updated:{" "}
          <b>{data?.generated_at ?? "—"}</b>
        </div>
      </div>

      {manifest?.min_date && manifest?.max_date && (
        <div style={{ opacity: 0.75, marginBottom: 12 }}>
          Available dates: <b>{manifest.min_date}</b> to <b>{manifest.max_date}</b>
        </div>
      )}

      {err && (
        <div style={{ color: "crimson", marginBottom: 12 }}>
          {err}
        </div>
      )}

      {!data && !err && <div>Loading…</div>}

      {data && (
        <>
          <h2 style={{ marginTop: 0 }}>Top 5 Most Likely Winners</h2>

          <div style={{ display: "grid", gap: 12 }}>
            {top5.length === 0 && (
              <div style={cardStyle}>No games found for this date.</div>
            )}

            {top5.map((p) => (
              <div key={p.game_id} style={cardStyle}>
                <div style={{ fontSize: 18, fontWeight: 800 }}>
                  {p.away_team} @ {p.home_team}
                </div>

                <div style={{ marginTop: 6 }}>
                  Pick: <b>{p.pick_team}</b> • Confidence: <b>{Math.round(p.win_prob * 100)}%</b>
                </div>

                <div style={{ marginTop: 8, opacity: 0.75, fontSize: 13 }}>
                  Elo (raw): {Math.round(p.home_elo_raw)} vs {Math.round(p.away_elo)} • HCA: +{p.hca}
                </div>
              </div>
            ))}
          </div>

          <h2 style={{ marginTop: 22 }}>All Games</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 520 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid var(--table-border)" }}>
                  <th align="left" style={{ padding: "8px 0" }}>Matchup</th>
                  <th align="left" style={{ padding: "8px 0" }}>Pick</th>
                  <th align="left" style={{ padding: "8px 0" }}>Win %</th>
                </tr>
              </thead>
              <tbody>
                {(data.picks ?? []).map((p) => (
                  <tr key={`row-${p.game_id}`} style={{ borderTop: "1px solid var(--table-border)" }}>
                    <td style={{ padding: "10px 0" }}>{p.away_team} @ {p.home_team}</td>
                    <td>{p.pick_team}</td>
                    <td>{Math.round(p.win_prob * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 18, opacity: 0.75, fontSize: 13 }}>
            Share a date directly with <code>?date=YYYY-MM-DD</code>
          </div>
        </>
      )}
    </div>
  );
}
