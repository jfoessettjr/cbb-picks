import { useEffect, useMemo, useState } from "react";
import "./App.css";

export default function App() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}picks/latest.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, []);

  const top5 = useMemo(() => (data?.picks ?? []).slice(0, 5), [data]);

  return (
    <div style={{ maxWidth: 980, margin: "0 auto", padding: 16, fontFamily: "system-ui, Arial" }}>
      <h1 style={{ marginBottom: 6 }}>NCAAB Top 5 Picks</h1>
      <div style={{ opacity: 0.8, marginBottom: 16 }}>
        Date: <b>{data?.date ?? "—"}</b> • Last updated: <b>{data?.generated_at ?? "—"}</b>
      </div>

      {err && (
        <div style={{ color: "crimson", marginBottom: 12 }}>
          Error loading picks: {err}
        </div>
      )}

      {!data && !err && <div>Loading…</div>}

      {data && (
        <>
          <h2 style={{ marginTop: 0 }}>Top 5 Most Likely Winners</h2>

          <div style={{ display: "grid", gap: 12 }}>
            {top5.length === 0 && (
              <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 14 }}>
                No games found for this date (or the data source didn&apos;t return games).
              </div>
            )}

            {top5.map((p) => (
              <div
                key={p.game_id}
                style={{
                  border: "1px solid #ddd",
                  borderRadius: 12,
                  padding: 14,
                  background: "white",
                }}
              >
                <div style={{ fontSize: 18, fontWeight: 800 }}>
                  {p.away_team} @ {p.home_team}
                </div>

                <div style={{ marginTop: 6 }}>
                  Pick: <b>{p.pick_team}</b> • Confidence:{" "}
                  <b>{Math.round(p.win_prob * 100)}%</b>
                </div>

                <div style={{ marginTop: 8, opacity: 0.75, fontSize: 13 }}>
                  Elo (raw): {Math.round(p.home_elo_raw)} (home) vs {Math.round(p.away_elo)} (away) •
                  HCA: +{p.hca} • Home Elo w/ HCA: {Math.round(p.home_elo_with_hca)}
                </div>
              </div>
            ))}
          </div>

          <h2 style={{ marginTop: 22 }}>All Games</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 520 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #eee" }}>
                  <th align="left" style={{ padding: "8px 0" }}>Matchup</th>
                  <th align="left" style={{ padding: "8px 0" }}>Pick</th>
                  <th align="left" style={{ padding: "8px 0" }}>Win %</th>
                </tr>
              </thead>
              <tbody>
                {(data.picks ?? []).map((p) => (
                  <tr key={`row-${p.game_id}`} style={{ borderTop: "1px solid #f0f0f0" }}>
                    <td style={{ padding: "10px 0" }}>
                      {p.away_team} @ {p.home_team}
                    </td>
                    <td>{p.pick_team}</td>
                    <td>{Math.round(p.win_prob * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 18, opacity: 0.75, fontSize: 13 }}>
            Model: Elo (season-to-date) + home-court advantage.
          </div>
        </>
      )}
    </div>
  );
}

