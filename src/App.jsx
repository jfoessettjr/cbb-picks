import React, { useState, useEffect } from 'react';
import { Trophy, SlidersHorizontal, Calendar as CalendarIcon } from 'lucide-react';

// Helper to get today's date string in Eastern Time
const getTodayEasternStr = () => {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  const parts = formatter.formatToParts(new Date());
  const year = parts.find(p => p.type === 'year').value;
  const month = parts.find(p => p.type === 'month').value;
  const day = parts.find(p => p.type === 'day').value;
  return `${year}-${month}-${day}`;
};

export default function App() {
  const [picksByDate, setPicksByDate] = useState({});
  const [selectedDate, setSelectedDate] = useState('');
  const [availableDates, setAvailableDates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('All Picks');
  const [sortBy, setSortBy] = useState('edge');

  useEffect(() => {
    fetch('/picks.json')
      .then((res) => res.json())
      .then((data) => {
        // data is expected to be an object: { "2026-03-01": [...], "2026-03-02": [...] }
        setPicksByDate(data);
        const dates = Object.keys(data).sort();
        setAvailableDates(dates);

        // Default to today's date in Eastern Time if available, otherwise the first available date
        const todayStr = getTodayEasternStr();
        if (dates.includes(todayStr)) {
          setSelectedDate(todayStr);
        } else if (dates.length > 0) {
          setSelectedDate(dates[0]);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error("Could not load picks.json, using fallback mockup data.", err);
        const todayStr = getTodayEasternStr();
        setPicksByDate({
          [todayStr]: [
            {
              id: 'demo-1',
              matchup: 'Sample Team A @ Sample Team B',
              team: 'Sample Team A',
              category: 'Value Parlay Leg',
              odds: '+150',
              modelWinPct: '45.5%',
              impliedPct: '40.0%',
              edge: '+5.5%',
              rationale: 'Run "npm run update-picks" in your terminal to fetch live daily games.',
              commenceTime: new Date().toISOString()
            }
          ]
        });
        setAvailableDates([todayStr]);
        setSelectedDate(todayStr);
        setLoading(false);
      });
  }, []);

  // Get picks for the currently selected date
  const currentDayPicks = picksByDate[selectedDate] || [];

  const filteredPicks = currentDayPicks.filter(pick => {
    if (filter === 'All Picks') return true;
    return pick.category === filter;
  });

  const sortedPicks = [...filteredPicks].sort((a, b) => {
    if (sortBy === 'edge') {
      return parseFloat(b.edge) - parseFloat(a.edge);
    } else if (sortBy === 'winPct') {
      return parseFloat(b.modelWinPct) - parseFloat(a.modelWinPct);
    }
    return 0;
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center font-sans">
        <div className="text-xl animate-pulse text-amber-400">Loading NCAA Analytical Value Engine...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header Section */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-800 pb-6 gap-4">
          <div>
            <div className="flex items-center gap-2 text-amber-400 font-semibold tracking-wider text-sm uppercase">
              <Trophy className="w-5 h-5" /> NCAAB Analytical Value Engine
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight mt-1">Value Vault: Parlay Legs & Anchors</h1>
          </div>
          <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl text-xs text-slate-400 max-w-xs">
            <span className="text-amber-400 font-medium">Informational Notice:</span> Statistical models track value divergence from Vegas odds. Not financial or gambling advice.
          </div>
        </header>

        {/* Date Selector Bar */}
        <div className="flex items-center gap-3 bg-slate-900 p-4 rounded-xl border border-slate-800 overflow-x-auto">
          <div className="flex items-center gap-2 text-slate-400 text-sm font-medium whitespace-nowrap">
            <CalendarIcon className="w-4 h-4 text-amber-400" /> Select Slate Date (EST):
          </div>
          <div className="flex gap-2">
            {availableDates.map((date) => (
              <button
                key={date}
                onClick={() => setSelectedDate(date)}
                className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-colors whitespace-nowrap ${
                  selectedDate === date
                    ? 'bg-amber-500 text-slate-950 shadow-md'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {new Date(date + 'T00:00:00').toLocaleDateString(undefined, {
                  month: 'short',
                  day: 'numeric',
                  weekday: 'short',
                  timeZone: 'America/New_York'
                })}
              </button>
            ))}
          </div>
        </div>

        {/* Filters & Controls */}
        <div className="flex flex-col sm:flex-row justify-between gap-4 bg-slate-900/50 p-4 rounded-xl border border-slate-800">
          <div className="flex flex-wrap gap-2">
            {['All Picks', 'ML Anchor Leg', 'Value Parlay Leg'].map((cat) => (
              <button
                key={cat}
                onClick={() => setFilter(cat)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === cat
                    ? 'bg-amber-500 text-slate-950 font-bold'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-slate-400" />
            <span className="text-sm text-slate-400">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-sm rounded-lg px-3 py-2 text-white focus:outline-none focus:border-amber-500"
            >
              <option value="edge">Best Value Edge</option>
              <option value="winPct">Highest Win %</option>
            </select>
          </div>
        </div>

        {/* Picks Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {sortedPicks.length === 0 ? (
            <div className="col-span-2 text-center py-12 bg-slate-900/40 rounded-xl border border-slate-800 text-slate-400">
              No picks available for <span className="text-white font-semibold">{selectedDate}</span>.
            </div>
          ) : (
            sortedPicks.map((pick) => (
              <div key={pick.id} className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between hover:border-slate-700 transition-all shadow-xl">
                <div>
                  <div className="flex justify-between items-start mb-3">
                    <span className="px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      {pick.category}
                    </span>
                    <span className="text-2xl font-black text-emerald-400 tracking-tight">
                      {pick.odds}
                    </span>
                  </div>

                  <h3 className="text-lg font-bold text-white mb-1">{pick.team}</h3>
                  <p className="text-sm text-slate-400 mb-4">{pick.matchup}</p>

                  <div className="grid grid-cols-3 gap-2 bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 mb-4 text-center">
                    <div>
                      <div className="text-xs text-slate-500 font-medium">Model Win</div>
                      <div className="text-sm font-bold text-white mt-0.5">{pick.modelWinPct}</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 font-medium">Vegas Implied</div>
                      <div className="text-sm font-bold text-slate-300 mt-0.5">{pick.impliedPct}</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 font-medium">Value Edge</div>
                      <div className="text-sm font-bold text-amber-400 mt-0.5">{pick.edge}</div>
                    </div>
                  </div>

                  <p className="text-xs text-slate-400 leading-relaxed bg-slate-950/30 p-3 rounded-lg border border-slate-900">
                    {pick.rationale}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>

      </div>
    </div>
  );
}