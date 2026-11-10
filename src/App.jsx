import React, { useState, useEffect } from 'react';
import { Trophy, TrendingUp, SlidersHorizontal, Calculator, CheckCircle2, X, Plus, Trash2 } from 'lucide-react';

// Helper: Convert American odds to decimal
function americanToDecimal(oddsStr) {
  const odds = parseInt(oddsStr.replace('+', ''));
  if (isNaN(odds)) return 1.0;
  if (odds > 0) {
    return (odds / 100) + 1;
  } else {
    return (100 / Math.abs(odds)) + 1;
  }
}

// Helper: Convert decimal odds back to American string
function decimalToAmerican(decimal) {
  if (decimal <= 1.0) return '+100';
  if (decimal >= 2.0) {
    return `+${Math.round((decimal - 1) * 100)}`;
  } else {
    return `-${Math.round(100 / (decimal - 1))}`;
  }
}

export default function App() {
  const [picks, setPicks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('All Picks');
  const [sortBy, setSortBy] = useState('winPct');
  
  // Parlay Builder State
  const [parlayLegs, setParlayLegs] = useState([]);

  useEffect(() => {
    fetch('/picks.json')
      .then((res) => res.json())
      .then((data) => {
        setPicks(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Could not load picks.json, using fallback data.", err);
        setPicks([
          {
            id: 'demo-1',
            matchup: 'Sample Team A @ Sample Team B',
            team: 'Sample Team A',
            category: 'ML Anchor Leg',
            odds: '-200',
            modelWinPct: '72.5%',
            impliedPct: '66.7%',
            edge: '+5.8%',
            rationale: 'Run "npm run update-picks" in your terminal to fetch live daily parlay legs.',
            commenceTime: new Date().toISOString()
          }
        ]);
        setLoading(false);
      });
  }, []);

  // Toggle pick in/out of parlay builder
  const toggleParlayLeg = (pick) => {
    if (parlayLegs.some(leg => leg.id === pick.id)) {
      setParlayLegs(parlayLegs.filter(leg => leg.id !== pick.id));
    } else {
      setParlayLegs([...parlayLegs, pick]);
    }
  };

  // Calculate Combined Parlay Odds & Joint Win Probability
  const combinedDecimal = parlayLegs.reduce((acc, leg) => acc * americanToDecimal(leg.odds), 1);
  const combinedAmericanOdds = decimalToAmerican(combinedDecimal);
  
  const jointWinProbability = parlayLegs.reduce((acc, leg) => {
    const prob = parseFloat(leg.modelWinPct) / 100;
    return acc * prob;
  }, 1);

  const filteredPicks = picks.filter(pick => {
    if (filter === 'All Picks') return true;
    return pick.category === filter;
  });

  const sortedPicks = [...filteredPicks].sort((a, b) => {
    if (sortBy === 'winPct') {
      return parseFloat(b.modelWinPct) - parseFloat(a.modelWinPct);
    } else if (sortBy === 'edge') {
      return parseFloat(b.edge) - parseFloat(a.edge);
    }
    return 0;
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center font-sans">
        <div className="text-xl animate-pulse text-amber-400">Loading Parlay Leg Vault...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans p-4 md:p-8 pb-32">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header Section */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-800 pb-6 gap-4">
          <div>
            <div className="flex items-center gap-2 text-amber-400 font-semibold tracking-wider text-sm uppercase">
              <Trophy className="w-5 h-5" /> NCAAB Parlay Optimizer & Value Engine
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight mt-1">Value Vault: Parlay Legs</h1>
          </div>
          <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl text-xs text-slate-400 max-w-xs">
            <span className="text-amber-400 font-medium">Informational Notice:</span> Statistical models track value divergence & parlay hit probabilities. Not financial advice.
          </div>
        </header>

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
              <option value="winPct">Highest Win % (Safest Parlay Legs)</option>
              <option value="edge">Best Value Edge</option>
            </select>
          </div>
        </div>

        {/* Picks Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {sortedPicks.length === 0 ? (
            <div className="col-span-2 text-center py-12 bg-slate-900/40 rounded-xl border border-slate-800 text-slate-400">
              No picks available under this category. Try running <code className="text-amber-400">npm run update-picks</code> in your terminal.
            </div>
          ) : (
            sortedPicks.map((pick) => {
              const isSelected = parlayLegs.some(leg => leg.id === pick.id);
              return (
                <div key={pick.id} className={`bg-slate-900 border rounded-2xl p-6 flex flex-col justify-between transition-all shadow-xl ${isSelected ? 'border-amber-500/80 bg-slate-900/90' : 'border-slate-800 hover:border-slate-700'}`}>
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

                    <p className="text-xs text-slate-400 leading-relaxed bg-slate-950/30 p-3 rounded-lg border border-slate-900 mb-4">
                      {pick.rationale}
                    </p>
                  </div>

                  <button
                    onClick={() => toggleParlayLeg(pick)}
                    className={`w-full py-2.5 px-4 rounded-xl text-sm font-bold flex items-center justify-center gap-2 transition-all ${
                      isSelected
                        ? 'bg-amber-500 text-slate-950 hover:bg-amber-400'
                        : 'bg-slate-800 text-slate-200 hover:bg-slate-700 border border-slate-700'
                    }`}
                  >
                    {isSelected ? (
                      <>
                        <CheckCircle2 className="w-4 h-4" /> Added to Parlay Slip
                      </>
                    ) : (
                      <>
                        <Plus className="w-4 h-4" /> Add to Parlay Slip
                      </>
                    )}
                  </button>
                </div>
              );
            })
          )}
        </div>

      </div>

      {/* Sticky Parlay Probability Calculator Footer Drawer */}
      <div className="fixed bottom-0 left-0 right-0 bg-slate-900/95 backdrop-blur-md border-t border-slate-800 shadow-2xl p-4 md:px-8 z-50">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-400">
              <Calculator className="w-6 h-6" />
            </div>
            <div>
              <div className="text-sm font-bold text-white flex items-center gap-2">
                Parlay Probability Slip 
                <span className="bg-amber-500 text-slate-950 text-xs px-2 py-0.5 rounded-full font-extrabold">
                  {parlayLegs.length} Legs
                </span>
              </div>
              <p className="text-xs text-slate-400">
                {parlayLegs.length === 0 
                  ? "Select picks from above to calculate combined parlay odds & joint hit probability."
                  : `${parlayLegs.map(l => l.team).join(' • ')}`}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            {parlayLegs.length > 0 && (
              <>
                <div className="text-right">
                  <div className="text-xs text-slate-400">Combined Odds</div>
                  <div className="text-lg font-black text-emerald-400">{combinedAmericanOdds}</div>
                </div>
                <div className="text-right border-l border-slate-800 pl-6">
                  <div className="text-xs text-slate-400">Joint Win Prob</div>
                  <div className="text-lg font-black text-amber-400">{(jointWinProbability * 100).toFixed(1)}%</div>
                </div>
                <button
                  onClick={() => setParlayLegs([])}
                  className="p-2 text-slate-400 hover:text-red-400 transition-colors"
                  title="Clear Parlay"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </>
            )}
          </div>
        </div>
      </div>

    </div>
  );
}