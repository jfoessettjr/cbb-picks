import React, { useState, useMemo } from 'react';
import { 
  Trophy, 
  TrendingUp, 
  Percent, 
  ShieldAlert, 
  Search, 
  Filter, 
  PlusCircle, 
  Bookmark, 
  BookmarkCheck, 
  LayoutGrid, 
  List, 
  Flame, 
  Zap, 
  Calculator, 
  ExternalLink,
  ChevronDown,
  Info,
  CheckCircle2,
  Trash2,
  BarChart2
} from 'lucide-react';

// Utility to convert American Plus Odds (+250) to Implied Probability
const calcImpliedProb = (americanOdds) => {
  if (americanOdds <= 0) return 50.0;
  return Number(((100 / (americanOdds + 100)) * 100).toFixed(1));
};

// Initial Sample NCAA Longshot Data
const INITIAL_PICKS = [
  {
    id: 1,
    team: "Oakland Golden Grizzlies",
    opponent: "#3 Kentucky Wildcats",
    matchup: "Oakland vs Kentucky",
    category: "Upset Specials",
    pickType: "Moneyline Upset",
    odds: 450,
    modelWinPct: 24.5,
    rationale: "Oakland runs a unique 3-2 matchup zone that disrupts rhythm shooters. Ranked top 15 in 3PT attempt rate. Kentucky allows 38% from deep on neutral courts.",
    bookmaker: "DraftKings",
    gameTime: "Tonight, 7:10 PM ET",
    conference: "Horizon / SEC",
    isHot: true
  },
  {
    id: 2,
    team: "McNeese State Cowboys",
    opponent: "#5 Gonzaga Bulldogs",
    matchup: "McNeese St vs Gonzaga",
    category: "Upset Specials",
    pickType: "Moneyline Upset",
    odds: 310,
    modelWinPct: 31.8,
    rationale: "McNeese leads the nation in turnover forcing % (24.2%). Gonzaga's backcourt has turned it over on 19% of possessions in high-major non-conference games.",
    bookmaker: "FanDuel",
    gameTime: "Tonight, 9:40 PM ET",
    conference: "Southland / WCC",
    isHot: true
  },
  {
    id: 3,
    team: "Trey Townsend (Oakland)",
    opponent: "vs Kentucky",
    matchup: "Trey Townsend Over 21.5 Pts",
    category: "Player Props",
    pickType: "Points Prop Over",
    odds: 225,
    modelWinPct: 38.0,
    rationale: "Townsend usage rate expands to 34% in postseason settings. Kentucky ranks 210th in defensive rebounding rate, yielding high second-chance points.",
    bookmaker: "BetMGM",
    gameTime: "Tonight, 7:10 PM ET",
    conference: "Horizon",
    isHot: false
  },
  {
    id: 4,
    team: "James Madison Dukes",
    opponent: "#6 Wisconsin Badgers",
    matchup: "James Madison vs Wisconsin",
    category: "Upset Specials",
    pickType: "Spread (+5.5) & ML",
    odds: 185,
    modelWinPct: 42.0,
    rationale: "JMU riding a 13-game win streak with top-20 transition offense rating. Wisconsin struggles defending early-clock fast breaks.",
    bookmaker: "Caesars",
    gameTime: "Tomorrow, 6:50 PM ET",
    conference: "Sun Belt / Big Ten",
    isHot: false
  },
  {
    id: 5,
    team: "Samford Bulldogs",
    opponent: "#4 Kansas Jayhawks",
    matchup: "Samford vs Kansas",
    category: "Parlay Builders",
    pickType: "1st Half Team Total Over 36.5",
    odds: 240,
    modelWinPct: 35.5,
    rationale: "Bucky Ball full-court pressure forces early 1st half pace (74.2 possessions/game). Kansas depth is compromised due to key center injury.",
    bookmaker: "FanDuel",
    gameTime: "Tomorrow, 9:55 PM ET",
    conference: "SoCon / Big 12",
    isHot: true
  },
  {
    id: 6,
    team: "Grand Canyon Antelopes",
    opponent: "#5 Saint Mary's Gaels",
    matchup: "Grand Canyon vs Saint Mary's",
    category: "Parlay Builders",
    pickType: "Moneyline Upset",
    odds: 175,
    modelWinPct: 44.2,
    rationale: "GCU draws free throws at the 4th highest rate in NCAA (24.8 FTA/gm). Saint Mary's aggressive post defense leads to early foul trouble.",
    bookmaker: "DraftKings",
    gameTime: "Tomorrow, 10:05 PM ET",
    conference: "WAC / WCC",
    isHot: false
  }
];

export default function App() {
  const [picks, setPicks] = useState(INITIAL_PICKS);
  const [activeCategory, setActiveCategory] = useState("All Picks");
  const [sortBy, setSortBy] = useState("winPct"); // winPct, odds, edge
  const [searchQuery, setSearchQuery] = useState("");
  const [viewMode, setViewMode] = useState("grid"); // grid or table
  const [watchlist, setWatchlist] = useState([]);
  const [parlayLegs, setParlayLegs] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);

  // New Pick Form State
  const [newPick, setNewPick] = useState({
    team: "",
    opponent: "",
    matchup: "",
    category: "Upset Specials",
    pickType: "Moneyline Upset",
    odds: 250,
    modelWinPct: 32.0,
    rationale: "",
    bookmaker: "DraftKings",
    gameTime: "Tonight"
  });

  // Calculate Implied Prob & Edge for any pick object
  const processPickData = (item) => {
    const implied = calcImpliedProb(item.odds);
    const edge = Number((item.modelWinPct - implied).toFixed(1));
    return { ...item, impliedProb: implied, edge };
  };

  // Process all picks with dynamic math
  const processedPicks = useMemo(() => {
    return picks.map(processPickData);
  }, [picks]);

  // Filter & Sort Logic
  const filteredPicks = useMemo(() => {
    return processedPicks
      .filter(item => {
        const matchesCategory = activeCategory === "All Picks" || item.category === activeCategory;
        const matchesSearch = item.matchup.toLowerCase().includes(searchQuery.toLowerCase()) ||
                              item.team.toLowerCase().includes(searchQuery.toLowerCase()) ||
                              item.pickType.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesCategory && matchesSearch;
      })
      .sort((a, b) => {
        if (sortBy === "winPct") return b.modelWinPct - a.modelWinPct;
        if (sortBy === "odds") return b.odds - a.odds;
        if (sortBy === "edge") return b.edge - a.edge;
        return 0;
      });
  }, [processedPicks, activeCategory, searchQuery, sortBy]);

  // Dynamic Summary Metrics
  const summaryMetrics = useMemo(() => {
    if (processedPicks.length === 0) return { total: 0, highestWinPct: 0, avgOdds: 0, bestEdge: 0 };
    const total = processedPicks.length;
    const highestWinPct = Math.max(...processedPicks.map(p => p.modelWinPct));
    const avgOddsVal = Math.round(processedPicks.reduce((acc, p) => acc + p.odds, 0) / total);
    const bestEdge = Math.max(...processedPicks.map(p => p.edge));
    return {
      total,
      highestWinPct,
      avgOdds: `+${avgOddsVal}`,
      bestEdge: `+${bestEdge.toFixed(1)}%`
    };
  }, [processedPicks]);

  // Bookmark Toggle
  const toggleBookmark = (id) => {
    setWatchlist(prev => 
      prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]
    );
  };

  // Add to Parlay Builder
  const toggleParlayLeg = (pick) => {
    setParlayLegs(prev => {
      const exists = prev.some(item => item.id === pick.id);
      if (exists) return prev.filter(item => item.id !== pick.id);
      return [...prev, pick];
    });
  };

  // Calculate Parlay Stats
  const parlayStats = useMemo(() => {
    if (parlayLegs.length === 0) return { combinedOdds: 0, combinedProb: 0 };
    let decimalOdds = 1;
    let combinedProb = 1;
    parlayLegs.forEach(leg => {
      const dec = (leg.odds / 100) + 1;
      decimalOdds *= dec;
      combinedProb *= (leg.modelWinPct / 100);
    });
    const finalAmerican = Math.round((decimalOdds - 1) * 100);
    return {
      combinedOdds: `+${finalAmerican}`,
      combinedProb: (combinedProb * 100).toFixed(2)
    };
  }, [parlayLegs]);

  // Handle Form Submit
  const handleCreatePick = (e) => {
    e.preventDefault();
    if (!newPick.team || !newPick.odds || !newPick.modelWinPct) return;
    
    const created = {
      ...newPick,
      id: Date.now(),
      odds: Number(newPick.odds),
      modelWinPct: Number(newPick.modelWinPct),
      matchup: `${newPick.team} vs ${newPick.opponent || 'Opponent'}`,
      isHot: Number(newPick.modelWinPct) >= 35
    };

    setPicks([created, ...picks]);
    setShowAddModal(false);
    setNewPick({
      team: "",
      opponent: "",
      matchup: "",
      category: "Upset Specials",
      pickType: "Moneyline Upset",
      odds: 250,
      modelWinPct: 32.0,
      rationale: "",
      bookmaker: "DraftKings",
      gameTime: "Tonight"
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased pb-16">
      
      {/* HEADER SECTION */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            
            {/* Branding */}
            <div className="flex items-center space-x-3">
              <div className="bg-emerald-500/10 p-2.5 rounded-xl border border-emerald-500/30 text-emerald-400">
                <Trophy className="h-7 w-7 text-emerald-400" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl sm:text-2xl font-black tracking-tight text-white uppercase">
                    VALUE VAULT <span className="text-emerald-400 font-normal text-sm sm:text-base border border-emerald-500/30 px-2 py-0.5 rounded-md bg-emerald-950/40">NCAA MBB</span>
                  </h1>
                </div>
                <p className="text-xs text-slate-400">Small Longshot Picks Ranked by Model Hit Probability & Edge</p>
              </div>
            </div>

            {/* Quick Actions & Modal Trigger */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowAddModal(true)}
                className="flex items-center gap-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold px-4 py-2 rounded-lg transition text-sm shadow-lg shadow-emerald-500/20"
              >
                <PlusCircle className="h-4 w-4" />
                <span>Add Longshot Pick</span>
              </button>
            </div>

          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 space-y-6">

        {/* METRICS SUMMARY CARDS */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex items-center justify-between shadow-sm">
            <div>
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Active Longshots</p>
              <h3 className="text-2xl font-extrabold text-white mt-1">{summaryMetrics.total} <span className="text-xs font-normal text-slate-400">Picks</span></h3>
            </div>
            <div className="p-3 bg-slate-800/80 rounded-lg text-emerald-400">
              <Zap className="h-6 w-6" />
            </div>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex items-center justify-between shadow-sm">
            <div>
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Top Model Win %</p>
              <h3 className="text-2xl font-extrabold text-emerald-400 mt-1">{summaryMetrics.highestWinPct}%</h3>
            </div>
            <div className="p-3 bg-slate-800/80 rounded-lg text-emerald-400">
              <Percent className="h-6 w-6" />
            </div>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex items-center justify-between shadow-sm">
            <div>
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Average Vault Odds</p>
              <h3 className="text-2xl font-extrabold text-amber-400 mt-1">{summaryMetrics.avgOdds}</h3>
            </div>
            <div className="p-3 bg-slate-800/80 rounded-lg text-amber-400">
              <TrendingUp className="h-6 w-6" />
            </div>
          </div>

          <div className="bg-emerald-950/20 border border-emerald-500/20 rounded-xl p-4 flex items-center justify-between shadow-sm">
            <div>
              <p className="text-xs font-medium text-emerald-400/80 uppercase tracking-wider">Max Value Edge</p>
              <h3 className="text-2xl font-extrabold text-emerald-300 mt-1">{summaryMetrics.bestEdge}</h3>
            </div>
            <div className="p-3 bg-emerald-500/10 rounded-lg text-emerald-400 border border-emerald-500/30">
              <BarChart2 className="h-6 w-6" />
            </div>
          </div>

        </div>

        {/* DISCLAIMER BANNER */}
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-lg p-3 flex items-start gap-3 text-xs text-slate-400">
          <ShieldAlert className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <p>
            <strong className="text-slate-300">Informational & Entertainment Purpose Only:</strong> Win probabilities are derived from analytical regression models and historical data simulations. Longshot betting carries inherent volatility. Always practice responsible wagering.
          </p>
        </div>

        {/* FILTERING, SEARCH & SORT TOOLBAR */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            
            {/* Category Tabs */}
            <div className="flex flex-wrap gap-2">
              {["All Picks", "Upset Specials", "Player Props", "Parlay Builders"].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setActiveCategory(cat)}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
                    activeCategory === cat
                      ? "bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/10"
                      : "bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Search & Sort & View Mode */}
            <div className="flex flex-wrap items-center gap-3">
              
              {/* Search Bar */}
              <div className="relative flex-1 min-w-[180px]">
                <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                <input
                  type="text"
                  placeholder="Search team or bet..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                />
              </div>

              {/* Sort Dropdown */}
              <div className="flex items-center space-x-2 bg-slate-950 border border-slate-800 rounded-lg px-2 py-1">
                <Filter className="h-3.5 w-3.5 text-slate-400" />
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="bg-transparent text-xs text-slate-300 focus:outline-none cursor-pointer pr-1"
                >
                  <option value="winPct" className="bg-slate-900">Rank: Win %</option>
                  <option value="edge" className="bg-slate-900">Rank: Best Edge</option>
                  <option value="odds" className="bg-slate-900">Rank: Highest Odds</option>
                </select>
              </div>

              {/* Layout Toggle */}
              <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5">
                <button
                  onClick={() => setViewMode("grid")}
                  className={`p-1.5 rounded ${viewMode === "grid" ? "bg-slate-800 text-emerald-400" : "text-slate-400 hover:text-slate-200"}`}
                  title="Grid View"
                >
                  <LayoutGrid className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setViewMode("table")}
                  className={`p-1.5 rounded ${viewMode === "table" ? "bg-slate-800 text-emerald-400" : "text-slate-400 hover:text-slate-200"}`}
                  title="Table View"
                >
                  <List className="h-4 w-4" />
                </button>
              </div>

            </div>

          </div>
        </div>

        {/* PARLAY BUILDER SLOT (If legs selected) */}
        {parlayLegs.length > 0 && (
          <div className="bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border border-emerald-500/40 rounded-xl p-4 transition-all">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <Calculator className="h-5 w-5 text-emerald-400" />
                  <h3 className="font-bold text-white text-sm">Longshot Parlay Builder ({parlayLegs.length} Legs)</h3>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {parlayLegs.map(leg => (
                    <span key={leg.id} className="inline-flex items-center gap-1 bg-slate-800 text-slate-200 text-xs px-2.5 py-1 rounded-md border border-slate-700">
                      {leg.team} ({leg.odds > 0 ? `+${leg.odds}` : leg.odds})
                      <button onClick={() => toggleParlayLeg(leg)} className="text-slate-400 hover:text-red-400 ml-1">×</button>
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-4 border-t sm:border-t-0 border-slate-800 pt-3 sm:pt-0">
                <div className="text-right">
                  <div className="text-xs text-slate-400">Combined Odds</div>
                  <div className="text-lg font-black text-amber-400">{parlayStats.combinedOdds}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-slate-400">Model Win Prob</div>
                  <div className="text-lg font-black text-emerald-400">{parlayStats.combinedProb}%</div>
                </div>
                <button
                  onClick={() => setParlayLegs([])}
                  className="text-xs text-slate-400 hover:text-slate-200 underline pl-2"
                >
                  Clear All
                </button>
              </div>
            </div>
          </div>
        )}

        {/* PICKS DISPLAY AREA */}
        {filteredPicks.length === 0 ? (
          <div className="text-center py-16 bg-slate-900/50 border border-slate-800 rounded-xl">
            <Search className="h-10 w-10 text-slate-600 mx-auto mb-3" />
            <h3 className="text-base font-semibold text-slate-300">No Longshots Found</h3>
            <p className="text-xs text-slate-500 mt-1">Try adjusting your filters or search terms.</p>
          </div>
        ) : viewMode === "grid" ? (
          /* GRID VIEW */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {filteredPicks.map((pick, idx) => {
              const isBookmarked = watchlist.includes(pick.id);
              const inParlay = parlayLegs.some(l => l.id === pick.id);

              return (
                <div 
                  key={pick.id}
                  className="bg-slate-900 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition flex flex-col justify-between shadow-sm relative group"
                >
                  {/* Top Header Card */}
                  <div>
                    <div className="flex items-start justify-between gap-2 mb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-extrabold text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                          #{idx + 1}
                        </span>
                        <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-500/20">
                          {pick.category}
                        </span>
                        {pick.isHot && (
                          <span className="flex items-center gap-1 text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
                            <Flame className="h-3 w-3 fill-amber-400" /> HOT VALUE
                          </span>
                        )}
                      </div>
                      
                      <button
                        onClick={() => toggleBookmark(pick.id)}
                        className="text-slate-500 hover:text-amber-400 transition"
                        title={isBookmarked ? "Remove from Watchlist" : "Save Pick"}
                      >
                        {isBookmarked ? (
                          <BookmarkCheck className="h-5 w-5 text-amber-400 fill-amber-400/20" />
                        ) : (
                          <Bookmark className="h-5 w-5" />
                        )}
                      </button>
                    </div>

                    {/* Team & Pick Title */}
                    <h3 className="text-lg font-bold text-white tracking-wide">{pick.team}</h3>
                    <p className="text-xs text-slate-400 mb-4">{pick.pickType} • vs {pick.opponent}</p>

                    {/* ODDS & METRICS BOX */}
                    <div className="grid grid-cols-3 gap-2 bg-slate-950/80 p-3 rounded-lg border border-slate-800/80 mb-4 text-center">
                      <div>
                        <p className="text-[10px] uppercase font-bold text-slate-500">Odds</p>
                        <p className="text-base font-black text-amber-400 mt-0.5">+{pick.odds}</p>
                      </div>
                      <div className="border-x border-slate-800">
                        <p className="text-[10px] uppercase font-bold text-slate-500">Model Win %</p>
                        <p className="text-base font-black text-emerald-400 mt-0.5">{pick.modelWinPct}%</p>
                      </div>
                      <div>
                        <p className="text-[10px] uppercase font-bold text-slate-500">Value Edge</p>
                        <p className={`text-base font-black mt-0.5 ${pick.edge >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                          +{pick.edge}%
                        </p>
                      </div>
                    </div>

                    {/* Rationale Breakdown */}
                    <div className="space-y-1 mb-4">
                      <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                        <Info className="h-3 w-3 text-emerald-400" /> Analytical Rationale
                      </p>
                      <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/30 p-2.5 rounded border border-slate-800/50">
                        {pick.rationale}
                      </p>
                    </div>
                  </div>

                  {/* Card Bottom Meta & Actions */}
                  <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-slate-300">{pick.bookmaker}</span>
                      <span>•</span>
                      <span className="text-[11px] text-slate-500">{pick.gameTime}</span>
                    </div>

                    <button
                      onClick={() => toggleParlayLeg(pick)}
                      className={`text-xs font-medium px-2.5 py-1 rounded transition border ${
                        inParlay 
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/40" 
                          : "bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700"
                      }`}
                    >
                      {inParlay ? "Added to Parlay" : "+ Build Parlay"}
                    </button>
                  </div>

                </div>
              );
            })}
          </div>
        ) : (
          /* TABLE VIEW */
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-950 text-slate-400 text-xs uppercase font-bold border-b border-slate-800">
                    <th className="py-3 px-4">Rank</th>
                    <th className="py-3 px-4">Matchup / Pick</th>
                    <th className="py-3 px-4">Category</th>
                    <th className="py-3 px-4 text-center">Book Odds</th>
                    <th className="py-3 px-4 text-center">Implied Prob</th>
                    <th className="py-3 px-4 text-center">Model Hit %</th>
                    <th className="py-3 px-4 text-center">Edge</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-xs">
                  {filteredPicks.map((pick, idx) => {
                    const isBookmarked = watchlist.includes(pick.id);
                    const inParlay = parlayLegs.some(l => l.id === pick.id);

                    return (
                      <tr key={pick.id} className="hover:bg-slate-800/40 transition">
                        <td className="py-3 px-4 font-bold text-slate-400">#{idx + 1}</td>
                        <td className="py-3 px-4">
                          <div className="font-bold text-slate-100">{pick.team}</div>
                          <div className="text-[11px] text-slate-400">{pick.pickType} ({pick.opponent})</div>
                        </td>
                        <td className="py-3 px-4">
                          <span className="bg-slate-800 text-slate-300 px-2 py-0.5 rounded text-[11px]">
                            {pick.category}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center font-extrabold text-amber-400">+{pick.odds}</td>
                        <td className="py-3 px-4 text-center text-slate-400">{pick.impliedProb}%</td>
                        <td className="py-3 px-4 text-center font-bold text-emerald-400">{pick.modelWinPct}%</td>
                        <td className="py-3 px-4 text-center">
                          <span className="font-extrabold text-emerald-400 bg-emerald-950/60 border border-emerald-500/20 px-2 py-0.5 rounded">
                            +{pick.edge}%
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right">
                          <div className="flex items-center justify-end gap-2">
                            <button
                              onClick={() => toggleBookmark(pick.id)}
                              className="text-slate-400 hover:text-amber-400 p-1"
                            >
                              <Bookmark className={`h-4 w-4 ${isBookmarked ? "fill-amber-400 text-amber-400" : ""}`} />
                            </button>
                            <button
                              onClick={() => toggleParlayLeg(pick)}
                              className={`px-2 py-1 rounded text-[11px] font-medium border ${
                                inParlay ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/40" : "bg-slate-800 text-slate-300 border-slate-700"
                              }`}
                            >
                              {inParlay ? "Added" : "+ Parlay"}
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </main>

      {/* ADD LONGSHOT PICK MODAL */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4 relative">
            
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <PlusCircle className="h-5 w-5 text-emerald-400" /> Add Custom NCAA Longshot
              </h3>
              <button 
                onClick={() => setShowAddModal(false)}
                className="text-slate-400 hover:text-slate-200 text-lg font-bold"
              >
                ×
              </button>
            </div>

            <form onSubmit={handleCreatePick} className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">Underdog Team</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Duquesne"
                    value={newPick.team}
                    onChange={(e) => setNewPick({ ...newPick, team: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">Opponent</label>
                  <input
                    type="text"
                    placeholder="e.g. BYU"
                    value={newPick.opponent}
                    onChange={(e) => setNewPick({ ...newPick, opponent: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">Category</label>
                  <select
                    value={newPick.category}
                    onChange={(e) => setNewPick({ ...newPick, category: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                  >
                    <option value="Upset Specials">Upset Specials</option>
                    <option value="Player Props">Player Props</option>
                    <option value="Parlay Builders">Parlay Builders</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">Pick Type</label>
                  <input
                    type="text"
                    placeholder="e.g. Moneyline Upset"
                    value={newPick.pickType}
                    onChange={(e) => setNewPick({ ...newPick, pickType: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">Plus Odds (+)</label>
                  <input
                    type="number"
                    required
                    min="100"
                    placeholder="350"
                    value={newPick.odds}
                    onChange={(e) => setNewPick({ ...newPick, odds: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-amber-400 font-bold focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">Model Win %</label>
                  <input
                    type="number"
                    required
                    step="0.1"
                    min="1"
                    max="99"
                    placeholder="30.5"
                    value={newPick.modelWinPct}
                    onChange={(e) => setNewPick({ ...newPick, modelWinPct: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-emerald-400 font-bold focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              {/* Dynamic Math Preview */}
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex justify-between items-center text-xs">
                <div>
                  <span className="text-slate-400">Vegas Implied: </span>
                  <span className="font-bold text-slate-200">{calcImpliedProb(Number(newPick.odds))}%</span>
                </div>
                <div>
                  <span className="text-slate-400">Calculated Edge: </span>
                  <span className="font-bold text-emerald-400">
                    +{(Number(newPick.modelWinPct) - calcImpliedProb(Number(newPick.odds))).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1">Analytical Rationale</label>
                <textarea
                  rows="3"
                  placeholder="Explain the matchup breakdown, tempo, or stat edge..."
                  value={newPick.rationale}
                  onChange={(e) => setNewPick({ ...newPick, rationale: e.target.value })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                ></textarea>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg text-xs font-semibold hover:bg-slate-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-emerald-500 text-slate-950 rounded-lg text-xs font-bold hover:bg-emerald-400 shadow-md shadow-emerald-500/20"
                >
                  Save Pick
                </button>
              </div>
            </form>

          </div>
        </div>
      )}

    </div>
  );
}