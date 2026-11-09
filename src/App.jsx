import React, { useState, useEffect } from 'react';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-900 text white p-8">
      <h1 className="text-3xl font-bold">NCAA Longshot Picks Vault</h1>
      <p className="mt-2 text-slate-400">System online and rendering correctly!</p>
    </div>
  );
}
  const [picks, setPicks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetches the live generated file from the public folder
    fetch('/picks.json')
      .then((res) => res.json())
      .then((data) => {
        setPicks(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Could not load picks.json, ensure you ran 'npm run update-picks'", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="p-8 text-center text-white">Loading daily longshot board...</div>;
  }

  return (
    <div>
      {/* Your existing UI rendering components using the 'picks' state array */}
    </div>
  );
}