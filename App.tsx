import { useEffect, useMemo, useState } from "react";

type League = { id: number; name: string; key: string };

type Fixture = {
  fixture: { id: number; date: string; status: { short: string; long: string } };
  league: { id: number; name: string; round?: string };
  teams: { home: { id: number; name: string }; away: { id: number; name: string } };
};

type PredictionResponse = {
  fixtureId: number;
  prediction: "Home Win" | "Draw" | "Away Win";
  confidence: number;
  probabilities: {
    homeWin: number;
    draw: number;
    awayWin: number;
    btts: number;
    over25: number;
    homeCleanSheet: number;
    awayCleanSheet: number;
  };
  expectedGoals: { home: number; away: number };
  explanation: string;
};

const DEFAULT_LEAGUES = [39, 140, 2, 1];

function todayISODate() {
  return new Date().toISOString().slice(0, 10);
}

export default function App() {
  const [date, setDate] = useState(todayISODate());
  const [season, setSeason] = useState(new Date().getUTCFullYear());
  const [allLeagues, setAllLeagues] = useState<League[]>([]);
  const [selectedLeagueIds, setSelectedLeagueIds] = useState<number[]>(DEFAULT_LEAGUES);
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [loadingFixtures, setLoadingFixtures] = useState(false);
  const [fixturesError, setFixturesError] = useState<string | null>(null);
  const [predictingId, setPredictingId] = useState<number | null>(null);
  const [predictionsByFixture, setPredictionsByFixture] = useState<Record<number, PredictionResponse>>({});
  const [predictionError, setPredictionError] = useState<string | null>(null);

  useEffect(() => {
    const loadLeagues = async () => {
      const res = await fetch("/api/football/leagues");
      const json = await res.json();
      setAllLeagues(json.leagues || []);
    };
    loadLeagues().catch(() => {
      setAllLeagues([
        { id: 39, name: "England Premier League", key: "EPL" },
        { id: 140, name: "Spain LaLiga", key: "LALIGA" },
        { id: 2, name: "UEFA Champions League", key: "UCL" },
        { id: 1, name: "FIFA World Cup", key: "WORLD_CUP" },
      ]);
    });
  }, []);

  const selectedLeagueSet = useMemo(() => new Set(selectedLeagueIds), [selectedLeagueIds]);

  const loadFixtures = async () => {
    setLoadingFixtures(true);
    setFixturesError(null);

    try {
      const params = new URLSearchParams({
        date,
        season: String(season),
        leagues: selectedLeagueIds.join(","),
      });
      const res = await fetch(`/api/football/fixtures?${params.toString()}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Failed to load fixtures");

      setFixtures(json.fixtures || []);
    } catch (err: any) {
      setFixturesError(err.message || "Failed to load fixtures");
      setFixtures([]);
    } finally {
      setLoadingFixtures(false);
    }
  };

  useEffect(() => {
    if (selectedLeagueIds.length) loadFixtures();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [date, season]);

  const toggleLeague = (leagueId: number) => {
    setSelectedLeagueIds((prev) => {
      if (prev.includes(leagueId)) return prev.filter((x) => x !== leagueId);
      return [...prev, leagueId];
    });
  };

  const predictFixture = async (fixtureId: number) => {
    setPredictingId(fixtureId);
    setPredictionError(null);

    try {
      const res = await fetch("/api/football/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fixtureId }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Prediction failed");

      setPredictionsByFixture((prev) => ({ ...prev, [fixtureId]: json }));
    } catch (err: any) {
      setPredictionError(err.message || "Prediction failed");
    } finally {
      setPredictingId(null);
    }
  };

  return (
    <div className="min-h-screen bg-warm-bg text-ink">
      <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
        <header className="space-y-2">
          <h1 className="text-3xl md:text-4xl font-bold serif">Football Match Predictor</h1>
          <p className="text-black/70">
            EPL, LaLiga, UCL, World Cup ပွဲစဉ်တွေကို ပြပြီး ပွဲတစ်ပွဲချင်းစီ <b>Predict</b> ခလုတ်နှိပ်မှ ခန့်မှန်းပေးပါမယ်။
          </p>
        </header>

        <section className="bg-white rounded-2xl border border-black/10 p-4 md:p-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
            <label className="space-y-1">
              <span className="text-xs uppercase font-semibold text-black/50">Date (UTC)</span>
              <input className="input-field" type="date" value={date} onChange={(e) => setDate(e.target.value)} />
            </label>
            <label className="space-y-1">
              <span className="text-xs uppercase font-semibold text-black/50">Season</span>
              <input
                className="input-field"
                type="number"
                value={season}
                onChange={(e) => setSeason(Number(e.target.value || season))}
              />
            </label>
            <div className="md:col-span-2 flex justify-start md:justify-end">
              <button className="btn-primary" onClick={loadFixtures} disabled={loadingFixtures || selectedLeagueIds.length === 0}>
                {loadingFixtures ? "Loading fixtures..." : "Reload Fixtures"}
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-xs uppercase font-semibold text-black/50">Leagues to show</p>
            <div className="flex flex-wrap gap-2">
              {allLeagues.map((league) => (
                <button
                  key={league.id}
                  onClick={() => toggleLeague(league.id)}
                  className={`px-3 py-2 rounded-full border text-sm transition ${
                    selectedLeagueSet.has(league.id)
                      ? "bg-olive text-white border-olive"
                      : "bg-white border-black/15 text-black/70"
                  }`}
                >
                  {league.name}
                </button>
              ))}
            </div>
          </div>
        </section>

        {fixturesError && <div className="p-3 rounded-xl bg-red-100 text-red-700 border border-red-300">{fixturesError}</div>}
        {predictionError && (
          <div className="p-3 rounded-xl bg-red-100 text-red-700 border border-red-300">{predictionError}</div>
        )}

        <section className="bg-white rounded-2xl border border-black/10 overflow-hidden">
          <div className="px-4 py-3 border-b border-black/10 flex items-center justify-between">
            <h2 className="font-semibold">Matches ({fixtures.length})</h2>
            <span className="text-sm text-black/50">Click Predict per match</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-black/[0.03]">
                <tr>
                  <th className="text-left px-3 py-2">Time</th>
                  <th className="text-left px-3 py-2">League</th>
                  <th className="text-left px-3 py-2">Match</th>
                  <th className="text-left px-3 py-2">Action</th>
                  <th className="text-left px-3 py-2">Prediction</th>
                </tr>
              </thead>
              <tbody>
                {fixtures.map((fx) => {
                  const kickoff = new Date(fx.fixture.date).toISOString().slice(11, 16);
                  const prediction = predictionsByFixture[fx.fixture.id];
                  return (
                    <tr key={fx.fixture.id} className="border-t border-black/5 align-top">
                      <td className="px-3 py-3 whitespace-nowrap">{kickoff} UTC</td>
                      <td className="px-3 py-3">{fx.league.name}</td>
                      <td className="px-3 py-3">
                        <div className="font-medium">
                          {fx.teams.home.name} vs {fx.teams.away.name}
                        </div>
                        <div className="text-black/50 text-xs">{fx.league.round || fx.fixture.status.long}</div>
                      </td>
                      <td className="px-3 py-3">
                        <button
                          className="btn-primary px-4 py-2 rounded-lg"
                          onClick={() => predictFixture(fx.fixture.id)}
                          disabled={predictingId === fx.fixture.id}
                        >
                          {predictingId === fx.fixture.id ? "Predicting..." : "Predict"}
                        </button>
                      </td>
                      <td className="px-3 py-3 min-w-[360px]">
                        {!prediction ? (
                          <span className="text-black/40">No prediction yet</span>
                        ) : (
                          <div className="space-y-1">
                            <div className="font-semibold text-olive">
                              {prediction.prediction} ({prediction.confidence.toFixed(1)}%)
                            </div>
                            <div className="text-xs text-black/70">
                              1X2: H {prediction.probabilities.homeWin}% | D {prediction.probabilities.draw}% | A {" "}
                              {prediction.probabilities.awayWin}%
                            </div>
                            <div className="text-xs text-black/70">
                              BTTS {prediction.probabilities.btts}% | O2.5 {prediction.probabilities.over25}% | CS(H){" "}
                              {prediction.probabilities.homeCleanSheet}% | CS(A) {prediction.probabilities.awayCleanSheet}%
                            </div>
                            <div className="text-xs text-black/70">
                              xG: {prediction.expectedGoals.home} - {prediction.expectedGoals.away}
                            </div>
                            <p className="text-xs text-black/60">{prediction.explanation}</p>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
                {!fixtures.length && !loadingFixtures && (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-black/40">
                      No fixtures found for selected filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
