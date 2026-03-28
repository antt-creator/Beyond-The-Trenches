import express from "express";
import { createServer as createViteServer } from "vite";
import { createClient } from "@supabase/supabase-js";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const RAPIDAPI_HOST = process.env.RAPIDAPI_HOST || "api-football-v1.p.rapidapi.com";
const RAPIDAPI_KEY = process.env.RAPIDAPI_KEY;

const FOOTBALL_LEAGUES = [
  { id: 39, name: "England Premier League", key: "EPL" },
  { id: 140, name: "Spain LaLiga", key: "LALIGA" },
  { id: 2, name: "UEFA Champions League", key: "UCL" },
  { id: 1, name: "FIFA World Cup", key: "WORLD_CUP" },
];

// Lazy Supabase initialization
let supabaseClient: any = null;
function getSupabase() {
  if (!supabaseClient) {
    const url = process.env.SUPABASE_URL;
    const key = process.env.SUPABASE_ANON_KEY;
    if (!url || !key) {
      console.warn("Missing SUPABASE_URL or SUPABASE_ANON_KEY. Admin features will not work.");
      return null;
    }
    supabaseClient = createClient(url, key);
  }
  return supabaseClient;
}

type CacheValue = { expiresAt: number; payload: any };
const apiCache = new Map<string, CacheValue>();
let todayDateKey = new Date().toISOString().slice(0, 10);
let todayRequestCount = 0;

function resetDailyBudgetIfNeeded() {
  const nowKey = new Date().toISOString().slice(0, 10);
  if (nowKey !== todayDateKey) {
    todayDateKey = nowKey;
    todayRequestCount = 0;
  }
}

function makeCacheKey(endpoint: string, params: Record<string, string | number>) {
  const sorted = Object.keys(params)
    .sort()
    .map((k) => `${k}=${params[k]}`)
    .join("&");
  return `${endpoint}?${sorted}`;
}

async function footballGet(endpoint: string, params: Record<string, string | number>, ttlSeconds = 600) {
  const cacheKey = makeCacheKey(endpoint, params);
  const cached = apiCache.get(cacheKey);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.payload;
  }

  if (!RAPIDAPI_KEY) {
    throw new Error("RAPIDAPI_KEY is missing. Add it to your .env file.");
  }

  resetDailyBudgetIfNeeded();
  if (todayRequestCount >= 100) {
    throw new Error("Daily API budget reached (100 requests). Try again tomorrow.");
  }

  const query = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => query.set(k, String(v)));
  const url = `https://${RAPIDAPI_HOST}/v3${endpoint}?${query.toString()}`;

  const response = await fetch(url, {
    method: "GET",
    headers: {
      "x-rapidapi-host": RAPIDAPI_HOST,
      "x-rapidapi-key": RAPIDAPI_KEY,
    },
  });

  todayRequestCount += 1;

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`API request failed (${response.status}): ${body}`);
  }

  const payload = await response.json();
  apiCache.set(cacheKey, { expiresAt: Date.now() + ttlSeconds * 1000, payload });
  return payload;
}

function buildTeamStats(fixtures: any[], teamId: number, venue: "all" | "home" | "away" = "all") {
  const recent = fixtures
    .filter((m) => {
      const homeId = m.teams.home.id;
      const awayId = m.teams.away.id;
      if (venue === "home") return homeId === teamId;
      if (venue === "away") return awayId === teamId;
      return homeId === teamId || awayId === teamId;
    })
    .slice(0, 5);

  let wins = 0;
  let draws = 0;
  let losses = 0;
  let goalsFor = 0;
  let goalsAgainst = 0;
  let btts = 0;
  let over25 = 0;
  let cleanSheets = 0;

  for (const fx of recent) {
    const homeId = fx.teams.home.id;
    const isHome = homeId === teamId;
    const hg = Number(fx.goals.home || 0);
    const ag = Number(fx.goals.away || 0);
    const gf = isHome ? hg : ag;
    const ga = isHome ? ag : hg;

    goalsFor += gf;
    goalsAgainst += ga;

    if (gf > ga) wins += 1;
    else if (gf === ga) draws += 1;
    else losses += 1;

    if (ga === 0) cleanSheets += 1;
    if (hg > 0 && ag > 0) btts += 1;
    if (hg + ag > 2) over25 += 1;
  }

  const n = recent.length || 1;
  return {
    formPoints: wins * 3 + draws,
    goalsForAvg: goalsFor / n,
    goalsAgainstAvg: goalsAgainst / n,
    bttsRate: btts / n,
    over25Rate: over25 / n,
    cleanSheetRate: cleanSheets / n,
  };
}

function poisson(lambda: number, k: number): number {
  const l = Math.max(0.05, Math.min(4, lambda));
  let factorial = 1;
  for (let i = 2; i <= k; i += 1) factorial *= i;
  return (Math.exp(-l) * l ** k) / factorial;
}

function deriveOutcomeProbs(expHomeGoals: number, expAwayGoals: number) {
  const maxGoals = 6;
  let home = 0;
  let draw = 0;
  let away = 0;
  let btts = 0;
  let over25 = 0;

  for (let hg = 0; hg <= maxGoals; hg += 1) {
    for (let ag = 0; ag <= maxGoals; ag += 1) {
      const p = poisson(expHomeGoals, hg) * poisson(expAwayGoals, ag);
      if (hg > ag) home += p;
      else if (hg === ag) draw += p;
      else away += p;

      if (hg > 0 && ag > 0) btts += p;
      if (hg + ag > 2) over25 += p;
    }
  }

  const sum = home + draw + away || 1;
  return {
    home: home / sum,
    draw: draw / sum,
    away: away / sum,
    btts,
    over25,
    homeCleanSheet: poisson(expAwayGoals, 0),
    awayCleanSheet: poisson(expHomeGoals, 0),
  };
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "10mb" }));

  // Existing API Routes
  app.get("/api/orders", async (req, res) => {
    const supabase = getSupabase();
    if (!supabase) return res.status(500).json({ error: "Supabase not configured" });

    try {
      const { data, error } = await supabase.from("orders").select("*").order("date", { ascending: false });

      if (error) throw error;
      res.json(data || []);
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: "Failed to fetch orders", details: error });
    }
  });

  app.post("/api/orders", async (req, res) => {
    const supabase = getSupabase();
    if (!supabase) return res.status(500).json({ error: "Supabase not configured" });

    try {
      const order = req.body;
      const { data, error } = await supabase
        .from("orders")
        .insert([
          {
            name: order.name,
            phone: order.phone,
            address: order.address,
            qty: order.qty,
            paymentType: order.paymentType,
            receiptUrl: order.receiptUrl || null,
            notes: order.notes || null,
          },
        ])
        .select();

      if (error) throw error;
      const generatedId = data && data[0] ? data[0].id : "Success";
      res.status(201).json({ success: true, orderId: generatedId });
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: "Failed to create order", details: error });
    }
  });

  // Football routes
  app.get("/api/football/leagues", (_req, res) => {
    res.json({ leagues: FOOTBALL_LEAGUES });
  });

  app.get("/api/football/fixtures", async (req, res) => {
    try {
      const date = String(req.query.date || new Date().toISOString().slice(0, 10));
      const season = Number(req.query.season || new Date().getUTCFullYear());
      const leagueParam = String(req.query.leagues || FOOTBALL_LEAGUES.map((l) => l.id).join(","));
      const leagueIds = leagueParam
        .split(",")
        .map((x) => Number(x.trim()))
        .filter((x) => Number.isFinite(x) && x > 0);

      const allFixtures: any[] = [];
      for (const leagueId of leagueIds) {
        const payload = await footballGet("/fixtures", { league: leagueId, season, date }, 60 * 30);
        const items = payload?.response || [];
        allFixtures.push(...items);
      }

      allFixtures.sort((a, b) => a.fixture.timestamp - b.fixture.timestamp);
      res.json({ fixtures: allFixtures, requestCountToday: todayRequestCount });
    } catch (error: any) {
      res.status(500).json({ error: error?.message || "Failed to fetch fixtures" });
    }
  });

  app.post("/api/football/predict", async (req, res) => {
    try {
      const fixtureId = Number(req.body?.fixtureId);
      if (!fixtureId) return res.status(400).json({ error: "fixtureId is required" });

      const fixturePayload = await footballGet("/fixtures", { id: fixtureId }, 60 * 10);
      const fixture = fixturePayload?.response?.[0];
      if (!fixture) return res.status(404).json({ error: "Fixture not found" });

      const season = Number(fixture.league.season);
      const homeId = Number(fixture.teams.home.id);
      const awayId = Number(fixture.teams.away.id);
      const homeName = String(fixture.teams.home.name);
      const awayName = String(fixture.teams.away.name);

      const [homeRecentPayload, awayRecentPayload, h2hPayload] = await Promise.all([
        footballGet("/fixtures", { team: homeId, season, last: 10 }, 60 * 60),
        footballGet("/fixtures", { team: awayId, season, last: 10 }, 60 * 60),
        footballGet("/fixtures/headtohead", { h2h: `${homeId}-${awayId}`, last: 5 }, 60 * 60),
      ]);

      const homeRecent = homeRecentPayload?.response || [];
      const awayRecent = awayRecentPayload?.response || [];
      const h2hRecent = h2hPayload?.response || [];

      const homeAll = buildTeamStats(homeRecent, homeId, "all");
      const awayAll = buildTeamStats(awayRecent, awayId, "all");
      const homeHome = buildTeamStats(homeRecent, homeId, "home");
      const awayAway = buildTeamStats(awayRecent, awayId, "away");

      const expHomeGoals = Math.max(0.2, Math.min(3.5, (homeHome.goalsForAvg + awayAway.goalsAgainstAvg) / 2));
      const expAwayGoals = Math.max(0.2, Math.min(3.5, (awayAway.goalsForAvg + homeHome.goalsAgainstAvg) / 2));

      const probs = deriveOutcomeProbs(expHomeGoals, expAwayGoals);

      const formDelta = (homeAll.formPoints - awayAll.formPoints) / 15;
      const formAdj = (1 / (1 + Math.exp(-2 * formDelta)) - 0.5) * 0.12;

      let pHome = probs.home + formAdj;
      let pAway = probs.away - formAdj;
      let pDraw = probs.draw;

      // Small H2H adjustment
      let hWins = 0;
      let aWins = 0;
      for (const item of h2hRecent) {
        const hg = Number(item.goals.home || 0);
        const ag = Number(item.goals.away || 0);
        const isCurrentHome = Number(item.teams.home.id) === homeId;
        const homeGoals = isCurrentHome ? hg : ag;
        const awayGoals = isCurrentHome ? ag : hg;
        if (homeGoals > awayGoals) hWins += 1;
        if (awayGoals > homeGoals) aWins += 1;
      }
      const h2hBias = (hWins - aWins) / Math.max(1, h2hRecent.length);
      pHome += 0.04 * h2hBias;
      pAway -= 0.04 * h2hBias;

      const total = pHome + pDraw + pAway || 1;
      pHome /= total;
      pDraw /= total;
      pAway /= total;

      const entries = [
        { label: "Home Win", value: pHome },
        { label: "Draw", value: pDraw },
        { label: "Away Win", value: pAway },
      ].sort((a, b) => b.value - a.value);

      const topPick = entries[0];

      const explanation = `${homeName} form ${homeAll.formPoints}/15 vs ${awayName} ${awayAll.formPoints}/15. ` +
        `${homeName} home GF/GA avg ${homeHome.goalsForAvg.toFixed(2)}/${homeHome.goalsAgainstAvg.toFixed(2)}, ` +
        `${awayName} away GF/GA avg ${awayAway.goalsForAvg.toFixed(2)}/${awayAway.goalsAgainstAvg.toFixed(2)}. ` +
        `H2H last ${h2hRecent.length}: ${hWins} home wins, ${aWins} away wins.`;

      res.json({
        fixtureId,
        prediction: topPick.label,
        confidence: Number((topPick.value * 100).toFixed(1)),
        probabilities: {
          homeWin: Number((pHome * 100).toFixed(2)),
          draw: Number((pDraw * 100).toFixed(2)),
          awayWin: Number((pAway * 100).toFixed(2)),
          btts: Number((probs.btts * 100).toFixed(2)),
          over25: Number((probs.over25 * 100).toFixed(2)),
          homeCleanSheet: Number((probs.homeCleanSheet * 100).toFixed(2)),
          awayCleanSheet: Number((probs.awayCleanSheet * 100).toFixed(2)),
        },
        expectedGoals: {
          home: Number(expHomeGoals.toFixed(2)),
          away: Number(expAwayGoals.toFixed(2)),
        },
        explanation,
      });
    } catch (error: any) {
      res.status(500).json({ error: error?.message || "Prediction failed" });
    }
  });

  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: "spa",
  });

  app.use(vite.middlewares);

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
