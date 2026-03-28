#!/usr/bin/env python3
"""
Personal Football Match Prediction System
=========================================

This script fetches fixtures and team data from API-Football (RapidAPI),
computes lightweight statistical features, and generates practical predictions
for personal use.

Setup
-----
1) Create a RapidAPI account and subscribe to API-Football (free tier):
   https://rapidapi.com/api-sports/api/api-football/
2) Copy `.env.example` to `.env` and add your API key.
3) Copy `config.example.json` to `config.json` and adjust preferences.
4) Install dependencies: `pip install -r requirements.txt`
5) Run manually: `python football_predictor.py --date YYYY-MM-DD`
   or daily scheduler: `python football_predictor.py --schedule`

Notes
-----
- Free tier is limited (commonly 100 requests/day). This script includes:
  * local SQLite cache
  * per-request expiry logic
  * daily request budget tracking
- Predictions are heuristic, not guaranteed. Use responsibly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import schedule
from dotenv import load_dotenv
from tabulate import tabulate

# ----------------------------- Configuration Models -----------------------------


@dataclass
class TeamRecentStats:
    """Container for team-level rolling statistics."""

    team_id: int
    team_name: str
    form_points_5: int
    goals_for_avg: float
    goals_against_avg: float
    clean_sheet_rate: float
    btts_rate: float
    over25_rate: float
    wins: int
    draws: int
    losses: int


@dataclass
class MatchPrediction:
    """Container for output prediction row."""

    fixture_id: int
    league: str
    kickoff_utc: str
    home_team: str
    away_team: str
    pred_outcome: str
    confidence: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    btts_prob: float
    over25_prob: float
    home_clean_sheet_prob: float
    away_clean_sheet_prob: float
    expected_home_goals: float
    expected_away_goals: float
    expected_value_note: str
    explanation: str


# --------------------------------- Utilities -----------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# ------------------------------ Database / Storage ------------------------------


class Storage:
    """SQLite storage for cache, predictions, and accuracy tracking."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                params_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS request_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_date TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                fixture_id INTEGER PRIMARY KEY,
                prediction_date TEXT NOT NULL,
                kickoff_utc TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                confidence REAL NOT NULL,
                home_win_prob REAL NOT NULL,
                draw_prob REAL NOT NULL,
                away_win_prob REAL NOT NULL,
                explanation TEXT NOT NULL,
                actual_outcome TEXT,
                result_checked_at TEXT
            );
            """
        )
        self.conn.commit()

    def _cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        payload = f"{endpoint}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_cached(self, endpoint: str, params: Dict[str, Any]) -> Optional[dict]:
        key = self._cache_key(endpoint, params)
        cur = self.conn.cursor()
        cur.execute(
            "SELECT response_json, expires_at FROM api_cache WHERE cache_key = ?",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None

        if parse_iso_datetime(row["expires_at"]) < utc_now():
            return None

        return json.loads(row["response_json"])

    def set_cached(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response: dict,
        ttl_minutes: int,
    ) -> None:
        key = self._cache_key(endpoint, params)
        created_at = utc_now()
        expires_at = created_at + timedelta(minutes=ttl_minutes)
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO api_cache
            (cache_key, endpoint, params_json, response_json, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                endpoint,
                json.dumps(params, sort_keys=True),
                json.dumps(response),
                created_at.isoformat(),
                expires_at.isoformat(),
            ),
        )
        self.conn.commit()

    def log_request(self, endpoint: str) -> None:
        now = utc_now()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO request_log (request_date, endpoint, created_at) VALUES (?, ?, ?)",
            (now.date().isoformat(), endpoint, now.isoformat()),
        )
        self.conn.commit()

    def requests_today(self) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS c FROM request_log WHERE request_date = ?",
            (utc_now().date().isoformat(),),
        )
        return int(cur.fetchone()["c"])

    def save_prediction(self, p: MatchPrediction) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO predictions
            (fixture_id, prediction_date, kickoff_utc, home_team, away_team,
             predicted_outcome, confidence, home_win_prob, draw_prob, away_win_prob,
             explanation, actual_outcome, result_checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT actual_outcome FROM predictions WHERE fixture_id = ?), NULL),
                    COALESCE((SELECT result_checked_at FROM predictions WHERE fixture_id = ?), NULL))
            """,
            (
                p.fixture_id,
                utc_now().date().isoformat(),
                p.kickoff_utc,
                p.home_team,
                p.away_team,
                p.pred_outcome,
                p.confidence,
                p.home_win_prob,
                p.draw_prob,
                p.away_win_prob,
                p.explanation,
                p.fixture_id,
                p.fixture_id,
            ),
        )
        self.conn.commit()

    def update_actual_outcome(self, fixture_id: int, outcome: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE predictions
            SET actual_outcome = ?, result_checked_at = ?
            WHERE fixture_id = ?
            """,
            (outcome, utc_now().isoformat(), fixture_id),
        )
        self.conn.commit()

    def accuracy_summary(self) -> Dict[str, float]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN predicted_outcome = actual_outcome THEN 1 ELSE 0 END) AS correct
            FROM predictions
            WHERE actual_outcome IS NOT NULL
            """
        )
        row = cur.fetchone()
        total = int(row["total"] or 0)
        correct = int(row["correct"] or 0)
        acc = (100.0 * correct / total) if total else 0.0
        return {"resolved_predictions": total, "correct": correct, "accuracy_pct": acc}


# -------------------------------- API Client -----------------------------------


class APIFootballClient:
    """API client with caching + request budget guardrails."""

    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

    def __init__(
        self,
        api_key: str,
        storage: Storage,
        host: str,
        daily_limit: int = 100,
        timeout_seconds: int = 25,
    ):
        self.api_key = api_key
        self.host = host
        self.storage = storage
        self.daily_limit = daily_limit
        self.timeout_seconds = timeout_seconds

    def _check_budget(self, endpoint: str) -> None:
        used = self.storage.requests_today()
        if used >= self.daily_limit:
            raise RuntimeError(
                f"Daily API request limit reached ({used}/{self.daily_limit}). "
                f"Skipping endpoint: {endpoint}."
            )

    def get(
        self,
        endpoint: str,
        params: Dict[str, Any],
        ttl_minutes: int,
        allow_stale_on_error: bool = True,
    ) -> dict:
        cached = self.storage.get_cached(endpoint, params)
        if cached is not None:
            return cached

        self._check_budget(endpoint)

        url = f"{self.BASE_URL}{endpoint}"
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host,
        }

        for attempt in range(3):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429:
                    sleep_for = min(2 ** attempt, 8)
                    logging.warning("Rate limited (429). Retrying in %ss...", sleep_for)
                    time.sleep(sleep_for)
                    continue

                response.raise_for_status()
                data = response.json()
                self.storage.log_request(endpoint)
                self.storage.set_cached(endpoint, params, data, ttl_minutes)
                return data

            except requests.RequestException as err:
                logging.warning("API request failed for %s: %s", endpoint, err)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                if allow_stale_on_error and cached is not None:
                    logging.warning("Using stale cache due to API errors.")
                    return cached

                raise

        raise RuntimeError(f"Failed API request after retries: {endpoint}")


# ----------------------------- Prediction Functions -----------------------------


def points_for_result(result: str) -> int:
    return {"W": 3, "D": 1, "L": 0}.get(result, 0)


def outcome_from_scores(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "Home Win"
    if home_goals < away_goals:
        return "Away Win"
    return "Draw"


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def compute_recent_stats(
    fixtures: List[dict],
    team_id: int,
    team_name: str,
    venue_filter: Optional[str] = None,
) -> TeamRecentStats:
    """
    Compute rolling metrics from a fixture list.

    venue_filter:
      None -> all matches
      "home" -> only matches where team_id was home team
      "away" -> only matches where team_id was away team
    """
    filtered: List[dict] = []
    for fx in fixtures:
        home_id = fx["teams"]["home"]["id"]
        away_id = fx["teams"]["away"]["id"]
        if venue_filter == "home" and home_id != team_id:
            continue
        if venue_filter == "away" and away_id != team_id:
            continue
        if home_id != team_id and away_id != team_id:
            continue
        filtered.append(fx)

    filtered = filtered[:5]
    wins = draws = losses = 0
    goals_for = goals_against = 0
    clean_sheets = btts = over25 = 0

    for fx in filtered:
        home_id = fx["teams"]["home"]["id"]
        hg = int(fx["goals"]["home"] or 0)
        ag = int(fx["goals"]["away"] or 0)
        is_home = home_id == team_id

        gf = hg if is_home else ag
        ga = ag if is_home else hg

        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

        if ga == 0:
            clean_sheets += 1
        if hg > 0 and ag > 0:
            btts += 1
        if hg + ag > 2:
            over25 += 1

    matches = len(filtered)
    form_points = wins * 3 + draws
    return TeamRecentStats(
        team_id=team_id,
        team_name=team_name,
        form_points_5=form_points,
        goals_for_avg=safe_div(goals_for, matches),
        goals_against_avg=safe_div(goals_against, matches),
        clean_sheet_rate=safe_div(clean_sheets, matches),
        btts_rate=safe_div(btts, matches),
        over25_rate=safe_div(over25, matches),
        wins=wins,
        draws=draws,
        losses=losses,
    )


def poisson_goal_probs(lmbda: float, max_goals: int = 6) -> List[float]:
    probs = []
    for k in range(max_goals + 1):
        probs.append(math.exp(-lmbda) * (lmbda**k) / math.factorial(k))
    return probs


def derive_match_probabilities(
    exp_home_goals: float,
    exp_away_goals: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Uses truncated Poisson matrix to estimate:
      - Home/Draw/Away probabilities
      - BTTS probability
      - Over 2.5 probability
      - Home clean-sheet probability
      - Away clean-sheet probability
    """
    home_probs = poisson_goal_probs(clamp(exp_home_goals, 0.05, 4.0))
    away_probs = poisson_goal_probs(clamp(exp_away_goals, 0.05, 4.0))

    p_home = p_draw = p_away = 0.0
    p_btts = p_over25 = 0.0

    for hg, ph in enumerate(home_probs):
        for ag, pa in enumerate(away_probs):
            joint = ph * pa
            if hg > ag:
                p_home += joint
            elif hg == ag:
                p_draw += joint
            else:
                p_away += joint

            if hg > 0 and ag > 0:
                p_btts += joint
            if hg + ag > 2:
                p_over25 += joint

    p_home_cs = away_probs[0]
    p_away_cs = home_probs[0]
    return p_home, p_draw, p_away, p_btts, p_over25, p_home_cs, p_away_cs


def normalize_probs(*probs: float) -> Tuple[float, ...]:
    s = sum(probs)
    if s <= 0:
        return tuple(1 / len(probs) for _ in probs)
    return tuple(p / s for p in probs)


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        return 0.0
    return 1 / decimal_odds


def expected_value(prob: float, decimal_odds: float, stake: float = 1.0) -> float:
    """Simple betting EV = p*(odds-1)*stake - (1-p)*stake."""
    return prob * (decimal_odds - 1) * stake - (1 - prob) * stake


# ------------------------------ Core Orchestration ------------------------------


class PredictorEngine:
    def __init__(self, cfg: Dict[str, Any], api: APIFootballClient, storage: Storage):
        self.cfg = cfg
        self.api = api
        self.storage = storage

    def fetch_todays_fixtures(self, target_date: str) -> List[dict]:
        fixtures: List[dict] = []
        season = self.cfg["season"]

        for league_id in self.cfg["leagues"]:
            params = {"league": league_id, "season": season, "date": target_date}
            data = self.api.get("/fixtures", params=params, ttl_minutes=60)
            fixtures.extend(data.get("response", []))

        fixtures.sort(key=lambda x: x["fixture"]["date"])
        return fixtures

    def fetch_team_recent_fixtures(self, team_id: int, season: int, last_n: int = 10) -> List[dict]:
        params = {"team": team_id, "season": season, "last": last_n}
        data = self.api.get("/fixtures", params=params, ttl_minutes=180)
        response = data.get("response", [])
        # Already descending by date in API, but enforce for safety.
        response.sort(key=lambda x: x["fixture"]["timestamp"], reverse=True)
        return response

    def fetch_h2h(self, home_team_id: int, away_team_id: int) -> List[dict]:
        h2h_key = f"{home_team_id}-{away_team_id}"
        params = {"h2h": h2h_key, "last": int(self.cfg["h2h_last_n"]) }
        data = self.api.get("/fixtures/headtohead", params=params, ttl_minutes=360)
        return data.get("response", [])

    def fetch_odds_for_fixture(self, fixture_id: int) -> Optional[dict]:
        if not self.cfg.get("use_odds", True):
            return None

        try:
            data = self.api.get("/odds", params={"fixture": fixture_id}, ttl_minutes=120)
            entries = data.get("response", [])
            return entries[0] if entries else None
        except Exception as err:
            logging.info("Odds unavailable for fixture %s: %s", fixture_id, err)
            return None

    def update_past_results(self) -> None:
        """Resolve stored predictions that now have final scores."""
        cur = self.storage.conn.cursor()
        cur.execute(
            """
            SELECT fixture_id
            FROM predictions
            WHERE actual_outcome IS NULL
              AND date(kickoff_utc) <= date('now')
            """
        )
        fixture_ids = [int(r["fixture_id"]) for r in cur.fetchall()]

        for fixture_id in fixture_ids:
            try:
                data = self.api.get("/fixtures", params={"id": fixture_id}, ttl_minutes=30)
            except Exception:
                continue

            response = data.get("response", [])
            if not response:
                continue

            fx = response[0]
            status = fx["fixture"]["status"]["short"]
            if status not in {"FT", "AET", "PEN"}:
                continue

            hg = fx["goals"]["home"]
            ag = fx["goals"]["away"]
            if hg is None or ag is None:
                continue

            outcome = outcome_from_scores(int(hg), int(ag))
            self.storage.update_actual_outcome(fixture_id, outcome)

    def _extract_odds_1x2(self, odds_payload: Optional[dict]) -> Dict[str, float]:
        """
        Tries to extract decimal odds from API-Football odds payload.
        Fallback: empty mapping if structure unavailable.
        """
        out: Dict[str, float] = {}
        if not odds_payload:
            return out

        try:
            bookmakers = odds_payload.get("bookmakers", [])
            if not bookmakers:
                return out

            bets = bookmakers[0].get("bets", [])
            # Commonly, "Match Winner" is bet id=1 in many feeds.
            for bet in bets:
                label = str(bet.get("name", "")).lower()
                if "match winner" not in label and str(bet.get("id")) != "1":
                    continue

                for val in bet.get("values", []):
                    v = val.get("value", "")
                    odd = float(val.get("odd"))
                    if v == "Home":
                        out["Home Win"] = odd
                    elif v == "Draw":
                        out["Draw"] = odd
                    elif v == "Away":
                        out["Away Win"] = odd
                break
        except Exception:
            return {}

        return out

    def _league_weight(self, league_id: int) -> float:
        """Simple league-level calibration hook from config."""
        overrides = self.cfg.get("league_strength_adjustment", {})
        return float(overrides.get(str(league_id), 1.0))

    def build_prediction_for_fixture(self, fx: dict) -> MatchPrediction:
        fixture_id = int(fx["fixture"]["id"])
        league_name = fx["league"]["name"]
        league_id = int(fx["league"]["id"])
        kickoff_utc = fx["fixture"]["date"]
        season = int(fx["league"]["season"])

        home = fx["teams"]["home"]
        away = fx["teams"]["away"]

        home_id, away_id = int(home["id"]), int(away["id"])
        home_name, away_name = str(home["name"]), str(away["name"])

        home_recent = self.fetch_team_recent_fixtures(home_id, season, last_n=10)
        away_recent = self.fetch_team_recent_fixtures(away_id, season, last_n=10)
        h2h_recent = self.fetch_h2h(home_id, away_id)

        home_all = compute_recent_stats(home_recent, home_id, home_name, venue_filter=None)
        away_all = compute_recent_stats(away_recent, away_id, away_name, venue_filter=None)
        home_home = compute_recent_stats(home_recent, home_id, home_name, venue_filter="home")
        away_away = compute_recent_stats(away_recent, away_id, away_name, venue_filter="away")

        # xG fallback strategy:
        # API-Football free tier generally doesn't provide robust xG in fixtures feed,
        # so we derive expected goals from attack vs defense strength.
        # Blend home advantage (home_home) with overall stability (home_all/away_all).
        league_scale = self._league_weight(league_id)
        home_attack = 0.65 * home_home.goals_for_avg + 0.35 * home_all.goals_for_avg
        away_attack = 0.65 * away_away.goals_for_avg + 0.35 * away_all.goals_for_avg
        home_def = 0.65 * home_home.goals_against_avg + 0.35 * home_all.goals_against_avg
        away_def = 0.65 * away_away.goals_against_avg + 0.35 * away_all.goals_against_avg

        exp_home_goals = clamp((home_attack + away_def) / 2 * league_scale, 0.2, 3.5)
        exp_away_goals = clamp((away_attack + home_def) / 2 * league_scale, 0.2, 3.5)

        # H2H influence as small adjustment (avoid overfitting old rivalry data).
        h2h_home_wins = h2h_draws = h2h_away_wins = 0
        for item in h2h_recent:
            hg = int(item["goals"]["home"] or 0)
            ag = int(item["goals"]["away"] or 0)
            h_id = int(item["teams"]["home"]["id"])
            # Remap to current home/away perspective
            if h_id == home_id:
                res = outcome_from_scores(hg, ag)
            else:
                res = outcome_from_scores(ag, hg)
            if res == "Home Win":
                h2h_home_wins += 1
            elif res == "Draw":
                h2h_draws += 1
            else:
                h2h_away_wins += 1

        p_home, p_draw, p_away, p_btts, p_over25, p_home_cs, p_away_cs = derive_match_probabilities(
            exp_home_goals, exp_away_goals
        )

        # Add form signal using logistic adjustment.
        form_delta = (home_all.form_points_5 - away_all.form_points_5) / 15.0
        form_boost = sigmoid(2.2 * form_delta) - 0.5
        p_home += 0.12 * form_boost
        p_away -= 0.12 * form_boost

        # H2H micro-adjustment.
        h2h_total = max(1, h2h_home_wins + h2h_draws + h2h_away_wins)
        h2h_bias = (h2h_home_wins - h2h_away_wins) / h2h_total
        p_home += 0.04 * h2h_bias
        p_away -= 0.04 * h2h_bias

        p_home, p_draw, p_away = normalize_probs(p_home, p_draw, p_away)

        probs = {
            "Home Win": p_home,
            "Draw": p_draw,
            "Away Win": p_away,
        }
        pred_outcome = max(probs, key=probs.get)
        confidence = round(probs[pred_outcome] * 100, 1)

        # Betting EV note from available odds.
        odds_payload = self.fetch_odds_for_fixture(fixture_id)
        odds_1x2 = self._extract_odds_1x2(odds_payload)
        ev_note = "Odds unavailable"
        if odds_1x2:
            best_market = None
            best_ev = -999.0
            for k, odd in odds_1x2.items():
                ev = expected_value(probs[k], odd)
                if ev > best_ev:
                    best_ev = ev
                    best_market = (k, odd, ev)
            if best_market:
                label, odd, ev = best_market
                ev_note = f"Best EV: {label} @ {odd:.2f} (EV={ev:+.3f})"

        explanation = (
            f"{home_name} form {home_all.form_points_5}/15 vs {away_name} {away_all.form_points_5}/15; "
            f"home avg GF/GA {home_home.goals_for_avg:.2f}/{home_home.goals_against_avg:.2f}, "
            f"away avg GF/GA {away_away.goals_for_avg:.2f}/{away_away.goals_against_avg:.2f}; "
            f"H2H(last {len(h2h_recent)}): {h2h_home_wins}-{h2h_draws}-{h2h_away_wins}."
        )

        return MatchPrediction(
            fixture_id=fixture_id,
            league=league_name,
            kickoff_utc=kickoff_utc,
            home_team=home_name,
            away_team=away_name,
            pred_outcome=pred_outcome,
            confidence=confidence,
            home_win_prob=round(p_home * 100, 2),
            draw_prob=round(p_draw * 100, 2),
            away_win_prob=round(p_away * 100, 2),
            btts_prob=round(p_btts * 100, 2),
            over25_prob=round(p_over25 * 100, 2),
            home_clean_sheet_prob=round(p_home_cs * 100, 2),
            away_clean_sheet_prob=round(p_away_cs * 100, 2),
            expected_home_goals=round(exp_home_goals, 2),
            expected_away_goals=round(exp_away_goals, 2),
            expected_value_note=ev_note,
            explanation=explanation,
        )


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Copy config.example.json to config.json first."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_predictions_csv(predictions: Iterable[MatchPrediction], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"predictions_{ts}.csv"

    fields = [
        "fixture_id",
        "league",
        "kickoff_utc",
        "home_team",
        "away_team",
        "pred_outcome",
        "confidence",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "btts_prob",
        "over25_prob",
        "home_clean_sheet_prob",
        "away_clean_sheet_prob",
        "expected_home_goals",
        "expected_away_goals",
        "expected_value_note",
        "explanation",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in predictions:
            writer.writerow(p.__dict__)

    return out_path


def render_terminal_table(predictions: List[MatchPrediction]) -> str:
    rows = []
    for p in predictions:
        rows.append(
            [
                p.kickoff_utc[11:16],
                p.league,
                f"{p.home_team} vs {p.away_team}",
                p.pred_outcome,
                f"{p.confidence:.1f}%",
                f"H:{p.home_win_prob:.1f} D:{p.draw_prob:.1f} A:{p.away_win_prob:.1f}",
                f"BTTS {p.btts_prob:.1f}% | O2.5 {p.over25_prob:.1f}%",
                p.expected_value_note,
            ]
        )

    return tabulate(
        rows,
        headers=["Kickoff", "League", "Match", "Prediction", "Confidence", "1X2 Prob", "Goals", "EV"],
        tablefmt="github",
    )


def run_once(cfg: Dict[str, Any], target_date: str) -> None:
    load_dotenv()

    api_key = os.getenv("RAPIDAPI_KEY", "").strip()
    host = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com").strip()

    if not api_key:
        raise RuntimeError(
            "Missing RAPIDAPI_KEY. Add it to your .env file (see .env.example)."
        )

    data_dir = Path(cfg.get("data_dir", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(data_dir / "football_predictor.db")
    api = APIFootballClient(
        api_key=api_key,
        host=host,
        storage=storage,
        daily_limit=int(cfg.get("api_daily_limit", 100)),
        timeout_seconds=int(cfg.get("api_timeout_seconds", 25)),
    )
    engine = PredictorEngine(cfg=cfg, api=api, storage=storage)

    engine.update_past_results()
    fixtures = engine.fetch_todays_fixtures(target_date)

    if not fixtures:
        print(f"No fixtures found for {target_date} in configured leagues.")
        acc = storage.accuracy_summary()
        print(f"Accuracy so far: {acc['correct']}/{acc['resolved_predictions']} = {acc['accuracy_pct']:.2f}%")
        return

    predictions: List[MatchPrediction] = []
    for fx in fixtures:
        try:
            p = engine.build_prediction_for_fixture(fx)
            predictions.append(p)
            storage.save_prediction(p)
        except Exception as err:
            logging.exception(
                "Failed to predict fixture %s: %s", fx.get("fixture", {}).get("id"), err
            )

    if not predictions:
        print("No predictions generated (errors encountered).")
        return

    print(f"\nFootball predictions for {target_date} (UTC):")
    print(render_terminal_table(predictions))

    csv_path = export_predictions_csv(predictions, data_dir / "exports")
    print(f"\nCSV export: {csv_path}")

    acc = storage.accuracy_summary()
    print(
        f"Historical accuracy: {acc['correct']}/{acc['resolved_predictions']} "
        f"= {acc['accuracy_pct']:.2f}%"
    )
    print(
        f"API usage today: {storage.requests_today()} / {cfg.get('api_daily_limit', 100)} "
        f"(cached calls are not counted)."
    )


def run_scheduler(cfg: Dict[str, Any], target_date: Optional[str]) -> None:
    schedule_time = cfg.get("schedule_time_utc", "09:00")

    def _job() -> None:
        date_str = target_date or utc_now().date().isoformat()
        logging.info("Running scheduled job for date %s", date_str)
        run_once(cfg, date_str)

    schedule.every().day.at(schedule_time).do(_job)
    logging.info("Scheduler active. Daily run at %s UTC. Press Ctrl+C to stop.", schedule_time)

    # Optionally run immediately on startup.
    if cfg.get("run_immediately_on_schedule_start", True):
        _job()

    while True:
        schedule.run_pending()
        time.sleep(20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personal football prediction engine")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config file (default: config.json)",
    )
    parser.add_argument(
        "--date",
        default=utc_now().date().isoformat(),
        help="Target fixture date in UTC (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run daily scheduler instead of single run",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        cfg = load_config(Path(args.config))
    except Exception as err:
        logging.error("Failed to load config: %s", err)
        return 1

    try:
        if args.schedule:
            run_scheduler(cfg, target_date=args.date)
        else:
            run_once(cfg, target_date=args.date)
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
        return 0
    except Exception as err:
        logging.error("Fatal error: %s", err)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
