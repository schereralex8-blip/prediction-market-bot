#!/usr/bin/env python3
"""Generate the synthetic slate that ships with the repo.

Everything under ``data/`` is fabricated by this script from a fixed seed --
no real box scores, no real prices. The point is that ``pmbot scan`` works
end to end with no API key, and that the test suite has a slate with known
properties: a few genuinely mispriced props, and a lot of efficient ones that
the bot should refuse to bet.

The prices are built from the *true* generating process (what an omniscient
book would post), plus vig, plus per-book noise. Designated "value" props get
one book left stale by a few points of probability. So the bot only finds the
edge if its models actually recover the truth -- the edge is not planted
where the model is guaranteed to look.
"""

from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 4242
N_GAMES = 46
BOOKS = ("pinnacle", "draftkings", "fanduel", "betmgm", "caesars")

rng = random.Random(SEED)


# --------------------------------------------------------------------------
# small samplers
# --------------------------------------------------------------------------
def poisson(lam: float) -> int:
    if lam <= 0:
        return 0
    if lam < 30:
        target, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= rng.random()
            if p <= target:
                return k
            k += 1
    return max(0, round(rng.gauss(lam, math.sqrt(lam))))


def negbin(mean: float, dispersion: float = 2.2) -> int:
    """Poisson mixed over a Gamma rate -- counts with realistic fat tails."""
    if mean <= 0:
        return 0
    shape = mean / max(dispersion - 1.0, 1e-6)
    return poisson(rng.gammavariate(shape, max(dispersion - 1.0, 1e-6)))


def clipped_normal(mu: float, sd: float, lo: float = 0.0, hi: float = 1e9) -> float:
    return min(max(rng.gauss(mu, sd), lo), hi)


def binomial(n: int, p: float) -> int:
    return sum(1 for _ in range(max(n, 0)) if rng.random() < p)


# --------------------------------------------------------------------------
# per-sport box score generators
# --------------------------------------------------------------------------
def nba_game(p: dict) -> dict:
    minutes = round(clipped_normal(p["minutes"], p["minutes_sd"], 12, 44), 1)
    pts = negbin(p["pts_per_min"] * minutes, 2.4)
    return {
        "minutes": minutes,
        "points": pts,
        "rebounds": negbin(p["reb_per_min"] * minutes, 1.7),
        "assists": negbin(p["ast_per_min"] * minutes, 1.8),
        "threes": negbin(p["threes_per_min"] * minutes, 1.5),
        "blocks": poisson(p["blk_per_min"] * minutes),
        "steals": poisson(p["stl_per_min"] * minutes),
        "turnovers": negbin(p["tov_per_min"] * minutes, 1.4),
    }


def mlb_batter_game(p: dict) -> dict:
    pa = max(1, round(clipped_normal(p["pa"], 0.75, 1, 6)))
    hits = binomial(pa, p["hit_rate"])
    hrs = binomial(pa, p["hr_rate"])
    extra = binomial(max(hits - hrs, 0), 0.30)  # doubles/triples among non-HR hits
    return {
        "plate_appearances": pa,
        "hits": hits,
        "home_runs": hrs,
        "total_bases": (hits - hrs - extra) + 2 * extra + 4 * hrs,
        "rbis": poisson(p["rbi_rate"] * pa),
        "runs": poisson(p["run_rate"] * pa),
        "stolen_bases": poisson(p["sb_rate"] * pa),
    }


def mlb_pitcher_game(p: dict) -> dict:
    bf = max(6, round(clipped_normal(p["batters_faced"], 3.4, 8, 34)))
    ks = binomial(bf, p["k_rate"])
    hits = binomial(bf, p["hit_rate"])
    return {
        "batters_faced": bf,
        "strikeouts": ks,
        "hits_allowed": hits,
        "outs": max(3, min(27, round(clipped_normal(bf * 0.685, 2.0, 3, 27)))),
        "earned_runs": poisson(p["er_rate"] * bf),
    }


def nfl_qb_game(p: dict) -> dict:
    att = max(12, round(clipped_normal(p["attempts"], 5.2, 12, 55)))
    comps = binomial(att, p["comp_rate"])
    return {
        "pass_attempts": att,
        "completions": comps,
        "pass_yards": max(0, round(rng.gauss(p["ypa"] * att, 2.1 * math.sqrt(att)))),
        "pass_tds": poisson(p["td_rate"] * att),
        "rush_attempts": max(0, round(clipped_normal(3.5, 2.0, 0, 12))),
        "rush_yards": max(-5, round(rng.gauss(14, 12))),
    }


def nfl_receiver_game(p: dict) -> dict:
    targets = max(0, round(clipped_normal(p["targets"], p["targets_sd"], 0, 20)))
    recs = binomial(targets, p["catch_rate"])
    yards = 0 if recs == 0 else max(0, round(rng.gauss(p["ypr"] * recs, 6.5 * math.sqrt(max(recs, 1)))))
    return {
        "targets": targets,
        "receptions": recs,
        "receiving_yards": yards,
        "rush_attempts": 0,
        "rush_yards": 0,
    }


def nfl_rb_game(p: dict) -> dict:
    carries = max(2, round(clipped_normal(p["carries"], 4.0, 2, 30)))
    targets = max(0, round(clipped_normal(p["targets"], 1.6, 0, 10)))
    recs = binomial(targets, 0.76)
    return {
        "rush_attempts": carries,
        "rush_yards": max(-5, round(rng.gauss(p["ypc"] * carries, 3.3 * math.sqrt(carries)))),
        "targets": targets,
        "receptions": recs,
        "receiving_yards": 0 if recs == 0 else max(0, round(rng.gauss(7.6 * recs, 6.0 * math.sqrt(recs)))),
    }


GENERATORS = {
    "nba": nba_game,
    "mlb_batter": mlb_batter_game,
    "mlb_pitcher": mlb_pitcher_game,
    "nfl_qb": nfl_qb_game,
    "nfl_wr": nfl_receiver_game,
    "nfl_rb": nfl_rb_game,
}


# --------------------------------------------------------------------------
# the roster
# --------------------------------------------------------------------------
PLAYERS: list[dict] = [
    # ---- NBA
    dict(sport="nba", kind="nba", name="Anthony Edwards", team="MIN", opponent="SAC",
         minutes=35.4, minutes_sd=3.1, pts_per_min=0.78, reb_per_min=0.16, ast_per_min=0.14,
         threes_per_min=0.085, blk_per_min=0.016, stl_per_min=0.033, tov_per_min=0.09),
    dict(sport="nba", kind="nba", name="Jalen Brunson", team="NYK", opponent="ORL",
         minutes=35.0, minutes_sd=2.8, pts_per_min=0.79, reb_per_min=0.10, ast_per_min=0.19,
         threes_per_min=0.070, blk_per_min=0.005, stl_per_min=0.025, tov_per_min=0.075),
    dict(sport="nba", kind="nba", name="Tyrese Haliburton", team="IND", opponent="CHI",
         minutes=33.6, minutes_sd=3.4, pts_per_min=0.56, reb_per_min=0.11, ast_per_min=0.29,
         threes_per_min=0.095, blk_per_min=0.017, stl_per_min=0.036, tov_per_min=0.065),
    dict(sport="nba", kind="nba", name="Rudy Gobert", team="MIN", opponent="SAC",
         minutes=31.2, minutes_sd=4.0, pts_per_min=0.42, reb_per_min=0.40, ast_per_min=0.05,
         threes_per_min=0.000, blk_per_min=0.055, stl_per_min=0.022, tov_per_min=0.055),
    dict(sport="nba", kind="nba", name="Domantas Sabonis", team="SAC", opponent="MIN",
         minutes=34.8, minutes_sd=3.0, pts_per_min=0.54, reb_per_min=0.38, ast_per_min=0.24,
         threes_per_min=0.020, blk_per_min=0.012, stl_per_min=0.023, tov_per_min=0.095),
    # ---- MLB
    dict(sport="mlb", kind="mlb_pitcher", name="Tarik Skubal", team="DET", opponent="CLE",
         batters_faced=25.5, k_rate=0.315, hit_rate=0.190, er_rate=0.075),
    dict(sport="mlb", kind="mlb_pitcher", name="Zack Wheeler", team="PHI", opponent="ATL",
         batters_faced=24.8, k_rate=0.285, hit_rate=0.205, er_rate=0.085),
    dict(sport="mlb", kind="mlb_batter", name="Aaron Judge", team="NYY", opponent="BOS",
         pa=4.3, hit_rate=0.295, hr_rate=0.082, rbi_rate=0.21, run_rate=0.22, sb_rate=0.012),
    dict(sport="mlb", kind="mlb_batter", name="Bobby Witt Jr.", team="KC", opponent="MIN",
         pa=4.4, hit_rate=0.315, hr_rate=0.042, rbi_rate=0.17, run_rate=0.21, sb_rate=0.055),
    # ---- NFL
    dict(sport="nfl", kind="nfl_qb", name="Patrick Mahomes", team="KC", opponent="LV",
         attempts=35.0, comp_rate=0.672, ypa=7.15, td_rate=0.052),
    dict(sport="nfl", kind="nfl_qb", name="Josh Allen", team="BUF", opponent="MIA",
         attempts=32.5, comp_rate=0.635, ypa=7.35, td_rate=0.057),
    dict(sport="nfl", kind="nfl_wr", name="CeeDee Lamb", team="DAL", opponent="PHI",
         targets=10.2, targets_sd=2.6, catch_rate=0.685, ypr=12.6),
    dict(sport="nfl", kind="nfl_rb", name="Bijan Robinson", team="ATL", opponent="TB",
         carries=17.5, ypc=4.55, targets=4.2),
]

OPPONENT_DATES = [date(2026, 8, 10) - timedelta(days=3 * i) for i in range(N_GAMES)]
OPPONENTS = ["BOS", "DEN", "LAL", "PHX", "MIA", "GSW", "OKC", "DAL", "NYK", "CLE"]


def build_gamelogs() -> None:
    for player in PLAYERS:
        generate = GENERATORS[player["kind"]]
        games = []
        for i in range(N_GAMES):
            box = generate(player)
            games.append(
                {
                    "date": OPPONENT_DATES[i].isoformat(),
                    "opponent": OPPONENTS[i % len(OPPONENTS)],
                    "home": i % 2 == 0,
                    **box,
                }
            )
        games.sort(key=lambda g: g["date"])
        folder = ROOT / "data" / "gamelogs" / player["sport"]
        folder.mkdir(parents=True, exist_ok=True)
        slug = player["name"].lower().replace(".", "").replace(" ", "-")
        (folder / f"{slug}.json").write_text(
            json.dumps(
                {"player": player["name"], "team": player["team"], "sport": player["sport"], "games": games},
                indent=1,
            )
            + "\n"
        )


# --------------------------------------------------------------------------
# the props
# --------------------------------------------------------------------------
# (player, market, half-point offset from the player's own sample mean,
#  whole-number line?, stale book). A stale book is left a few points of
# probability behind the rest of the market: the only real edge on the board.
PROPS: list[tuple[str, str, float, bool, str | None]] = [
    ("Anthony Edwards", "player_points", 0, False, "caesars"),
    ("Anthony Edwards", "player_rebounds", 0, False, None),
    ("Jalen Brunson", "player_points", 1, False, None),
    ("Jalen Brunson", "player_assists", 0, False, None),
    ("Tyrese Haliburton", "player_assists", 0, False, "betmgm"),
    ("Tyrese Haliburton", "player_points", -1, False, None),
    ("Rudy Gobert", "player_rebounds", 0, True, None),  # whole number: can push
    ("Domantas Sabonis", "player_points_rebounds_assists", 0, False, None),
    ("Tarik Skubal", "pitcher_strikeouts", 0, False, "fanduel"),
    ("Zack Wheeler", "pitcher_strikeouts", 0, False, None),
    ("Aaron Judge", "batter_total_bases", 0, False, None),
    ("Bobby Witt Jr.", "batter_hits", 0, True, None),  # whole number: can push
    ("Patrick Mahomes", "player_pass_yds", 0, False, None),
    ("Josh Allen", "player_pass_yds", 1, False, "draftkings"),
    ("CeeDee Lamb", "player_receptions", 0, False, None),
    ("Bijan Robinson", "player_rush_yds", -1, False, None),
]

MATCHUPS = {
    "nba": [("SAC", "MIN"), ("ORL", "NYK"), ("CHI", "IND")],
    "mlb": [("CLE", "DET"), ("ATL", "PHI"), ("BOS", "NYY"), ("MIN", "KC")],
    "nfl": [("LV", "KC"), ("MIA", "BUF"), ("PHI", "DAL"), ("TB", "ATL")],
}

MARKET_FIELDS: dict[str, tuple[str, ...]] = {
    "player_points": ("points",),
    "player_rebounds": ("rebounds",),
    "player_assists": ("assists",),
    "player_points_rebounds_assists": ("points", "rebounds", "assists"),
    "pitcher_strikeouts": ("strikeouts",),
    "batter_total_bases": ("total_bases",),
    "batter_hits": ("hits",),
    "player_pass_yds": ("pass_yards",),
    "player_receptions": ("receptions",),
    "player_rush_yds": ("rush_yards",),
}


def sample_values(player: dict, market: str) -> list[float]:
    """The stat, game by game, from the logs this player just got written."""
    slug = player["name"].lower().replace(".", "").replace(" ", "-")
    path = ROOT / "data" / "gamelogs" / player["sport"] / f"{slug}.json"
    games = json.loads(path.read_text())["games"]
    return [sum(g[f] for f in MARKET_FIELDS[market]) for g in games]


def matchup_factor(sport: str, market: str, opponent: str, is_home: bool) -> float:
    """The same opponent/pace/venue multiplier the bot's MatchupBook applies.

    The synthetic book has to know about the matchup too. If the model
    adjusts for a slow defence and the sample prices don't, every prop in
    that game looks like edge -- and it's an artefact of the fixture, not a
    finding.
    """
    book = DEFENSE.get(sport, {})
    factor = float(book.get("markets", {}).get(market, {}).get(opponent, 1.0))
    factor *= float(book.get("pace", {}).get(opponent, 1.0))
    home = float(book.get("home_factor", 1.0))
    return factor * (home if is_home else 1.0 / home)


def market_view(
    player: dict, market: str, line: float, adjust: float = 1.0, sims: int = 60_000
) -> tuple[float, float]:
    """What a well-informed book would post: P(over), P(push) at the line.

    The book is modelled as knowing the *shape* of the player's distribution
    and its true centre -- but the centre it knows is the one the logs
    actually realised, not the population parameter. Otherwise the bot would
    be rewarded for the gap between a 46-game sample and the truth behind it,
    which is sampling error dressed up as edge.
    """
    generate = GENERATORS[player["kind"]]
    fields = MARKET_FIELDS[market]
    draws = [sum(generate(player)[f] for f in fields) for _ in range(sims)]
    population_mean = sum(draws) / sims
    observed = sample_values(player, market)
    scale = adjust * (sum(observed) / len(observed)) / max(population_mean, 1e-9)

    counts = all(float(v).is_integer() for v in observed)
    over = push = 0
    for value in draws:
        v = round(value * scale) if counts else value * scale
        if abs(v - line) < 1e-9:
            push += 1
        elif v > line:
            over += 1
    return over / sims, push / sims


def pick_line(player: dict, market: str, offset: float, whole: bool, adjust: float = 1.0) -> float:
    """A realistic line: near the player's own mean, on a half or whole point."""
    observed = sample_values(player, market)
    base = adjust * sum(observed) / len(observed)
    if whole:
        line = float(round(base))
    else:
        line = round(base * 2.0) / 2.0
        if float(line).is_integer():
            line += 0.5
    step = 0.5 if base < 30 else (5.0 if base > 100 else 1.0)
    return line + offset * step


def american_from_prob(p: float) -> int:
    p = min(max(p, 0.02), 0.98)
    decimal = 1.0 / p
    raw = (decimal - 1.0) * 100.0 if decimal >= 2.0 else -100.0 / (decimal - 1.0)
    return int(round(raw / 5.0) * 5)  # books post round numbers


# Mild, plausible matchup multipliers. 1.0 means "no information". Both the
# bot (via data/defense.json) and the synthetic book read this same table.
DEFENSE: dict[str, dict] = {
    "nba": {
        "home_factor": 1.010,
        "pace": {"SAC": 1.030, "IND": 1.025, "MIN": 0.975, "ORL": 0.965, "CHI": 1.005, "NYK": 0.985},
        "markets": {
            "player_points": {"MIN": 0.955, "ORL": 0.965, "SAC": 1.035, "CHI": 1.020, "IND": 1.015},
            "player_rebounds": {"MIN": 0.980, "SAC": 1.010, "CHI": 1.005},
            "player_assists": {"MIN": 0.975, "CHI": 1.015, "ORL": 0.980},
            "player_points_rebounds_assists": {"MIN": 0.970, "SAC": 1.020},
        },
    },
    "mlb": {
        "home_factor": 1.008,
        "pace": {},
        "markets": {
            "pitcher_strikeouts": {"CLE": 0.960, "ATL": 1.020, "MIN": 0.985},
            "batter_total_bases": {"BOS": 1.030, "MIN": 0.975},
            "batter_hits": {"MIN": 0.985, "BOS": 1.020},
        },
    },
    "nfl": {
        "home_factor": 1.012,
        "pace": {"LV": 1.010, "MIA": 1.020, "PHI": 0.975, "TB": 0.995},
        "markets": {
            "player_pass_yds": {"LV": 1.035, "MIA": 1.010},
            "player_receptions": {"PHI": 0.955},
            "player_rush_yds": {"TB": 0.930},
        },
    },
}


def build_props() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    per_sport: dict[str, dict] = {}
    by_name = {p["name"]: p for p in PLAYERS}

    for name, market, offset, whole, stale in PROPS:
        player = by_name[name]
        sport = player["sport"]
        event_key = (player["team"], player["opponent"])
        away, home = sorted(event_key)  # deterministic; the sample slate is neutral
        adjust = matchup_factor(sport, market, player["opponent"], player["team"] == home)

        line = pick_line(player, market, offset, whole, adjust)
        p_over, p_push = market_view(player, market, line, adjust=adjust)
        live = max(1.0 - p_push, 1e-9)
        fair_over = p_over / live  # books quote conditional on no push

        payload = per_sport.setdefault(sport, {"sport": sport, "events": {}})
        event_id = f"{sport}-{away}-{home}".lower()
        event = payload["events"].setdefault(
            event_id,
            {
                "id": event_id,
                "commence_time": (now + timedelta(hours=6 + 2 * len(payload["events"]))).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "home_team": home,
                "away_team": away,
                "props": [],
            },
        )

        books = []
        for book in BOOKS:
            # Each book's own read on the number, then ~4.4% of vig split evenly.
            noise = rng.gauss(0.0, 0.012)
            est_over = min(max(fair_over + noise, 0.05), 0.95)
            if book == stale:
                # Stale by 6-7 points of probability on the over: real value.
                est_over = min(max(est_over - 0.065, 0.05), 0.95)
            vig = 1.044
            books.append(
                {
                    "book": book,
                    "line": line,
                    "over": american_from_prob(est_over * vig),
                    "under": american_from_prob((1.0 - est_over) * vig),
                }
            )
        event["props"].append(
            {
                "player": name,
                "team": player["team"],
                "opponent": player["opponent"],
                "market": market,
                "books": books,
            }
        )

    folder = ROOT / "data" / "props"
    folder.mkdir(parents=True, exist_ok=True)
    for sport, payload in per_sport.items():
        payload["events"] = list(payload["events"].values())
        (folder / f"{sport}_sample.json").write_text(json.dumps(payload, indent=1) + "\n")


def build_defense() -> None:
    (ROOT / "data" / "defense.json").write_text(json.dumps(DEFENSE, indent=1) + "\n")


if __name__ == "__main__":
    build_gamelogs()
    build_props()
    build_defense()
    print(f"wrote sample data for {len(PLAYERS)} players and {len(PROPS)} props under {ROOT / 'data'}")
