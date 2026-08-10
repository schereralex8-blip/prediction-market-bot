"""Which player props we cover, and how each one behaves statistically.

Each entry maps a bookmaker market key to the game-log fields that produce the
stat, the workload it should be modelled per (minutes, plate appearances,
targets...), and the distribution family that actually fits it.

``prior_rate`` / ``prior_strength`` define the league-average prior the model
shrinks toward. The strengths are deliberately *small* -- roughly one game's
worth of workload -- because between-player spread in these rates is enormous
next to within-player noise. Shrinking a 0.73 points-per-minute scorer toward
a 0.46 league average is not regularisation, it's a projection that will be
wrong by four points every night and quietly bet the under forever. The prior
exists to stop a six-game sample from exploding, nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SPORTS: tuple[str, ...] = ("nba", "mlb", "nfl")

SPORT_KEYS: dict[str, str] = {
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nfl": "americanfootball_nfl",
}


@dataclass(frozen=True)
class MarketSpec:
    key: str  # bookmaker market key, e.g. "player_points"
    sport: str
    label: str
    components: tuple[str, ...]  # game-log fields summed to form the stat
    workload: str | None  # game-log field the rate is computed per; None = per game
    family: str  # "normal" | "poisson" | "negbin" | "binomial"
    prior_rate: float  # league-average stat per unit of workload
    prior_strength: float  # prior weight, in workload units (~one game; see above)
    sd_floor_factor: float = 0.9  # normal only: sd >= factor * sqrt(mean)
    max_rate: float = float("inf")  # binomial rates are probabilities
    aliases: tuple[str, ...] = field(default=())

    @property
    def is_count(self) -> bool:
        return self.family in ("poisson", "negbin", "binomial")


_SPECS: tuple[MarketSpec, ...] = (
    # ---------------------------------------------------------------- NBA
    MarketSpec("player_points", "nba", "Points", ("points",), "minutes", "normal", 0.460, 25, 0.95, aliases=("pts",)),
    MarketSpec("player_rebounds", "nba", "Rebounds", ("rebounds",), "minutes", "negbin", 0.155, 30, aliases=("reb",)),
    MarketSpec("player_assists", "nba", "Assists", ("assists",), "minutes", "negbin", 0.110, 30, aliases=("ast",)),
    MarketSpec("player_threes", "nba", "Made threes", ("threes",), "minutes", "negbin", 0.055, 40, aliases=("3pm",)),
    MarketSpec("player_blocks", "nba", "Blocks", ("blocks",), "minutes", "poisson", 0.018, 70, aliases=("blk",)),
    MarketSpec("player_steals", "nba", "Steals", ("steals",), "minutes", "poisson", 0.026, 70, aliases=("stl",)),
    MarketSpec("player_turnovers", "nba", "Turnovers", ("turnovers",), "minutes", "negbin", 0.050, 45, aliases=("to",)),
    MarketSpec(
        "player_points_rebounds_assists", "nba", "PRA",
        ("points", "rebounds", "assists"), "minutes", "normal", 0.725, 25, 0.85, aliases=("pra",),
    ),
    MarketSpec(
        "player_points_rebounds", "nba", "Points + rebounds",
        ("points", "rebounds"), "minutes", "normal", 0.615, 25, 0.9, aliases=("pr",),
    ),
    MarketSpec(
        "player_points_assists", "nba", "Points + assists",
        ("points", "assists"), "minutes", "normal", 0.570, 25, 0.9, aliases=("pa",),
    ),
    MarketSpec(
        "player_rebounds_assists", "nba", "Rebounds + assists",
        ("rebounds", "assists"), "minutes", "negbin", 0.265, 30, aliases=("ra",),
    ),
    # ---------------------------------------------------------------- MLB
    MarketSpec(
        "batter_hits", "mlb", "Hits", ("hits",), "plate_appearances", "binomial",
        0.235, 25, max_rate=0.95, aliases=("hits",),
    ),
    MarketSpec(
        "batter_total_bases", "mlb", "Total bases", ("total_bases",), "plate_appearances",
        "negbin", 0.395, 25, aliases=("tb",),
    ),
    MarketSpec(
        "batter_home_runs", "mlb", "Home runs", ("home_runs",), "plate_appearances",
        "poisson", 0.034, 60, aliases=("hr",),
    ),
    MarketSpec("batter_rbis", "mlb", "RBIs", ("rbis",), "plate_appearances", "negbin", 0.120, 40),
    MarketSpec(
        "batter_runs_scored", "mlb", "Runs", ("runs",), "plate_appearances", "negbin", 0.128, 40,
    ),
    MarketSpec(
        "batter_stolen_bases", "mlb", "Stolen bases", ("stolen_bases",), "plate_appearances",
        "poisson", 0.019, 60,
    ),
    MarketSpec(
        "pitcher_strikeouts", "mlb", "Strikeouts", ("strikeouts",), "batters_faced", "negbin",
        0.230, 30, aliases=("k", "ks"),
    ),
    MarketSpec(
        "pitcher_outs", "mlb", "Outs recorded", ("outs",), "batters_faced", "normal", 0.680, 30, 0.7,
    ),
    MarketSpec(
        "pitcher_hits_allowed", "mlb", "Hits allowed", ("hits_allowed",), "batters_faced",
        "negbin", 0.212, 30,
    ),
    MarketSpec(
        "pitcher_earned_runs", "mlb", "Earned runs", ("earned_runs",), "batters_faced",
        "negbin", 0.113, 40,
    ),
    # ---------------------------------------------------------------- NFL
    MarketSpec(
        "player_pass_yds", "nfl", "Passing yards", ("pass_yards",), "pass_attempts", "normal",
        7.10, 25, 0.8, aliases=("pass_yards",),
    ),
    MarketSpec(
        "player_pass_tds", "nfl", "Passing TDs", ("pass_tds",), "pass_attempts", "poisson", 0.046, 40,
    ),
    MarketSpec(
        "player_pass_completions", "nfl", "Completions", ("completions",), "pass_attempts",
        "binomial", 0.645, 25, max_rate=0.95,
    ),
    MarketSpec(
        "player_pass_attempts", "nfl", "Pass attempts", ("pass_attempts",), None, "normal", 33.0, 2, 0.6,
    ),
    MarketSpec(
        "player_rush_yds", "nfl", "Rushing yards", ("rush_yards",), "rush_attempts", "normal",
        4.35, 20, 0.85, aliases=("rush_yards",),
    ),
    MarketSpec(
        "player_rush_attempts", "nfl", "Rush attempts", ("rush_attempts",), None, "normal", 11.5, 2, 0.7,
    ),
    MarketSpec(
        "player_receptions", "nfl", "Receptions", ("receptions",), "targets", "binomial",
        0.660, 12, max_rate=0.95, aliases=("recs",),
    ),
    MarketSpec(
        "player_reception_yds", "nfl", "Receiving yards", ("receiving_yards",), "targets",
        "normal", 8.15, 12, 0.85, aliases=("rec_yards",),
    ),
    MarketSpec(
        "player_rush_reception_yds", "nfl", "Rush + rec yards",
        ("rush_yards", "receiving_yards"), None, "normal", 62.0, 2, 0.75,
    ),
)

BY_KEY: dict[str, MarketSpec] = {s.key: s for s in _SPECS}
_ALIASES: dict[str, str] = {a: s.key for s in _SPECS for a in s.aliases}


def get_spec(key: str) -> MarketSpec:
    k = key.strip().lower()
    if k in BY_KEY:
        return BY_KEY[k]
    if k in _ALIASES:
        return BY_KEY[_ALIASES[k]]
    raise KeyError(f"unknown market {key!r}; try one of {', '.join(sorted(BY_KEY))}")


def specs_for_sport(sport: str) -> list[MarketSpec]:
    s = sport.strip().lower()
    if s not in SPORTS:
        raise KeyError(f"unknown sport {sport!r}; expected one of {', '.join(SPORTS)}")
    return [spec for spec in _SPECS if spec.sport == s]


def market_keys(sport: str) -> list[str]:
    return [spec.key for spec in specs_for_sport(sport)]


def sport_key(sport: str) -> str:
    """Bookmaker API sport key, e.g. 'nba' -> 'basketball_nba'."""
    s = sport.strip().lower()
    if s in SPORT_KEYS:
        return SPORT_KEYS[s]
    if s in SPORT_KEYS.values():
        return s
    raise KeyError(f"unknown sport {sport!r}")
