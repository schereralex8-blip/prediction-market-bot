"""Configuration: every knob that decides what counts as a bet.

Defaults are deliberately conservative. Loosening them (more Kelly, less edge
required, more model disagreement tolerated) is exactly how a positive-EV
strategy turns into a negative one, so each one carries a note about which
direction is dangerous.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("pmbot.config.json")


@dataclass
class DevigSettings:
    method: str = "power"  # multiplicative | additive | power | shin | conservative
    exclude_own_book: bool = True
    """Price a bet against the *other* books' consensus. Including the book
    you're betting drags the fair line toward its own price and manufactures
    edge out of nothing."""
    min_books: int = 2  # fewer quotes than this and the consensus isn't one
    sharp_books: tuple[str, ...] = ("pinnacle", "circa", "bookmaker", "betonlineag")
    sharp_weight: float = 3.0  # weight multiplier for the books above
    max_line_gap: float = 2.5
    """How far a book's line may sit from the line being priced before its
    quote is dropped from the interpolation."""


@dataclass
class ModelSettings:
    # --- model A: Bayesian rate model
    half_life_games: float = 12.0  # recency decay for the rate estimate
    workload_half_life: float = 6.0  # workload moves faster than skill
    min_games: int = 8
    # --- model B: bootstrap Monte Carlo (deliberately different knobs)
    mc_sims: int = 20000
    mc_half_life_games: float = 20.0
    matchup_uncertainty: float = 0.08  # lognormal sd on the opponent factor
    workload_jitter: float = 0.12  # multiplicative noise on projected workload
    seed: int = 20250817
    max_games: int = 60  # ignore ancient history entirely


@dataclass
class GateSettings:
    """The two-model agreement gate. This is the part that says 'pass'."""

    max_disagreement: float = 0.06
    """Max |p_A - p_B| on the same side. Two models that disagree by more than
    this are not two confirmations, they're a coin flip."""
    min_model_edge: float = 0.015
    """Each model, on its own, must beat the price's breakeven probability by
    this much. Measured against the price, not against the market: a book
    hanging +135 into a -115 market is an edge even when the model agrees
    with the market exactly."""
    max_market_divergence: float = 0.15
    """If the models land this far from the whole market's fair probability,
    pass. At that distance the usual explanation is that the models are
    reading stale data -- an injury, a lineup change, a pitching switch --
    not that five books are wrong together."""
    combine: str = "min"  # min | mean | geometric -- 'min' is the cautious pick
    market_blend: float = 0.25
    """Shrink the consensus toward the market's fair probability. The market
    is the single best-informed forecaster in the room; 0.25 says 'we are
    three parts model, one part humility'."""


@dataclass
class StakingSettings:
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.02
    max_slate_fraction: float = 0.10
    min_stake: float = 1.0
    rounding: float = 1.0
    min_ev: float = 0.02  # 2% EV floor; below this the model noise dominates
    min_price_edge: float = 0.02
    """Consensus probability minus the price's breakeven probability. This is
    the quantity EV is proportional to, and the one worth a floor."""
    max_american: float = 400.0  # skip lottery-ticket prices
    min_american: float = -350.0  # and skip prices where one loss undoes ten wins


@dataclass
class DataSettings:
    gamelogs: str = "data/gamelogs"
    props: str = "data/props"
    defense: str = "data/defense.json"
    db: str = "pmbot.sqlite3"
    cache: str = ".cache/odds"


@dataclass
class ApiSettings:
    provider: str = "local"  # local | theoddsapi
    api_key: str = ""
    base_url: str = "https://api.the-odds-api.com/v4"
    regions: str = "us,us2"
    odds_format: str = "american"
    timeout: float = 20.0
    cache_ttl: float = 120.0  # seconds; props move fast, but not that fast


@dataclass
class Settings:
    devig: DevigSettings = field(default_factory=DevigSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    gate: GateSettings = field(default_factory=GateSettings)
    staking: StakingSettings = field(default_factory=StakingSettings)
    data: DataSettings = field(default_factory=DataSettings)
    api: ApiSettings = field(default_factory=ApiSettings)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        """Load defaults, overlay a JSON config file, then overlay env vars."""
        settings = cls()
        candidate = Path(path) if path else Path(os.environ.get("PMBOT_CONFIG", DEFAULT_CONFIG_PATH))
        if candidate.exists():
            with candidate.open() as fh:
                settings.apply(json.load(fh))
        settings.apply_env()
        return settings

    def apply(self, data: dict[str, Any]) -> "Settings":
        """Overlay a nested dict of overrides, ignoring unknown keys loudly."""
        for section, values in data.items():
            if not hasattr(self, section):
                raise KeyError(f"unknown config section {section!r}")
            target = getattr(self, section)
            if not is_dataclass(target) or not isinstance(values, dict):
                raise TypeError(f"config section {section!r} must be an object")
            known = {f.name: f for f in fields(target)}
            for key, value in values.items():
                if key not in known:
                    raise KeyError(f"unknown config key {section}.{key}")
                current = getattr(target, key)
                if isinstance(current, tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(target, key, value)
        return self

    def apply_env(self) -> "Settings":
        env_map = {
            "PMBOT_ODDS_API_KEY": ("api", "api_key"),
            "PMBOT_PROVIDER": ("api", "provider"),
            "PMBOT_DB": ("data", "db"),
            "PMBOT_BANKROLL": ("staking", "bankroll"),
            "PMBOT_KELLY_FRACTION": ("staking", "kelly_fraction"),
            "PMBOT_DEVIG": ("devig", "method"),
        }
        for env_key, (section, key) in env_map.items():
            raw = os.environ.get(env_key)
            if raw is None or raw == "":
                continue
            target = getattr(self, section)
            current = getattr(target, key)
            setattr(target, key, type(current)(raw) if not isinstance(current, bool) else raw.lower() in ("1", "true", "yes"))
        if self.api.api_key and self.api.provider == "local" and "PMBOT_PROVIDER" not in os.environ:
            # An API key on its own is a clear statement of intent.
            self.api.provider = "theoddsapi"
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=list) + "\n")
        return p
