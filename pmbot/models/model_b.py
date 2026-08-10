"""Model B -- recency-weighted bootstrap simulation.

Deliberately built on different assumptions from Model A, because two models
that share assumptions agree even when they're both wrong, and an agreement
gate over two copies of the same model is theatre.

What differs:

* **No distributional family.** The shape of the outcome comes from
  resampling the player's own games, so a lumpy three-point shooter or a
  boom-bust running back keeps their real shape instead of being forced into
  a negative binomial.
* **Robust workload.** A weighted *median* of recent workload, so one blowout
  or one ejection doesn't move the projection the way a mean does.
* **Role similarity.** Games where the player's workload resembled tonight's
  expected workload count for more -- 14 minutes in garbage time says little
  about a 34-minute night.
* **Explicit variance decomposition.** A single game's rate is noisy. For
  counting stats the observed spread in per-game rates is split into true
  rate variation and pure counting noise, so the simulation doesn't
  double-count randomness that the Poisson/binomial draw already supplies.
* **Uncertainty in the matchup itself**, drawn per simulation rather than
  applied as a point estimate.
"""

from __future__ import annotations

import math
import random
import zlib
from typing import Sequence

from ..config import ModelSettings
from ..distributions import Empirical
from ..markets import MarketSpec
from ..types import GameLog, MatchupContext, ModelOutput
from .base import EPS, Fit, decay_weights, insufficient, recent_first, stat_of, workload_of

NAME = "bootstrap_mc"


class BootstrapModel:
    """Nonparametric Monte Carlo. See module docstring."""

    name = NAME

    def __init__(self, settings: ModelSettings | None = None) -> None:
        self.settings = settings or ModelSettings()

    # ------------------------------------------------------------------
    def predict(
        self,
        *,
        spec: MarketSpec,
        logs: Sequence[GameLog],
        line: float,
        context: MatchupContext | None = None,
    ) -> ModelOutput:
        return self.fit(spec=spec, logs=logs, context=context).at(self.name, line)

    def fit(
        self,
        *,
        spec: MarketSpec,
        logs: Sequence[GameLog],
        context: MatchupContext | None = None,
    ) -> Fit:
        cfg = self.settings
        context = context or MatchupContext()

        games = recent_first(logs, cfg.max_games)
        played = [g for g in games if workload_of(g, spec) > EPS]
        if len(played) < cfg.min_games:
            return insufficient(
                len(played), f"only {len(played)} usable games (need {cfg.min_games})"
            )

        loads = [workload_of(g, spec) for g in played]
        stats = [stat_of(g, spec) for g in played]
        rates = [s / l for s, l in zip(stats, loads)]
        recency = decay_weights(len(played), cfg.mc_half_life_games)

        # --- expected role tonight, robustly -------------------------------
        target_load = (
            context.workload_override
            if context.workload_override is not None
            else _weighted_median(loads, recency)
        )
        target_load = max(target_load, EPS)

        # --- weight games by recency AND role similarity --------------------
        load_spread = max(_weighted_sd(loads, recency), 0.15 * target_load, EPS)
        weights = [
            w * math.exp(-0.5 * ((l - target_load) / load_spread) ** 2)
            for w, l in zip(recency, loads)
        ]
        if sum(weights) <= EPS:  # pragma: no cover - degenerate
            weights = list(recency)

        rbar = _weighted_mean(rates, weights)
        var_r = _weighted_var(rates, weights, rbar)
        mean_load = _weighted_mean(loads, weights)

        # Split observed rate spread into "true skill/role variation" and the
        # counting noise that the per-sim Poisson/binomial draw will re-add.
        shrink = 1.0
        if spec.is_count and var_r > EPS:
            counting_component = rbar / max(mean_load, EPS)
            var_true = max(var_r - counting_component, 0.05 * var_r)
            shrink = math.sqrt(var_true / var_r)

        rng = random.Random(_seed(cfg.seed, spec.key, played[0].player))
        # Rates come from role-comparable games; workload comes from the plain
        # recency distribution. Drawing workload from the similarity-weighted
        # set instead would resample around the *median* workload and quietly
        # under-project anyone whose minutes are right-skewed.
        cum_rate = _cumulative(weights)
        cum_load = _cumulative(recency)
        sigma_m = max(cfg.matchup_uncertainty, 1e-6)
        mu_m = math.log(max(context.combined, 1e-6)) - 0.5 * sigma_m**2
        jitter = max(cfg.workload_jitter, 1e-6)

        samples: list[float] = []
        for _ in range(cfg.mc_sims):
            # workload for tonight
            if context.workload_override is not None:
                load_draw = context.workload_override
            else:
                load_draw = loads[_pick(cum_load, rng.random())]
            load_draw = max(load_draw * math.exp(rng.gauss(-0.5 * jitter**2, jitter)), 0.05)

            # rate for tonight, resampled from a comparable game
            i = _pick(cum_rate, rng.random())
            r_i = rates[i]
            if spec.is_count:
                rate_draw = rbar + (r_i - rbar) * shrink
            else:
                # A deviation seen over loads[i] units is noisier than the same
                # deviation over load_draw units: Var(stat) ~ load, so the
                # per-unit deviation rescales by sqrt(loads[i] / load_draw).
                scale = math.sqrt(max(loads[i], EPS) / load_draw)
                rate_draw = rbar + (r_i - rbar) * min(scale, 3.0)
            rate_draw = max(rate_draw, 0.0)

            m = math.exp(rng.gauss(mu_m, sigma_m))
            lam = rate_draw * load_draw * m

            if spec.family == "binomial":
                n = max(1, int(round(load_draw)))
                p = min(max(rate_draw * m, 0.0), spec.max_rate)
                samples.append(_binomial_sample(rng, n, p))
            elif spec.is_count:
                samples.append(_poisson_sample(rng, lam))
            else:
                samples.append(lam)

        dist = Empirical(samples)
        return Fit(
            ok=True,
            dist=dist,
            n_games=len(played),
            details={
                "target_workload": target_load,
                "rate_mean": rbar,
                "rate_shrink": shrink,
                "sims": cfg.mc_sims,
                "matchup_adj": context.combined,
                "p10": dist.quantile(0.10),
                "p90": dist.quantile(0.90),
                "effective_games": _effective_sample_size(weights),
            },
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _seed(base: int, market: str, player: str) -> int:
    """Stable per-prop seed: same inputs always give the same simulation."""
    key = f"{market}|{player}".encode()
    return (base ^ zlib.crc32(key)) & 0x7FFFFFFF


def _cumulative(weights: Sequence[float]) -> list[float]:
    total = sum(weights)
    out: list[float] = []
    acc = 0.0
    for w in weights:
        acc += w / total
        out.append(acc)
    out[-1] = 1.0
    return out


def _pick(cumulative: Sequence[float], u: float) -> int:
    import bisect

    return min(bisect.bisect_left(cumulative, u), len(cumulative) - 1)


def _weighted_mean(xs: Sequence[float], ws: Sequence[float]) -> float:
    return sum(w * x for w, x in zip(ws, xs)) / max(sum(ws), EPS)


def _weighted_var(xs: Sequence[float], ws: Sequence[float], mean: float | None = None) -> float:
    m = _weighted_mean(xs, ws) if mean is None else mean
    return sum(w * (x - m) ** 2 for w, x in zip(ws, xs)) / max(sum(ws), EPS)


def _weighted_sd(xs: Sequence[float], ws: Sequence[float]) -> float:
    return math.sqrt(max(_weighted_var(xs, ws), 0.0))


def _weighted_median(xs: Sequence[float], ws: Sequence[float]) -> float:
    pairs = sorted(zip(xs, ws))
    total = sum(ws)
    acc = 0.0
    for x, w in pairs:
        acc += w
        if acc >= 0.5 * total:
            return x
    return pairs[-1][0]  # pragma: no cover - unreachable for positive weights


def _effective_sample_size(ws: Sequence[float]) -> float:
    """Kish's effective N: how many games the weighting really leans on."""
    s1 = sum(ws)
    s2 = sum(w * w for w in ws)
    return (s1 * s1) / s2 if s2 > 0 else 0.0


def _poisson_sample(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam < 30.0:
        target = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= rng.random()
            if p <= target:
                return k
            k += 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


def _binomial_sample(rng: random.Random, n: int, p: float) -> int:
    if p <= 0:
        return 0
    if p >= 1:
        return n
    if n > 60:  # normal approximation is fine this far out
        x = rng.gauss(n * p, math.sqrt(n * p * (1 - p)))
        return min(n, max(0, int(round(x))))
    return sum(1 for _ in range(n) if rng.random() < p)
