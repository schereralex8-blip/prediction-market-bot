"""Model A -- empirical-Bayes rate model with a parametric outcome family.

The idea: a player's production is a *rate* per unit of opportunity (points
per minute, strikeouts per batter faced, yards per target) multiplied by how
much opportunity they'll get. Those two things behave differently -- rate is
a slow-moving skill, workload is a fast-moving role decision -- so they get
estimated separately, with different memory.

Shrinkage happens in two stages, and the order matters:

1. The player's *career* rate is shrunk toward the league prior. This is a
   weak pull, because between-player spread dwarfs within-player noise.
2. The player's *recent* (recency-weighted) rate is shrunk toward their own
   career rate. This is the pull that does the real work -- it damps a hot
   week without dragging a star toward a league average they have never been
   near.

Doing it in one stage, straight to the league mean, is the bug that makes a
28-point-per-game scorer project for 22 and turns the model into a permanent
under-bettor.

Uncertainty is then stacked in three layers:

    Var(total) = workload * Var(per-unit game noise)      # how the night goes
               + rate^2   * Var(workload)                 # how long they play
               + workload^2 * Var(rate estimate)          # what we don't know

Skipping the second term is why naive models are wildly overconfident on
passing yards -- attempts swing from 28 to 44, and that swing is most of the
variance. Skipping the third produces 65% confidence in props that are really
55%, and bets into vig while feeling clever.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..config import ModelSettings
from ..distributions import (
    Distribution,
    Normal,
    betabinom_from_moments,
    negbin_from_moments,
)
from ..markets import MarketSpec
from ..types import GameLog, MatchupContext, ModelOutput
from .base import EPS, Fit, decay_weights, insufficient, recent_first, stat_of, workload_of

NAME = "bayes_rate"


class BayesRateModel:
    """Parametric, shrinkage-based. See module docstring."""

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

        rate_w = decay_weights(len(played), cfg.half_life_games)
        stats = [stat_of(g, spec) for g in played]
        loads = [workload_of(g, spec) for g in played]

        # --- 1. stage one: career rate, weakly pulled to the league prior ---
        k_league, r_league = spec.prior_strength, spec.prior_rate
        career_load = sum(loads)
        career_rate = (k_league * r_league + sum(stats)) / max(k_league + career_load, EPS)

        # --- 2. stage two: recent form, pulled back to that career rate -----
        # k0 is about one game of workload: enough to damp a hot week, not
        # enough to argue with 40 games of evidence.
        k0 = max(career_load / len(played), EPS)
        num = k0 * career_rate + sum(w * s for w, s in zip(rate_w, stats))
        den = k0 + sum(w * l for w, l in zip(rate_w, loads))
        rate = num / max(den, EPS)

        # --- 3. workload projection (shorter memory than the rate) ---------
        load_w = decay_weights(len(played), cfg.workload_half_life)
        load_proj = sum(w * l for w, l in zip(load_w, loads)) / sum(load_w)
        var_load = sum(w * (l - load_proj) ** 2 for w, l in zip(load_w, loads)) / sum(load_w)
        if context.workload_override is not None:
            load_proj = context.workload_override
            # A stated cap is information, not certainty: keep 10% of it.
            var_load = (0.10 * load_proj) ** 2
        load_proj = max(load_proj, EPS)

        # --- 4. mean -------------------------------------------------------
        adj = context.combined
        mean = max(rate * load_proj * adj, 0.05)

        # --- 5. variance, in three parts -----------------------------------
        wsum = sum(rate_w)
        # (a) game-to-game noise given workload: Var(stat | load) = s2u * load
        s2u = sum(
            w * (s - rate * l) ** 2 / max(l, EPS) for w, s, l in zip(rate_w, stats, loads)
        ) / wsum
        # (b) how much the workload itself swings
        var_workload = (rate * adj) ** 2 * var_load
        # (c) what we don't know about the rate: SE^2 of a weighted rate estimate
        var_rate = s2u / max(den, EPS)
        var_param = (load_proj * adj) ** 2 * var_rate

        dist, extra = self._distribution(
            spec=spec,
            mean=mean,
            rate=rate * adj,
            load_proj=load_proj,
            s2u=s2u,
            var_workload=var_workload,
            var_param=var_param,
            eff_n=den,
            weights=rate_w,
            stats=stats,
            loads=loads,
        )
        return Fit(
            ok=True,
            dist=dist,
            n_games=len(played),
            details={
                "rate": rate,
                "career_rate": career_rate,
                "league_prior": r_league,
                "league_pull": k_league / max(k_league + career_load, EPS),
                "workload_proj": load_proj,
                "workload_sd": math.sqrt(max(var_load, 0.0)),
                "matchup_adj": adj,
                "family": type(dist).__name__,
                "var_share_workload": var_workload / max(dist.sd**2, EPS),
                "var_share_estimation": var_param / max(dist.sd**2, EPS),
                "dnp_games_dropped": len(games) - len(played),
                **extra,
            },
        )

    # ------------------------------------------------------------------
    def _distribution(
        self,
        *,
        spec: MarketSpec,
        mean: float,
        rate: float,
        load_proj: float,
        s2u: float,
        var_workload: float,
        var_param: float,
        eff_n: float,
        weights: Sequence[float],
        stats: Sequence[float],
        loads: Sequence[float],
    ) -> tuple[Distribution, dict[str, float]]:
        extra = var_workload + var_param  # the two terms every family shares

        if spec.family == "normal":
            var = s2u * load_proj + extra
            sd = max(math.sqrt(max(var, EPS)), spec.sd_floor_factor * math.sqrt(mean))
            return Normal(mean, sd), {"resid_var_per_unit": s2u}

        if spec.family == "poisson":
            return negbin_from_moments(mean, mean + extra), {}

        if spec.family == "negbin":
            # Pearson dispersion: how much fatter than Poisson the counts run.
            wsum = sum(weights)
            phi = sum(
                w * (s - rate * l) ** 2 / max(rate * l, 0.25)
                for w, s, l in zip(weights, stats, loads)
            ) / wsum
            phi = min(max(phi, 1.0), 4.0)
            return negbin_from_moments(mean, phi * mean + extra), {"dispersion": phi}

        if spec.family == "binomial":
            n = max(1, int(round(load_proj)))
            p = min(max(mean / n, 1e-4), spec.max_rate)
            var = n * p * (1.0 - p) + extra
            return (
                betabinom_from_moments(n, n * p, var),
                {"n_trials": float(n), "success_rate": p},
            )

        raise ValueError(f"unsupported family {spec.family!r} for market {spec.key}")
