"""The scan: odds in, bets (or passes) out.

Order of operations, and why:

1. Fit both models once per player/market. Fitting doesn't depend on the
   line, so every line the books offer is priced against the same fit.
2. For each line and each side, find the best price on the board.
3. Compute the fair probability from the *other* books -- excluding the one
   offering that price, so an outlier can't validate itself.
4. Ask the agreement gate. Two models, same side, both clearing their own
   edge floor, or it's a pass.
5. Only then check EV, and only then size the stake.

A bet has to survive all five. Most don't, and the scanner is written to
explain which step killed each one, because "no bet" with a reason is a
usable answer and "no bet" without one is just silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from .config import Settings
from .ev import EvResult, evaluate as evaluate_ev
from .fairline import FairLine, best_price, fair_prob_at
from .kelly import Stake, cap_total_exposure, size_bet
from .markets import MarketSpec, get_spec, market_keys
from .models.base import Fit
from .models.consensus import evaluate_consensus
from .models.model_a import BayesRateModel
from .models.model_b import BootstrapModel
from .oddsmath import american_to_decimal, format_american, prob_to_american
from .providers import (
    GameLogProvider,
    LocalGameLogProvider,
    MatchupBook,
    OddsProvider,
    build_odds_provider,
)
from .types import Consensus, PropMarket, Signal, sort_signals


@dataclass(frozen=True)
class Candidate:
    line: float
    side: str
    book: str
    american: float
    fair: FairLine
    consensus: Consensus
    ev: EvResult
    stake: Stake


class PropScanner:
    """Runs the whole funnel for a slate of player props."""

    def __init__(
        self,
        settings: Settings,
        odds: OddsProvider | None = None,
        logs: GameLogProvider | None = None,
        matchups: MatchupBook | None = None,
        model_a: BayesRateModel | None = None,
        model_b: BootstrapModel | None = None,
    ) -> None:
        self.settings = settings
        self.odds = odds or build_odds_provider(settings)
        self.logs = logs or LocalGameLogProvider(settings.data.gamelogs)
        self.matchups = matchups or MatchupBook(settings.data.defense)
        self.model_a = model_a or BayesRateModel(settings.models)
        self.model_b = model_b or BootstrapModel(settings.models)

    # ------------------------------------------------------------------
    def scan(
        self,
        sport: str,
        markets: Sequence[str] | None = None,
        players: Sequence[str] | None = None,
        bankroll: float | None = None,
        include_passes: bool = False,
        include_started: bool = False,
    ) -> list[Signal]:
        wanted = list(markets) if markets else market_keys(sport)
        props = self.odds.player_props(sport, wanted)
        if players:
            keep = {p.lower() for p in players}
            props = [p for p in props if p.player.lower() in keep]
        if not include_started:
            now = datetime.now(timezone.utc)
            props = [p for p in props if p.commence_time > now]

        roll = bankroll if bankroll is not None else self.settings.staking.bankroll
        signals = [self.evaluate(prop, roll) for prop in props]
        signals = self._apply_slate_cap(signals, roll)
        if not include_passes:
            signals = [s for s in signals if s.is_bet]
        return sort_signals(signals)

    # ------------------------------------------------------------------
    def evaluate(self, prop: PropMarket, bankroll: float) -> Signal:
        try:
            spec = get_spec(prop.market)
        except KeyError:
            return self._pass(prop, None, f"market {prop.market} not modelled")

        logs = self.logs.logs_for(prop.player, prop.sport)
        if not logs:
            return self._pass(prop, spec, f"no game logs for {prop.player}")

        context = self.matchups.context_for(
            sport=prop.sport,
            market=prop.market,
            opponent=prop.opponent,
            is_home=prop.is_home,
        )
        fit_a = self.model_a.fit(spec=spec, logs=logs, context=context)
        fit_b = self.model_b.fit(spec=spec, logs=logs, context=context)
        if not fit_a.ok or not fit_b.ok:
            broken = fit_a if not fit_a.ok else fit_b
            return self._pass(prop, spec, f"insufficient data: {broken.reason}")

        candidates: list[Candidate] = []
        rejections: list[str] = []
        for line in sorted({b.line for b in prop.books}):
            for side in ("over", "under"):
                price = best_price(prop, side, line)
                if price is None:
                    continue
                book, american = price
                fair = fair_prob_at(prop, line, self.settings.devig, exclude_book=book)
                if fair is None:
                    rejections.append(f"{side} {line:g}: no usable fair line")
                    continue
                consensus = evaluate_consensus(
                    fit_a.at(self.model_a.name, line),
                    fit_b.at(self.model_b.name, line),
                    side=side,
                    breakeven=1.0 / american_to_decimal(american),
                    fair_prob=fair.prob(side),
                    settings=self.settings.gate,
                )
                if not consensus.agree:
                    rejections.append(f"{side} {line:g}: {consensus.reason}")
                    continue
                candidate = self._price_candidate(
                    prop, spec, line, side, book, american, fair, consensus, bankroll
                )
                if isinstance(candidate, str):
                    rejections.append(f"{side} {line:g}: {candidate}")
                else:
                    candidates.append(candidate)

        if not candidates:
            reason = _best_reason(rejections) or "no two-way prices to work with"
            return self._pass(prop, spec, reason, fit_a=fit_a, fit_b=fit_b)

        best = max(candidates, key=lambda c: c.ev.ev_per_unit)
        return self._bet_signal(prop, spec, best, fit_a, fit_b)

    # ------------------------------------------------------------------
    def _price_candidate(
        self,
        prop: PropMarket,
        spec: MarketSpec,
        line: float,
        side: str,
        book: str,
        american: float,
        fair: FairLine,
        consensus: Consensus,
        bankroll: float,
    ) -> Candidate | str:
        cfg = self.settings.staking
        if american > cfg.max_american:
            return f"price {format_american(american)} longer than the {cfg.max_american:+.0f} cap"
        if american < cfg.min_american:
            return f"price {format_american(american)} shorter than the {cfg.min_american:+.0f} floor"

        ev = evaluate_ev(
            model_p_win=consensus.p_win,
            model_p_push=consensus.p_push,
            fair_prob=fair.prob(side),
            american_odds=american,
        )
        if ev.price_edge < cfg.min_price_edge:
            return (
                f"price edge {ev.price_edge * 100:+.1f} pts "
                f"(model {ev.model_prob_nopush:.1%} vs {ev.breakeven_prob / max(1 - ev.p_push, 1e-9):.1%} "
                f"breakeven) is under the {cfg.min_price_edge * 100:.1f} pt floor"
            )
        if ev.ev_per_unit < cfg.min_ev:
            return f"EV {ev.ev_pct:+.1f}% is under the {cfg.min_ev * 100:.1f}% floor"

        stake = size_bet(
            p_win=consensus.p_win,
            p_push=consensus.p_push,
            decimal_odds=ev.decimal_odds,
            bankroll=bankroll,
            kelly_fraction=cfg.kelly_fraction,
            max_bet_fraction=cfg.max_bet_fraction,
            min_stake=cfg.min_stake,
            rounding=cfg.rounding,
        )
        if stake.amount <= 0:
            return f"stake rounds to zero (min {cfg.min_stake:g})"
        return Candidate(line, side, book, american, fair, consensus, ev, stake)

    # ------------------------------------------------------------------
    def _bet_signal(
        self,
        prop: PropMarket,
        spec: MarketSpec,
        c: Candidate,
        fit_a: Fit,
        fit_b: Fit,
    ) -> Signal:
        projection = 0.5 * (fit_a.dist.mean + fit_b.dist.mean)  # type: ignore[union-attr]
        projection_sd = 0.5 * (fit_a.dist.sd + fit_b.dist.sd)  # type: ignore[union-attr]
        warnings = list(c.fair.warnings)
        if c.fair.interpolated:
            warnings.append(f"fair line interpolated to {c.line:g}")
        if c.stake.capped_by == "max_bet":
            warnings.append(f"stake capped at {self.settings.staking.max_bet_fraction:.1%} of bankroll")
        return Signal(
            action="BET",
            sport=prop.sport,
            event_id=prop.event_id,
            commence_time=prop.commence_time,
            matchup=f"{prop.away_team} @ {prop.home_team}",
            player=prop.player,
            market=prop.market,
            market_label=spec.label,
            reason=c.consensus.reason,
            side=c.side,
            line=c.line,
            book=c.book,
            american=c.american,
            decimal=c.ev.decimal_odds,
            fair_prob=c.ev.fair_prob_nopush,
            fair_american=prob_to_american(c.ev.fair_prob_nopush),
            model_prob=c.ev.model_prob_nopush,
            p_push=c.ev.p_push,
            edge=c.ev.price_edge,
            market_edge=c.ev.market_edge,
            ev_per_unit=c.ev.ev_per_unit,
            stake=c.stake.amount,
            stake_units=c.stake.units,
            full_kelly=c.stake.full_kelly,
            model_probs=c.consensus.model_probs,
            disagreement=c.consensus.disagreement,
            projection=projection,
            projection_sd=projection_sd,
            n_books=c.fair.n_books,
            devig_method=c.fair.method,
            warnings=tuple(warnings),
        )

    def _pass(
        self,
        prop: PropMarket,
        spec: MarketSpec | None,
        reason: str,
        fit_a: Fit | None = None,
        fit_b: Fit | None = None,
    ) -> Signal:
        projection = projection_sd = None
        if fit_a and fit_b and fit_a.dist and fit_b.dist:
            projection = 0.5 * (fit_a.dist.mean + fit_b.dist.mean)
            projection_sd = 0.5 * (fit_a.dist.sd + fit_b.dist.sd)
        return Signal(
            action="PASS",
            sport=prop.sport,
            event_id=prop.event_id,
            commence_time=prop.commence_time,
            matchup=f"{prop.away_team} @ {prop.home_team}",
            player=prop.player,
            market=prop.market,
            market_label=spec.label if spec else prop.market,
            reason=reason,
            line=prop.books[0].line if prop.books else None,
            projection=projection,
            projection_sd=projection_sd,
            devig_method=self.settings.devig.method,
        )

    # ------------------------------------------------------------------
    def _apply_slate_cap(self, signals: list[Signal], bankroll: float) -> list[Signal]:
        """Keep the whole slate's risk inside the configured ceiling."""
        bets = [s for s in signals if s.is_bet]
        if not bets:
            return signals
        stakes = [s.stake for s in bets]
        scaled = cap_total_exposure(stakes, bankroll, self.settings.staking.max_slate_fraction)
        if all(abs(a - b) < 1e-9 for a, b in zip(stakes, scaled)):
            return signals

        from dataclasses import replace

        adjusted = {
            id(s): replace(
                s,
                stake=round(new, 2),
                stake_units=100.0 * new / bankroll,
                warnings=s.warnings + (f"scaled down to respect the {self.settings.staking.max_slate_fraction:.0%} slate cap",),
            )
            for s, new in zip(bets, scaled)
        }
        return [adjusted.get(id(s), s) for s in signals]


def _best_reason(rejections: Sequence[str]) -> str | None:
    """Pick the most informative rejection to show the user.

    A gate rejection ("the models disagree") explains more than a plumbing one
    ("no usable fair line"), so it wins when both are present.
    """
    if not rejections:
        return None
    def rank(reason: str) -> int:
        if "disagree" in reason or "whole market" in reason:
            return 0
        if "clears the price" in reason or "price edge" in reason or "EV" in reason:
            return 1
        return 2

    return sorted(rejections, key=rank)[0]
