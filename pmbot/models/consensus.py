"""The agreement gate: two models have to say the same thing, or we pass.

A single model that likes a prop is an opinion. Two models built on different
assumptions that independently call the same price +EV is weak evidence that
the edge is in the data rather than in one model's quirks. When they disagree,
the honest output is "pass" -- not "average them and bet anyway", which is how
you end up betting the midpoint of two wrong answers.

The gate measures each model against the **price**, not against the market
consensus. That distinction matters more than it looks: the most reliable edge
in props is not "my model disagrees with the market", it's "four books say
-115 and this one says +135". Gating on disagreement-with-consensus throws
those away and keeps only the bets where the model is out on a limb alone.

The consensus fair probability still does three jobs:

* the models get shrunk toward it, because the market is the best-informed
  forecaster in the room and the models are not;
* the distance from it is reported, so a bet that looks good only because the
  model is an outlier is visible as such;
* if the models diverge from the whole market by more than
  ``max_market_divergence``, that is treated as a data problem rather than an
  edge, and the prop is passed. When a model says 68% and five books say 45%,
  the model is nearly always looking at a stale injury report.

Be clear about what agreement does and doesn't buy: the models share the same
inputs, so their errors are correlated. Agreement is a robustness check
against modelling choices, not two independent samples of the truth.
"""

from __future__ import annotations

import math

from ..config import GateSettings
from ..types import Consensus, ModelOutput

PASS_INSUFFICIENT = "insufficient data"


def evaluate_consensus(
    out_a: ModelOutput,
    out_b: ModelOutput,
    *,
    side: str,
    breakeven: float,
    fair_prob: float,
    settings: GateSettings,
) -> Consensus:
    """Decide whether both models back ``side`` at this price.

    ``breakeven`` is the win probability the price needs (1 / decimal odds,
    excluding pushes). ``fair_prob`` is the market's de-vigged probability of
    the same side, computed from the *other* books.
    """
    if not out_a.ok or not out_b.ok:
        broken = out_a if not out_a.ok else out_b
        return Consensus(
            agree=False,
            side=None,
            p_win=0.0,
            p_push=0.0,
            p_over_nopush=0.0,
            reason=f"{PASS_INSUFFICIENT}: {broken.model} -- {broken.reason}",
        )

    assert out_a.probs is not None and out_b.probs is not None
    p_a = out_a.probs.side_nopush(side)
    p_b = out_b.probs.side_nopush(side)
    probs = {out_a.model: p_a, out_b.model: p_b}
    disagreement = abs(p_a - p_b)
    p_push = 0.5 * (out_a.probs.p_push + out_b.probs.p_push)

    def result(agree: bool, p_side: float, reason: str) -> Consensus:
        return Consensus(
            agree=agree,
            side=side if agree else None,
            p_win=p_side * (1.0 - p_push),
            p_push=p_push,
            p_over_nopush=p_side if side == "over" else 1.0 - p_side,
            reason=reason,
            model_probs=probs,
            disagreement=disagreement,
        )

    if disagreement > settings.max_disagreement:
        return result(
            False, 0.5 * (p_a + p_b),
            f"models disagree by {disagreement * 100:.1f} pts "
            f"({out_a.model} {p_a:.1%} vs {out_b.model} {p_b:.1%}, "
            f"limit {settings.max_disagreement * 100:.1f})",
        )

    edge_a, edge_b = p_a - breakeven, p_b - breakeven
    if min(edge_a, edge_b) < settings.min_model_edge:
        weaker = out_a.model if edge_a < edge_b else out_b.model
        return result(
            False, min(p_a, p_b),
            f"{weaker} clears the price by only {min(edge_a, edge_b) * 100:+.1f} pts "
            f"(needs {settings.min_model_edge * 100:.1f})",
        )

    combined = _combine(p_a, p_b, settings.combine)
    divergence = combined - fair_prob
    if abs(divergence) > settings.max_market_divergence:
        return result(
            False, combined,
            f"models are {divergence * 100:+.1f} pts away from the whole market "
            f"({combined:.1%} vs fair {fair_prob:.1%}) -- that is usually stale data, not edge",
        )

    blended = (1.0 - settings.market_blend) * combined + settings.market_blend * fair_prob
    return result(
        True, blended,
        f"both models on {side} ({out_a.model} {p_a:.1%}, {out_b.model} {p_b:.1%}) "
        f"vs {breakeven:.1%} breakeven; fair {fair_prob:.1%}",
    )


def _combine(a: float, b: float, method: str) -> float:
    if method == "min":
        return min(a, b)
    if method == "mean":
        return 0.5 * (a + b)
    if method == "geometric":
        return math.sqrt(max(a, 1e-9) * max(b, 1e-9))
    raise ValueError(f"unknown combine method {method!r}; expected min|mean|geometric")
