# pmbot

A player-prop betting bot that is mostly built to say **no**.

Two independent models have to agree before it opens its mouth. The price has
to beat the market's *de-vigged* number, not its posted one. The stake comes
from fractional Kelly against your real bankroll. Every bet lands in a journal
that tracks ROI, win rate and closing line value, because a strategy you can't
measure is a hobby.

Covers NBA, MLB and NFL player props. Standard library only — no dependencies.

```
$ pmbot scan --sport nfl --bankroll 5000

NFL player props │ bankroll 5,000.00 │ 0.25x Kelly │ devig=power

BET   Josh Allen             Passing yards          OVER 236.5  +150 @ draftkings
      │ stake 89.00 (1.78u)  EV +10.8%  edge +4.3%
      │ model 44.3% vs fair 44.1% (+127) from 4 book(s), devig=power
      │ both models on over (bayes_rate 44.4%, bootstrap_mc 44.5%) vs 40.0% breakeven; fair 44.1%
      │ projection 230.4 +/- 48.8  full Kelly 7.2%  BUF @ MIA

PASS  CeeDee Lamb            Receptions
      │ over 6.5: models disagree by 8.3 pts (bayes_rate 51.5% vs bootstrap_mc 43.2%, limit 6.0)

1 bet(s) of 4 prop(s) scanned │ total risk 89.00 (1.8% of bankroll)
```

---

## Quick start

No API key needed — the repo ships with a synthetic slate so everything runs
offline.

```bash
git clone <this repo> && cd prediction-market-bot
python3 -m pmbot.cli scan --sport nba --bankroll 5000 --all   # --all shows passes too

# or install it properly
pip install -e .
pmbot scan --sport mlb --bankroll 5000
```

Then run the whole loop:

```bash
pmbot bankroll --deposit 5000
pmbot scan --sport nba --log                  # log the recommendations it made
pmbot bets                                    # see what's open
pmbot settle 1 --result 27                    # he scored 27
pmbot close 1 --over -130 --under 110         # what the market closed at
pmbot report --group market                   # ROI, win rate, CLV, calibration
```

---

## How it works

### 1. Two models that have to agree

Both models read the same game logs and produce a full outcome distribution,
but they are built on deliberately different assumptions, so agreement means
something more than "the same code ran twice".

**Model A — `bayes_rate`** ([`models/model_a.py`](pmbot/models/model_a.py)).
Production is a *rate* per unit of opportunity (points per minute, strikeouts
per batter faced, yards per target) times how much opportunity there'll be.
Rate is slow-moving skill; workload is a fast-moving role decision, so they're
estimated separately with different memory. Shrinkage runs in two stages —
career rate pulled weakly toward a league prior, recent form pulled back toward
that career rate — which damps a hot week without dragging a 28-point scorer
toward a league average they've never been near. The outcome distribution is
parametric: negative binomial for counts, beta-binomial for catch-rate style
markets, Gaussian on the integer lattice for yardage.

**Model B — `bootstrap_mc`** ([`models/model_b.py`](pmbot/models/model_b.py)).
No distributional family at all. It resamples the player's own games 20,000
times, so a boom-bust receiver keeps their real lumpy shape. Workload comes
from a robust weighted *median*, games are weighted by how closely their role
resembles tonight's expected role, and the matchup adjustment is drawn per
simulation rather than applied as a point estimate.

**The gate** ([`models/consensus.py`](pmbot/models/consensus.py)) demands:

* both models on the same side of the **price**, each beating its breakeven by
  at least `min_model_edge`;
* the two probabilities within `max_disagreement` of each other;
* the consensus within `max_market_divergence` of the whole market — because a
  model that's 20 points away from five books is reading stale data far more
  often than it's finding an edge.

Then it takes the **more pessimistic** of the two and shrinks it 25% toward the
market's fair number. Anything else prints a pass with the reason.

> **Honest caveat.** The models share inputs, so their errors are correlated.
> Agreement is a robustness check against modelling choices, not two
> independent samples of the truth. It's a filter that removes bets built on
> one model's quirks — nothing stronger than that.

### 2. Vig removal, done per book and per line

Comparing a model to a posted price is how people talk themselves into bets
that were never there: at -110/-110 the book's implied probabilities sum to
1.048, and that 4.8% is exactly what you're trying to beat.

[`oddsmath.py`](pmbot/oddsmath.py) implements five de-vig methods —
multiplicative, additive, **power** (the default for two-way props), Shin, and
a deliberately pessimistic "conservative" mode. Each book's two-way pair is
de-vigged on its own, then [`fairline.py`](pmbot/fairline.py) builds the
consensus:

* books quoting a *different* line still count — the fair probability is
  fitted in log-odds space against the line and evaluated at the line you're
  actually betting;
* sharp books carry more weight than square ones;
* **the book you're betting is excluded from its own fair line.** If Caesars is
  hanging +165 into a -115 market, letting Caesars vote on what's fair drags
  the number toward the outlier and hides the edge you found.

### 3. Value means price edge, not disagreement

There are two different "edges", and conflating them costs you the best bets:

| | what it asks | what it's for |
|---|---|---|
| **price edge** | does the model beat what this price needs? | drives EV — this is the one with a floor |
| **market edge** | does the model disagree with the consensus? | how far out on a limb you are — reported, not required |

The Josh Allen bet above has a *market* edge of +0.2 points — the model agrees
with the market almost exactly — and it's still the best bet on the board,
because one book is 6 points behind the other four. A gate that required
disagreement with the consensus would throw that away and keep only the bets
where the model is alone. Everything is push-aware: a whole-number line
refunds instead of losing, and that's worth real basis points.

### 4. Kelly staking with the brakes on

[`kelly.py`](pmbot/kelly.py). Full Kelly maximises long-run growth *if your
probabilities are exactly right*. They aren't. So:

* **quarter Kelly** by default (`kelly_fraction`);
* **2% of bankroll** ceiling on any single bet (`max_bet_fraction`);
* **10% of bankroll** ceiling on a whole slate (`max_slate_fraction`) — Kelly
  sizes bets as though they settle one at a time, and a twelve-leg Tuesday
  does not;
* push-aware closed form, `f* = (p·b − q) / (b·(p + q))`, verified against a
  brute-force search of the growth-rate curve in the tests.

### 5. The journal is the point

[`journal/`](pmbot/journal). SQLite, one file. It stores not just what you bet
but *why* — model probability, fair probability, edge and EV frozen at
placement — so that later you can tell a bad process from bad luck.

`pmbot report` gives you:

* **ROI, win rate vs breakeven, units, max drawdown**, with a bootstrap
  confidence interval and a t-statistic, because 40 bets is not a verdict;
* **CLV**, measured against the *de-vigged* close. Note the bar: taking -110 on
  a market that closes -110/-110 scores −4.5%, not zero. You paid the hold and
  the line never moved. Above zero means you were still ahead of a vig-free
  market at the close, which is the only version of "beat the close" that pays.
  (`clv_raw_pct` keeps the familiar line-movement number too.);
* **model vs actual** — total EV the model claimed against what actually
  landed, so you can see how much was skill and how much was running good;
* **calibration** — bucket bets by model probability and compare predicted to
  actual. If the 60% bucket hits 52%, the edge was never there.

Bankroll is always *derived* (`deposits + settled profit`, minus pending stakes
for what's available), never a mutable balance that can drift out of sync.

---

## Live odds

The bot reads [The Odds API](https://the-odds-api.com/) v4. Player props live
on the per-event endpoint, so a slate costs one call per game — responses are
cached on disk with a short TTL.

```bash
export PMBOT_ODDS_API_KEY=your_key_here      # switches the provider on by itself
pmbot scan --sport nba --bankroll 5000
```

Snapshots are worth taking right before lock — they replay offline *and*
become the closing-line record CLV needs:

```bash
pmbot snapshot --sport nba --out data/props/nba_close.json
pmbot close-all --sport nba        # match the close against open bets
```

To plug in a different feed, implement `player_props()` from
[`providers/base.py`](pmbot/providers/base.py); nothing downstream cares.

### Your own data

Everything under `data/` is synthetic, generated by
[`scripts/make_sample_data.py`](scripts/make_sample_data.py) from a fixed seed.
Swap in real numbers by matching the shapes:

* `data/gamelogs/<sport>/<player-slug>.json` — one object per game with the
  raw box-score fields (`minutes`, `points`, `plate_appearances`, `targets`, …)
* `data/defense.json` — opponent, pace and venue multipliers; `1.0` means "no
  information", which is the right default
* `data/props/<sport>_*.json` — an odds snapshot, if you're replaying offline

`pmbot markets` lists every supported market and the workload each is modelled
against.

---

## Configuration

`pmbot config` prints the effective settings; `pmbot config --write` saves them
to `pmbot.config.json` for editing. Every default is conservative, and each one
notes which direction is dangerous. The ones that matter:

| setting | default | what it does |
|---|---|---|
| `devig.method` | `power` | vig removal; `shin` is a good alternative on big favourites |
| `devig.exclude_own_book` | `true` | keep an outlier out of its own fair line |
| `gate.max_disagreement` | `0.06` | how far apart the two models may be |
| `gate.min_model_edge` | `0.015` | each model must beat the price by this alone |
| `gate.max_market_divergence` | `0.15` | past this, assume stale data, not edge |
| `gate.market_blend` | `0.25` | shrink the consensus toward the market |
| `staking.kelly_fraction` | `0.25` | quarter Kelly |
| `staking.max_bet_fraction` | `0.02` | single-bet ceiling |
| `staking.min_price_edge` | `0.02` | 2 points of probability over breakeven |

Env overrides: `PMBOT_ODDS_API_KEY`, `PMBOT_BANKROLL`, `PMBOT_KELLY_FRACTION`,
`PMBOT_DEVIG`, `PMBOT_DB`, `PMBOT_PROVIDER`, `PMBOT_CONFIG`.

---

## Layout

```
pmbot/
  oddsmath.py     odds conversions, five de-vig methods
  fairline.py     per-book de-vig -> weighted consensus at any line
  distributions.py Poisson / negative binomial / beta-binomial / lattice normal / empirical
  markets.py      supported props: workload, family, league priors
  models/         model_a (Bayesian rate), model_b (bootstrap MC), consensus gate
  ev.py           price edge vs market edge, push-aware EV
  kelly.py        fractional Kelly, per-bet and per-slate caps
  pipeline.py     the scan: odds in, bets or explained passes out
  journal/        SQLite storage, settlement, ROI / CLV / calibration
  providers/      The Odds API client, file-backed providers
  cli.py          the command line
```

Run the tests with `pytest` (259 of them, no network, ~7s).

---

## What this is not

Read this part.

* **It is not an edge in itself.** It's a disciplined framework for finding and
  sizing one. Game logs alone don't know about a minutes restriction, a
  blowout, a bullpen game, 20mph wind, or a coach who told a beat reporter
  something at 4pm. Those are where prop edges actually live, and none of them
  are in this repo.
* **The shipped data is fake.** It's built so the pipeline can be tested and
  demonstrated end to end. Backtest results on it mean nothing.
* **Prop markets hold 5–8%.** Being right isn't enough; you have to be right by
  more than that, repeatedly.
* **Books limit winners.** Sustained CLV on props gets stake limits or an
  account closure long before it gets rich.
* **Two agreeing models are still two models.** See the caveat above.
* **CLV is the scoreboard, not profit.** Judge the process on closing line
  value over hundreds of bets; ROI over dozens is noise wearing a number.

Bet only money you can afford to lose, and check the law where you live. If
gambling has stopped being fun, in the US: 1-800-GAMBLER.

MIT licensed.
