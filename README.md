# pmbot

A player-prop betting bot that is mostly built to say **no**.

Two independent models have to agree before it opens its mouth. The price has
to beat the market's *de-vigged* number, not its posted one. The stake comes
from fractional Kelly against your real bankroll. Every bet lands in a journal
that tracks ROI, win rate and closing line value, because a strategy you can't
measure is a hobby.

Covers NBA, MLB and NFL player props. Standard library only — no dependencies.

```
$ pmbot --sample scan --sport nfl --bankroll 5000

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
pip install -e .

pmbot --sample scan --sport nba --bankroll 5000 --all   # --all shows passes too
```

`--sample` runs against the synthetic slate that ships with the repo. For real
work, pull real game logs (see [Real game logs](#real-game-logs)) and point it
at a live odds feed:

```bash
pmbot fetch --sport nfl --player "Josh Allen"
export PMBOT_ODDS_API_KEY=...
pmbot scan --sport nfl --bankroll 5000
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

### Real game logs

`pmbot fetch` pulls actual box scores and writes them where the models look.
One source per sport, because no single free feed covers all three well:

| sport | source | notes |
|---|---|---|
| NFL | [nflverse](https://github.com/nflverse/nflverse-data) | weekly player stats as CSV release assets, joined to the schedule for real dates and venue. No key, no scraping. |
| NBA | [hoopR](https://github.com/sportsdataverse/sportsdataverse-data) *(default)* | ESPN box scores, one tidy CSV per season, refreshed daily. Reachable from anywhere. |
| NBA | [stats.nba.com](https://stats.nba.com) | `--source nba-stats`. Official and complete, but rate-limits hard and refuses many networks outright, which is why it isn't the default. |
| MLB | [MLB Stats API](https://statsapi.mlb.com) | official and keyless. Hitting and pitching are fetched separately and merged per game, so two-way players come out whole. |

All three are free and need no account.

```bash
pmbot sources                                          # what's available
pmbot fetch --sport nfl --player "Josh Allen" --player "CeeDee Lamb"
pmbot fetch --sport nba --player "Nikola Jokic" --seasons 2024,2025
pmbot fetch --sport mlb --player "Aaron Judge"
pmbot fetch --sport nba --from-props                   # everyone on tonight's board
```

Seasons are named for the year they **start**, so `--seasons 2024` is the
2024-25 NBA season. (hoopR files are named for the year a season ends; the
conversion happens inside the source so you never see it.)

Then check a projection before trusting it:

```
$ pmbot project --sport nfl --player "Patrick Mahomes" --market player_pass_yds --line 249.5

  bayes_rate     over  46.0%  (fair   +117)  projection 242.74 +/- 68.05  n=39
  bootstrap_mc   over  44.7%  (fair   +124)  projection 243.94 +/- 75.31  n=39
```

Details worth knowing:

* **Fetches merge, and never blend feeds.** New games are added, corrected
  stat lines overwrite old ones, and history survives a source going down.
  But a file written by one source is *replaced* rather than merged when
  another source writes it — two feeds disagree about field definitions and
  date conventions, and the blend looks fine while modelling nothing.
* **Everything is cached** with a TTL, so re-running a fetch is free and a
  season file is downloaded once a day at most. `--offline` uses only the cache.
* **Ambiguous names stop rather than guess.** There are two Josh Allens; the
  fetch says so and asks for `--team BUF`. Accents and `Jr.` are folded, so
  `Jokić`/`Jokic` and `Witt Jr.`/`Witt` match either way.
* **DNPs are dropped, and nothing else is.** Getting this wrong is quiet and
  expensive: ESPN's `active` column is a roster flag rather than an appearance
  flag, and filtering on it deleted 40% of Anthony Edwards' season while the
  resulting logs still looked entirely plausible.
* **Two seasons by default** — enough that an early-season slate has a sample
  behind it without dragging in ancient form. Change with `--seasons`.

Adding a source means implementing `search()` and `game_logs()` from
[`ingest/base.py`](pmbot/ingest/base.py) and registering it in
[`ingest/__init__.py`](pmbot/ingest/__init__.py); the mapping onto pmbot's
box-score fields lives in the source, so nothing downstream changes.

### Real data vs the shipped fixture

Two directories, deliberately kept apart:

* `data/gamelogs/`, `data/props/` — **real**. Written by `pmbot fetch` and
  `pmbot snapshot`. Git-ignored: your data, not the repo's.
* `data/sample/` — **synthetic**, generated from a fixed seed by
  [`scripts/make_sample_data.py`](scripts/make_sample_data.py) and checked in
  so the demo and the tests run offline. Reach it with the `--sample` flag.

Mixing them produces enormous imaginary edges — real logs priced against
invented lines. `pmbot fetch` refuses to write into `data/sample`, and the
store refuses to merge across sources, because both mistakes were made while
building this and neither is visible in the output.

Matchup adjustments live in `data/defense.json` (opponent, pace and venue
multipliers; `1.0` means "no information", which is what ships). `pmbot
markets` lists every supported market and the workload it's modelled against.

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
  ingest/         real game logs: nflverse, hoopR, MLB Stats API, stats.nba.com
  cli.py          the command line
```

Run the tests with `pytest` (338 of them, no network, ~7s). Two opt-in live
checks hit the real nflverse and hoopR feeds: `PMBOT_LIVE_TESTS=1 pytest -k live`.

---

## What this is not

Read this part.

* **It is not an edge in itself.** It's a disciplined framework for finding and
  sizing one. Box scores don't know about a minutes restriction, a blowout, a
  bullpen game, 20mph wind, or a coach who told a beat reporter something at
  4pm. Those are where prop edges actually live, and none of them are in this
  repo — `pmbot fetch` gets you real history, not real information.
* **The shipped fixture is fake.** `data/sample/` exists so the pipeline can
  be tested and demonstrated end to end. Backtest results on it mean nothing.
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
