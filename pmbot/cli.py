"""Command line interface.

    pmbot scan --sport nba --bankroll 5000
    pmbot project --sport nba --player "Anthony Edwards" --market player_points --line 27.5
    pmbot log --sport nba --player "..." --market player_points --side over --line 27.5 \
              --book caesars --odds +140 --stake 62
    pmbot settle 3 --result 31
    pmbot close 3 --over -108 --under -112
    pmbot report --group market
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Sequence

from .config import Settings
from .journal import Journal, bankroll_curve, breakdown, calibration, max_drawdown, summarise
from .journal.metrics import Performance
from .markets import SPORTS, get_spec, market_keys
from .models.consensus import evaluate_consensus
from .models.model_a import BayesRateModel
from .models.model_b import BootstrapModel
from .oddsmath import american_to_decimal, format_american, prob_to_american
from .pipeline import PropScanner
from .providers import LocalGameLogProvider, MatchupBook, build_odds_provider
from .types import Signal

DIM = "│"


# ==========================================================================
# rendering
# ==========================================================================
def money(x: float) -> str:
    return f"{x:,.2f}"


def pct(x: float | None, digits: int = 1, signed: bool = False) -> str:
    if x is None:
        return "--"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def render_signal(sig: Signal, verbose: bool = False) -> str:
    if not sig.is_bet:
        head = f"PASS  {sig.player:<22} {sig.market_label:<22}"
        body = f"      {DIM} {sig.reason}"
        if verbose and sig.projection is not None:
            body += f"\n      {DIM} projection {sig.projection:.1f} +/- {sig.projection_sd:.1f}"
        return f"{head}\n{body}"

    assert sig.side and sig.line is not None and sig.american is not None
    lines = [
        f"BET   {sig.player:<22} {sig.market_label:<22} "
        f"{sig.side.upper()} {sig.line:g}  {format_american(sig.american)} @ {sig.book}",
        f"      {DIM} stake {money(sig.stake)} ({sig.stake_units:.2f}u)  "
        f"EV {pct(sig.ev_per_unit, signed=True)}  edge {pct(sig.edge, signed=True)}",
        f"      {DIM} model {pct(sig.model_prob)} vs fair {pct(sig.fair_prob)} "
        f"({format_american(sig.fair_american)}) from {sig.n_books} book(s), devig={sig.devig_method}",
        f"      {DIM} {sig.reason}",
    ]
    if sig.p_push > 0.005:
        lines.append(f"      {DIM} push probability {pct(sig.p_push)} (whole-number line)")
    if verbose:
        lines.append(
            f"      {DIM} projection {sig.projection:.1f} +/- {sig.projection_sd:.1f}  "
            f"full Kelly {pct(sig.full_kelly)}  {sig.matchup}  {sig.commence_time:%b %d %H:%M UTC}"
        )
    for warning in sig.warnings:
        lines.append(f"      ! {warning}")
    return "\n".join(lines)


def render_performance(perf: Performance, title: str = "PERFORMANCE") -> str:
    rows = [
        f"{title} ({perf.n_settled} settled, {perf.n_pending} pending)",
        f"  record            {perf.wins}-{perf.losses}"
        + (f"-{perf.pushes}" if perf.pushes else "")
        + (f"  ({perf.voids} void)" if perf.voids else ""),
        f"  staked            {money(perf.staked):>12}     ROI      {pct(perf.roi, signed=True)}",
        f"  profit            {money(perf.profit):>12}     units    {perf.units_profit:+.2f}u",
        f"  win rate          {pct(perf.win_rate):>12}     breakeven {pct(perf.breakeven_rate)}",
        f"  avg price         {format_american(perf.avg_american):>12}     avg stake {money(perf.avg_stake)}",
    ]
    if perf.clv_n:
        rows.append(
            f"  CLV               {pct(perf.clv_avg, 2, signed=True):>12}     "
            f"beat close {pct(perf.beat_close_rate)} of {perf.clv_n}"
        )
    if perf.expected_profit is not None:
        rows.append(
            f"  model expected    {money(perf.expected_profit):>12}     luck     {money(perf.luck or 0.0)}"
        )
    if perf.roi_ci:
        lo, hi = perf.roi_ci
        t = f"t = {perf.t_stat:+.2f}" if perf.t_stat is not None else ""
        rows.append(f"  95% CI on ROI     [{pct(lo, signed=True)}, {pct(hi, signed=True)}]   {t}")
    return "\n".join(rows)


def render_breakdown(rows: Sequence[Performance], key: str) -> str:
    if not rows:
        return f"(no settled bets to break down by {key})"
    out = [f"BY {key.upper():<12} {'bets':>5} {'record':>10} {'staked':>10} {'profit':>10} {'ROI':>8} {'CLV':>8}"]
    for perf in rows:
        record = f"{perf.wins}-{perf.losses}" + (f"-{perf.pushes}" if perf.pushes else "")
        out.append(
            f"  {perf.label:<21} {perf.n_bets:>5} {record:>10} "
            f"{money(perf.staked):>10} {money(perf.profit):>10} "
            f"{pct(perf.roi, signed=True):>8} {pct(perf.clv_avg, 2, signed=True) if perf.clv_n else '--':>8}"
        )
    return "\n".join(out)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=_json_default))


# ==========================================================================
# commands
# ==========================================================================
def cmd_scan(args: argparse.Namespace, settings: Settings) -> int:
    if args.bankroll is not None:
        settings.staking.bankroll = args.bankroll
    scanner = PropScanner(settings)
    bankroll = settings.staking.bankroll

    signals = scanner.scan(
        sport=args.sport,
        markets=args.market or None,
        players=args.player or None,
        bankroll=bankroll,
        include_passes=args.all,
        include_started=args.include_started,
    )
    if args.json:
        emit_json([asdict(s) for s in signals])
        return 0

    bets = [s for s in signals if s.is_bet]
    print(
        f"{args.sport.upper()} player props {DIM} bankroll {money(bankroll)} {DIM} "
        f"{settings.staking.kelly_fraction:g}x Kelly {DIM} devig={settings.devig.method}\n"
    )
    if not signals:
        print("Nothing to scan: no upcoming props matched those filters.")
        return 0
    for sig in signals:
        print(render_signal(sig, verbose=args.verbose))
        print()

    exposure = sum(s.stake for s in bets)
    print(
        f"{len(bets)} bet(s) of {len(signals)} prop(s) scanned {DIM} "
        f"total risk {money(exposure)} ({exposure / bankroll:.1%} of bankroll)"
    )
    if args.log and bets:
        with Journal(settings.data.db) as journal:
            ids = [journal.log_signal(s, bankroll=bankroll) for s in bets]
        print(f"logged to the journal as bet(s) {', '.join(str(i) for i in ids)}")
    return 0


def cmd_project(args: argparse.Namespace, settings: Settings) -> int:
    """Show both models side by side for one prop -- the transparency view."""
    spec = get_spec(args.market)
    logs = LocalGameLogProvider(settings.data.gamelogs).logs_for(args.player, args.sport)
    if not logs:
        print(f"no game logs for {args.player} under {settings.data.gamelogs}/{args.sport}/")
        return 1
    context = MatchupBook(settings.data.defense).context_for(
        sport=args.sport,
        market=spec.key,
        opponent=args.opponent,
        is_home=None if args.home is None else args.home,
        workload_override=args.workload,
    )
    model_a, model_b = BayesRateModel(settings.models), BootstrapModel(settings.models)
    out_a = model_a.predict(spec=spec, logs=logs, line=args.line, context=context)
    out_b = model_b.predict(spec=spec, logs=logs, line=args.line, context=context)

    print(f"{args.player} {DIM} {spec.label} {DIM} line {args.line:g}")
    if context.notes:
        print(f"  matchup: {', '.join(context.notes)} (combined x{context.combined:.3f})")
    print()
    for out in (out_a, out_b):
        if not out.ok:
            print(f"  {out.model:<14} unavailable: {out.reason}")
            continue
        assert out.probs
        print(
            f"  {out.model:<14} over {out.probs.p_over_nopush:6.1%}  "
            f"(fair {format_american(prob_to_american(out.probs.p_over_nopush)):>6})  "
            f"projection {out.mean:6.2f} +/- {out.sd:5.2f}  n={out.n_games}"
        )
        if out.probs.p_push > 0.005:
            print(f"  {'':<14} push {out.probs.p_push:.1%}")
        if args.verbose:
            for key, value in sorted(out.details.items()):
                shown = f"{value:.4g}" if isinstance(value, (int, float)) else value
                print(f"  {'':<14}   {key}: {shown}")

    if out_a.ok and out_b.ok:
        model_over = 0.5 * (out_a.probs.p_over_nopush + out_b.probs.p_over_nopush)  # type: ignore[union-attr]
        fair_over = args.market_prob
        notes = []
        if fair_over is None:
            fair_over = model_over
            notes.append("no market price given, so the gate is shown against the models themselves")
        else:
            notes.append(f"vs supplied fair over probability {fair_over:.1%}")

        side = "over" if model_over >= fair_over else "under"
        fair_side = fair_over if side == "over" else 1.0 - fair_over
        if args.odds is not None:
            breakeven = 1.0 / american_to_decimal(args.odds)
            notes.append(f"priced at {format_american(args.odds)} ({breakeven:.1%} breakeven)")
        else:
            breakeven = fair_side
            notes.append("no odds given, so breakeven is the fair price itself")

        consensus = evaluate_consensus(
            out_a, out_b, side=side, breakeven=breakeven,
            fair_prob=fair_side, settings=settings.gate,
        )
        print(f"\n  gate ({side}): {'AGREE' if consensus.agree else 'PASS'} {DIM} {consensus.reason}")
        for note in notes:
            print(f"  {DIM} {note}")
    return 0


def cmd_markets(args: argparse.Namespace, settings: Settings) -> int:
    for sport in [args.sport] if args.sport else SPORTS:
        print(f"{sport.upper()}")
        for key in market_keys(sport):
            spec = get_spec(key)
            per = f"per {spec.workload}" if spec.workload else "per game"
            print(f"  {key:<34} {spec.label:<22} {spec.family:<9} {per}")
        print()
    return 0


def cmd_bankroll(args: argparse.Namespace, settings: Settings) -> int:
    with Journal(settings.data.db) as journal:
        for amount, kind, verb in (
            (args.deposit, "deposit", "deposited"),
            (args.withdraw, "withdrawal", "withdrew"),
            (args.adjust, "adjustment", "adjusted by"),
        ):
            if not amount:
                continue
            journal.add_funds(amount, kind, args.note)
            if not args.json:
                print(f"{verb} {money(amount)}")

        balance, pending = journal.balance(), journal.pending_stake()
        if args.json:
            emit_json(
                {
                    "deposits": journal.deposits(),
                    "realised_profit": journal.realised_profit(),
                    "balance": balance,
                    "pending_stake": pending,
                    "available": journal.available(),
                }
            )
            return 0
        print("BANKROLL")
        print(f"  deposits          {money(journal.deposits()):>12}")
        print(f"  realised P/L      {money(journal.realised_profit()):>12}")
        print(f"  balance           {money(balance):>12}")
        print(f"  pending stakes    {money(pending):>12}")
        print(f"  available         {money(journal.available()):>12}")
    return 0


def cmd_log(args: argparse.Namespace, settings: Settings) -> int:
    with Journal(settings.data.db) as journal:
        bet_id = journal.log_bet(
            sport=args.sport,
            player=args.player,
            market=args.market,
            side=args.side,
            line=args.line,
            book=args.book,
            american=args.odds,
            stake=args.stake,
            model_prob=args.model_prob,
            fair_prob=args.fair_prob,
            edge=(args.model_prob - args.fair_prob)
            if args.model_prob is not None and args.fair_prob is not None
            else None,
            tags=args.tag,
            notes=args.note,
        )
        bet = journal.get(bet_id)
        assert bet
        print(
            f"logged #{bet_id}: {bet.player} {bet.market} {bet.side} {bet.line:g} "
            f"{format_american(bet.american)} @ {bet.book} for {money(bet.stake)}"
        )
    return 0


def cmd_bets(args: argparse.Namespace, settings: Settings) -> int:
    with Journal(settings.data.db) as journal:
        rows = journal.bets(
            status=args.status, sport=args.sport, market=args.market, book=args.book, limit=args.limit
        )
        if args.json:
            emit_json([b.to_dict() for b in rows])
            return 0
        if not rows:
            print("no bets recorded yet")
            return 0
        header = (
            f"{'id':>4} {'placed':<11} {'player':<20} {'market':<26} {'sel':<20} "
            f"{'odds':>6} {'stake':>9} {'status':<8} {'P/L':>9} {'CLV':>7}"
        )
        print(header)
        for bet in rows:
            selection = f"{bet.side} {bet.line:g}"
            clv = f"{bet.clv_pct * 100:+.1f}%" if bet.clv_pct is not None else "--"
            profit = f"{bet.profit:+,.2f}" if bet.profit is not None else "--"
            print(
                f"{bet.id:>4} {bet.placed_at[:10]:<11} {bet.player[:20]:<20} {bet.market[:26]:<26} "
                f"{selection:<20} {format_american(bet.american):>6} {bet.stake:>9,.2f} "
                f"{bet.status:<8} {profit:>9} {clv:>7}"
            )
    return 0


def cmd_settle(args: argparse.Namespace, settings: Settings) -> int:
    with Journal(settings.data.db) as journal:
        if args.result is not None:
            bet = journal.settle_with_result(args.id, args.result)
        elif args.status:
            bet = journal.settle(args.id, args.status)
        else:
            print("give either --result <actual stat> or --status won|lost|push|void")
            return 2
        print(
            f"#{bet.id} {bet.player} {bet.side} {bet.line:g} -> {bet.status.upper()} "
            f"({bet.profit:+,.2f}) {DIM} balance {money(journal.balance())}"
        )
    return 0


def cmd_close(args: argparse.Namespace, settings: Settings) -> int:
    with Journal(settings.data.db) as journal:
        bet = journal.record_closing(
            args.id,
            over_american=args.over,
            under_american=args.under,
            line=args.line,
            devig_method=settings.devig.method,
        )
        print(
            f"#{bet.id} closing fair {pct(bet.closing_fair_prob or 0)} "
            f"({format_american(bet.closing_american or 0)}) {DIM} "
            f"CLV {pct(bet.clv_pct, 2, signed=True)}"
        )
    return 0


def cmd_close_all(args: argparse.Namespace, settings: Settings) -> int:
    """Pull a fresh snapshot and stamp closing lines onto bets that lack them."""
    provider = build_odds_provider(settings)
    with Journal(settings.data.db) as journal:
        pending_markets = sorted({b.market for b in journal.bets() if b.closing_fair_prob is None})
        if not pending_markets:
            print("every bet already has a closing line")
            return 0
        props = provider.player_props(args.sport, pending_markets)
        updated = journal.record_closing_from_props(props, devig_method=settings.devig.method)
        print(f"stamped closing lines on {len(updated)} bet(s): {updated or '--'}")
    return 0


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    with Journal(settings.data.db) as journal:
        bets = journal.bets(sport=args.sport, since=args.since)
        perf = summarise(bets)
        curve = bankroll_curve(bets, starting=journal.deposits())
        if args.json:
            emit_json(
                {
                    "bankroll": {
                        "deposits": journal.deposits(),
                        "balance": journal.balance(),
                        "available": journal.available(),
                    },
                    "performance": asdict(perf) | {"roi": perf.roi, "win_rate": perf.win_rate},
                    "breakdown": [asdict(p) | {"roi": p.roi} for p in breakdown(bets, args.group)]
                    if bets
                    else [],
                    "max_drawdown": max_drawdown(curve),
                }
            )
            return 0

        print("BANKROLL")
        print(f"  balance           {money(journal.balance()):>12}")
        print(f"  pending stakes    {money(journal.pending_stake()):>12}")
        print(f"  available         {money(journal.available()):>12}")
        print(f"  max drawdown      {money(max_drawdown(curve)):>12}")
        print()
        print(render_performance(perf))
        if not bets:
            return 0
        print()
        print(render_breakdown(breakdown(bets, args.group, min_bets=args.min_bets), args.group))

        cal = calibration([b for b in bets if b.is_settled])
        if cal.buckets:
            print()
            print(f"CALIBRATION (Brier {cal.brier():.4f})")
            print(f"  {'bucket':<12} {'bets':>5} {'predicted':>10} {'actual':>8}")
            for label, n, predicted, actual in cal.buckets:
                print(f"  {label:<12} {n:>5} {predicted:>9.1%} {actual:>8.1%}")
    return 0


def cmd_snapshot(args: argparse.Namespace, settings: Settings) -> int:
    provider = build_odds_provider(settings)
    if not hasattr(provider, "snapshot"):
        print("the local provider reads snapshots; it cannot create them. Set an API key first.")
        return 2
    markets = args.market or market_keys(args.sport)
    path = provider.snapshot(args.sport, markets, args.out)
    remaining = getattr(provider, "requests_remaining", None)
    print(f"wrote {path}" + (f" ({remaining} API requests left)" if remaining is not None else ""))
    return 0


def cmd_fetch(args: argparse.Namespace, settings: Settings) -> int:
    """Pull real game logs and write them where the models look for them."""
    from .ingest import (
        AmbiguousPlayer,
        FetchError,
        HttpClient,
        UnknownPlayer,
        build_source,
        default_seasons,
        write_logs,
    )

    if "sample" in Path(settings.data.gamelogs).parts:
        print(
            "refusing to write fetched logs into the synthetic fixture "
            f"({settings.data.gamelogs}).\n"
            "Drop --sample: real logs belong in data/gamelogs, and mixing the two "
            "produces enormous imaginary edges."
        )
        return 2

    players = list(args.player or [])
    if args.from_props:
        provider = build_odds_provider(settings)
        props = provider.player_props(args.sport, market_keys(args.sport))
        players.extend(sorted({p.player for p in props}))
    if not players:
        print("give --player NAME (repeatable) or --from-props to take tonight's board")
        return 2

    seasons = (
        [int(s) for s in args.seasons.split(",")]
        if args.seasons
        else default_seasons(args.sport, args.season_count)
    )
    client = HttpClient(cache_dir=args.cache or settings.data.ingest_cache, offline=args.offline)
    source = build_source(args.sport, args.source, client=client)
    print(
        f"fetching {args.sport.upper()} logs for {len(players)} player(s) "
        f"from {source.name}, seasons {', '.join(str(s) for s in seasons)}\n"
    )

    failures = unreachable = 0
    for name in dict.fromkeys(players):  # de-duplicate, keep order
        try:
            fetched = source.fetch(name, seasons, team=args.team)
        except (UnknownPlayer, AmbiguousPlayer) as exc:
            print(f"  !! {name}: {exc}")
            failures += 1
            continue
        except FetchError as exc:
            print(f"  !! {name}: {exc}")
            failures += 1
            unreachable += 1
            continue
        if not fetched.games:
            print(f"  -- {name}: matched {fetched.ref.describe()} but no games in those seasons")
            failures += 1
            continue
        path = write_logs(fetched, settings.data.gamelogs, merge=not args.replace)
        print(
            f"  ok {fetched.player:<24} {len(fetched.games):>3} games  "
            f"{fetched.games[0]['date']} to {fetched.games[-1]['date']}  -> {path}"
        )

    print(f"\n{len(players) - failures} of {len(players)} player(s) written to {settings.data.gamelogs}/{args.sport}/")
    if unreachable:
        print(
            f"{unreachable} failure(s) were network or policy errors, not bad names -- "
            f"check that {source.name} is reachable from this machine"
        )
    elif failures:
        print("re-run failures with --team to disambiguate, or check the spelling")
    return 1 if failures and failures == len(players) else 0


def cmd_sources(args: argparse.Namespace, settings: Settings) -> int:
    from .ingest import DEFAULT_SOURCE, SOURCE_NOTES, SOURCES

    for sport, available in SOURCES.items():
        print(f"{sport.upper()}")
        for name in available:
            marker = "*" if DEFAULT_SOURCE[sport] == name else " "
            print(f"  {marker} {name:<16} {SOURCE_NOTES.get(name, '')}")
    print("\n* = default. Pick another with --source.")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """Run the read-only dashboard / JSON API."""
    import logging
    import os

    from .server import make_server

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    port = args.port or int(os.environ.get("PORT", 8080))
    token = args.token or os.environ.get("PMBOT_API_TOKEN", "")

    if not token and not args.allow_anonymous:
        print(
            "refusing to start without authentication.\n\n"
            "This service exposes your bankroll, open positions and betting history, "
            "and a hosted URL is public by default.\n"
            "Set PMBOT_API_TOKEN to a long random string, or pass --allow-anonymous "
            "if you are genuinely binding to a private network.",
            file=sys.stderr,
        )
        return 2
    if not token:
        print("WARNING: serving with no authentication -- anyone with the URL sees your journal.")

    server = make_server(settings, host=args.host, port=port, token=token or None, scan_ttl=args.scan_ttl)
    print(f"pmbot serving on http://{args.host}:{port} (auth: {'on' if token else 'OFF'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


def cmd_cron(args: argparse.Namespace, settings: Settings) -> int:
    """One scheduled pass: refresh logs, snapshot odds, scan, record closings.

    Written to be run by a scheduler. Every step is opt-out except logging
    bets, which is opt-*in*: a journal should only ever contain wagers a
    person actually placed.
    """
    from .ingest import FetchError, HttpClient, build_source, default_seasons, write_logs

    started = datetime.now(timezone.utc)
    print(f"=== pmbot cron {args.sport.upper()} {started:%Y-%m-%d %H:%M UTC}")
    failures = 0

    if not args.skip_fetch:
        try:
            provider = build_odds_provider(settings)
            props = provider.player_props(args.sport, market_keys(args.sport))
            players = sorted({p.player for p in props})
            print(f"[fetch] {len(players)} player(s) on the board")
            source = build_source(
                args.sport, args.source, client=HttpClient(cache_dir=settings.data.ingest_cache)
            )
            seasons = default_seasons(args.sport, args.season_count)
            written = 0
            for name in players:
                try:
                    fetched = source.fetch(name, seasons)
                except (LookupError, FetchError) as exc:
                    print(f"[fetch] skip {name}: {exc}")
                    continue
                if fetched.games:
                    write_logs(fetched, settings.data.gamelogs)
                    written += 1
            print(f"[fetch] refreshed {written} player(s) from {source.name}")
        except Exception as exc:  # noqa: BLE001 - one broken step shouldn't kill the run
            print(f"[fetch] FAILED: {exc}")
            failures += 1

    if not args.skip_close:
        try:
            with Journal(settings.data.db) as journal:
                pending = sorted({b.market for b in journal.bets() if b.closing_fair_prob is None})
                if pending:
                    provider = build_odds_provider(settings)
                    props = provider.player_props(args.sport, pending)
                    updated = journal.record_closing_from_props(props, devig_method=settings.devig.method)
                    print(f"[close] stamped closing lines on {len(updated)} bet(s)")
                else:
                    print("[close] nothing waiting on a closing line")
        except Exception as exc:  # noqa: BLE001
            print(f"[close] FAILED: {exc}")
            failures += 1

    if not args.skip_scan:
        try:
            bankroll = args.bankroll
            if bankroll is None:
                with Journal(settings.data.db) as journal:
                    balance = journal.balance()
                bankroll = balance if balance > 0 else settings.staking.bankroll
            signals = PropScanner(settings).scan(args.sport, bankroll=bankroll, include_passes=False)
            print(f"[scan] {len(signals)} bet(s) at bankroll {money(bankroll)}")
            for sig in signals:
                print("  " + sig.headline())
            if args.log and signals:
                with Journal(settings.data.db) as journal:
                    ids = [journal.log_signal(s, bankroll=bankroll) for s in signals]
                print(f"[scan] logged bet(s) {ids}")
        except Exception as exc:  # noqa: BLE001
            print(f"[scan] FAILED: {exc}")
            failures += 1

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"=== done in {elapsed:.1f}s, {failures} step(s) failed")
    return 1 if failures else 0


def cmd_config(args: argparse.Namespace, settings: Settings) -> int:
    if args.write:
        path = settings.save(args.write)
        print(f"wrote {path}")
        return 0
    emit_json(settings.to_dict())
    return 0


# ==========================================================================
# parser
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pmbot",
        description="Two-model, de-vigged, Kelly-sized player prop bot with a betting journal.",
    )
    parser.add_argument("--config", help="path to a JSON config file")
    parser.add_argument("--db", help="override the journal database path")
    parser.add_argument(
        "--sample", action="store_true",
        help="run against the synthetic slate in data/sample instead of fetched data",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- scan
    p = sub.add_parser("scan", help="find +EV player props on a slate")
    p.add_argument("--sport", required=True, choices=SPORTS)
    p.add_argument("--market", action="append", help="restrict to a market (repeatable)")
    p.add_argument("--player", action="append", help="restrict to a player (repeatable)")
    p.add_argument("--bankroll", type=float, help="override the configured bankroll")
    p.add_argument("--all", action="store_true", help="show passes as well as bets")
    p.add_argument("--include-started", action="store_true", help="include games already underway")
    p.add_argument("--log", action="store_true", help="write the recommended bets to the journal")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_scan)

    # -- project
    p = sub.add_parser("project", help="show both models' view of one prop")
    p.add_argument("--sport", required=True, choices=SPORTS)
    p.add_argument("--player", required=True)
    p.add_argument("--market", required=True)
    p.add_argument("--line", type=float, required=True)
    p.add_argument("--opponent")
    p.add_argument("--home", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--workload", type=float, help="override projected minutes/PA/targets")
    p.add_argument("--market-prob", type=float, help="fair over probability to gate against")
    p.add_argument("--odds", type=float, help="american odds on offer, to gate against a real price")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_project)

    # -- markets
    p = sub.add_parser("markets", help="list supported prop markets")
    p.add_argument("--sport", choices=SPORTS)
    p.set_defaults(func=cmd_markets)

    # -- bankroll
    p = sub.add_parser("bankroll", help="show or move the bankroll")
    p.add_argument("--deposit", type=float)
    p.add_argument("--withdraw", type=float)
    p.add_argument("--adjust", type=float, help="signed correction, e.g. a bonus or a fee")
    p.add_argument("--note")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_bankroll)

    # -- log
    p = sub.add_parser("log", help="record a bet you placed")
    p.add_argument("--sport", required=True, choices=SPORTS)
    p.add_argument("--player", required=True)
    p.add_argument("--market", required=True)
    p.add_argument("--side", required=True, choices=("over", "under"))
    p.add_argument("--line", type=float, required=True)
    p.add_argument("--book", required=True)
    p.add_argument("--odds", type=float, required=True, help="american odds, e.g. -115 or 140")
    p.add_argument("--stake", type=float, required=True)
    p.add_argument("--model-prob", type=float)
    p.add_argument("--fair-prob", type=float)
    p.add_argument("--tag")
    p.add_argument("--note")
    p.set_defaults(func=cmd_log)

    # -- bets
    p = sub.add_parser("bets", help="list journal entries")
    p.add_argument("--status", help="pending | settled | won | lost | push | void")
    p.add_argument("--sport", choices=SPORTS)
    p.add_argument("--market")
    p.add_argument("--book")
    p.add_argument("--limit", type=int)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_bets)

    # -- settle
    p = sub.add_parser("settle", help="grade a bet")
    p.add_argument("id", type=int)
    p.add_argument("--result", type=float, help="what the player actually did")
    p.add_argument("--status", choices=("won", "lost", "push", "void", "half_won", "half_lost"))
    p.set_defaults(func=cmd_settle)

    # -- close
    p = sub.add_parser("close", help="record a closing line for CLV")
    p.add_argument("id", type=int)
    p.add_argument("--over", type=float, required=True, help="closing american odds on the over")
    p.add_argument("--under", type=float, required=True, help="closing american odds on the under")
    p.add_argument("--line", type=float, help="closing line, if it moved")
    p.set_defaults(func=cmd_close)

    p = sub.add_parser("close-all", help="stamp closing lines from a fresh odds snapshot")
    p.add_argument("--sport", required=True, choices=SPORTS)
    p.set_defaults(func=cmd_close_all)

    # -- report
    p = sub.add_parser("report", help="ROI, win rate, CLV and calibration")
    p.add_argument("--group", default="market", help="sport | market | book | side | player | month | tag")
    p.add_argument("--sport", choices=SPORTS)
    p.add_argument("--since", help="ISO date, e.g. 2026-08-01")
    p.add_argument("--min-bets", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    # -- snapshot
    p = sub.add_parser("snapshot", help="save a live odds snapshot (needs an API key)")
    p.add_argument("--sport", required=True, choices=SPORTS)
    p.add_argument("--market", action="append")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_snapshot)

    # -- fetch
    p = sub.add_parser("fetch", help="download real game logs for the models to read")
    p.add_argument("--sport", required=True, choices=SPORTS)
    p.add_argument("--player", action="append", help="player name (repeatable)")
    p.add_argument("--from-props", action="store_true", help="every player on the current board")
    p.add_argument("--seasons", help="comma-separated, e.g. 2024,2025")
    p.add_argument("--season-count", type=int, default=2, help="how many recent seasons (default 2)")
    p.add_argument("--source", help="override the default source for the sport")
    p.add_argument("--team", help="disambiguate a shared name, e.g. --team BUF")
    p.add_argument("--replace", action="store_true", help="overwrite instead of merging")
    p.add_argument("--offline", action="store_true", help="use only what's already cached")
    p.add_argument("--cache", help="cache directory (default .cache/ingest)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("sources", help="list the game-log sources per sport")
    p.set_defaults(func=cmd_sources)

    # -- serve
    p = sub.add_parser("serve", help="run the read-only dashboard and JSON API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, help="defaults to $PORT, then 8080")
    p.add_argument("--token", help="defaults to $PMBOT_API_TOKEN")
    p.add_argument("--allow-anonymous", action="store_true", help="serve with no auth (private networks only)")
    p.add_argument("--scan-ttl", type=float, default=300.0, help="seconds to cache a slate scan")
    p.set_defaults(func=cmd_serve)

    # -- cron
    p = sub.add_parser("cron", help="one scheduled pass: fetch, close, scan")
    p.add_argument("--sport", required=True, choices=SPORTS)
    p.add_argument("--bankroll", type=float, help="defaults to the journal balance")
    p.add_argument("--log", action="store_true", help="write the recommended bets to the journal")
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--skip-scan", action="store_true")
    p.add_argument("--skip-close", action="store_true")
    p.add_argument("--source", help="override the game-log source")
    p.add_argument("--season-count", type=int, default=2)
    p.set_defaults(func=cmd_cron)

    # -- config
    p = sub.add_parser("config", help="show or write the effective configuration")
    p.add_argument("--write", nargs="?", const="pmbot.config.json", help="write config to a file")
    p.set_defaults(func=cmd_config)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    if args.sample:
        settings.data.use_sample()
    if args.db:
        settings.data.db = args.db
    try:
        return int(args.func(args, settings))
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
