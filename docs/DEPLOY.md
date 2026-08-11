# Deploying pmbot on Railway

**One service.** The web process serves the dashboard *and* runs the daily
refresh on a timer inside it.

That is not the obvious design, and it's deliberate: **a Railway volume
attaches to exactly one service.** A separate cron service would get its own
empty disk — refreshing game logs nowhere useful and stamping closing lines on
an empty journal — while the real journal never updated. Nothing would error.
The numbers would just quietly be wrong, which is the failure mode this whole
project exists to avoid.

`pmbot cron` still exists as a CLI command for manual runs, and for platforms
where a scheduled job *can* share the same storage.

---

## 1. Create the web service

New Project → Deploy from GitHub repo → pick this repo. Railway reads
`railway.toml` and builds the `Dockerfile`; there's nothing to configure for
the build.

## 2. Attach a volume — do this before anything else

**Railway's container filesystem is wiped on every deploy.** Without a volume
your journal, your bankroll and your fetched game logs disappear the next time
you push, and the bot silently starts over from an empty database.

Service → Variables → **Volumes** → New Volume, mount path **`/data`**.

The image already sets `PMBOT_DATA_DIR=/data`, which moves every writable path
in one go:

```
/data/pmbot.sqlite3     the journal and bankroll
/data/gamelogs/         fetched game logs
/data/props/            odds snapshots
/data/cache/            HTTP caches (odds + ingest)
```

## 3. Set variables

| variable | required | what it's for |
|---|---|---|
| `PMBOT_API_TOKEN` | **yes** | auth for the dashboard. A Railway URL is public and this page shows your bankroll and open positions. The server *refuses to start* without it. |
| `PMBOT_ODDS_API_KEY` | for real odds | [The Odds API](https://the-odds-api.com/) key. Without it the scanner has no prices and the dashboard renders empty. |
| `PMBOT_BANKROLL` | no | starting bankroll if the journal has no deposits yet |
| `PMBOT_KELLY_FRACTION` | no | defaults to `0.25` |
| `PMBOT_MIN_EV` | no | EV floor, defaults to `0.02` |
| `PMBOT_DAILY_AT` | **for the daily job** | UTC time to run the daily pass, e.g. `16:00`. Unset means no scheduled run at all. |
| `PMBOT_DAILY_SPORTS` | no | comma-separated, defaults to `nba`. Use `nba,nfl` for both. |
| `PMBOT_DAILY_LOG_BETS` | no | `1` makes the daily pass write its bets to the journal. Off by default — see below. |
| `PMBOT_MC_SIMS` | no | lower it (e.g. `8000`) if a big slate is slow on a small instance |
| `PORT` | no | injected by Railway |

Generate a token that isn't guessable:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 4. Turn on the daily run

Set two variables on the web service:

```
PMBOT_DAILY_AT=16:00
PMBOT_DAILY_SPORTS=nba,nfl
```

That's it — no second service, no cron syntax. On boot the log says:

```
daily pass armed for 16:00 UTC on nba, nfl
pmbot serving on http://0.0.0.0:8080 (auth: on)
```

Each run, in order:

1. **fetch** — refresh game logs for every player on tonight's board
2. **close** — stamp closing lines on bets still missing them, for CLV
3. **scan** — price the slate and log what it would bet

Then it drops the dashboard's scan cache so the page reflects the new logs.

**It does not write bets to the journal** unless you set
`PMBOT_DAILY_LOG_BETS=1`. A journal is a record of wagers a person actually
placed; one that fills itself with bets you never made destroys the only
honest measurement you have. Turn it on only if the deployment really is
placing them.

### Watching it

The job is observable, because a scheduler you can't see is one you can't
trust:

```bash
curl -H "Authorization: Bearer $PMBOT_API_TOKEN" https://<app>/api/schedule
```
```json
{
  "enabled": true, "schedule_utc": "16:00", "running": true, "runs": 4,
  "last_status": "ok", "last_finished_at": "2026-08-11T16:00:09+00:00",
  "last_duration_seconds": 7.6, "last_summary": "nba: 2 bet(s); nfl: 1 bet(s)",
  "next_run": "2026-08-12T16:00:00+00:00"
}
```

The same summary sits in the dashboard footer, and in `/healthz` under
`daily_job`.

### What it does about restarts and misses

* **A redeploy doesn't re-run it.** State lives on the volume, keyed by UTC
  date, so restarting at 16:05 after a 16:00 run does nothing.
* **A missed window catches up.** Down at 16:00, up at 18:00 → it runs at
  18:00. For a data refresh, late beats skipped.
* **A failure doesn't kill the loop or spin.** The day is claimed before the
  job runs, so a job that fails in two seconds waits for tomorrow instead of
  retrying in a tight loop. The error lands in `last_summary`.

### Picking a time

`16:00` UTC (noon ET) is a reasonable default: late enough that lineups are
firming, early enough to be ahead of evening slates.

One honest caveat: **a once-daily job at noon is poor at capturing closing
lines.** CLV wants a reading near lock. If you care about CLV — and you
should, it's the metric that converges fastest — run the CLI closer to game
time as well:

```bash
pmbot close-all --sport nba     # near lock, on whatever schedule suits you
```

Run steps selectively with `--skip-fetch`, `--skip-close`, `--skip-scan`.

## 5. First run

The journal starts empty. Open a shell (`railway run`, or a one-off command)
and seed the bankroll:

```bash
pmbot bankroll --deposit 5000
pmbot fetch --sport nba --from-props
```

Then visit `https://<your-app>.up.railway.app/?token=<PMBOT_API_TOKEN>`.

---

## The API

Everything is read-only. Auth via `Authorization: Bearer <token>` or
`?token=<token>`.

| route | returns |
|---|---|
| `GET /` | HTML dashboard |
| `GET /healthz` | liveness — **unauthenticated**, so the platform can probe it |
| `GET /api/signals?sport=nba&all=1` | tonight's signals |
| `GET /api/overview` | bankroll, performance, open bets |
| `GET /api/bets?status=pending` | journal entries |
| `GET /api/schedule` | the daily job: last run, outcome, next run |
| `GET /api/config` | effective settings, API key masked |

```bash
curl -H "Authorization: Bearer $PMBOT_API_TOKEN" \
  "https://<your-app>.up.railway.app/api/signals?sport=nfl"
```

Slate scans are cached for `--scan-ttl` seconds (default 300). A full slate is
thousands of Monte Carlo simulations; without the cache a held-down refresh key
pegs the CPU.

---

## Things that will bite you

* **No volume = no journal.** The single most common way to lose everything
  here. Attach it before your first real bet, not after.
* **Don't scale past one instance.** SQLite on a volume attaches to a single
  instance; a second replica either can't mount it or corrupts it. The daily
  job would also fire once per replica.
* **Don't add a second service for cron.** It gets its own volume and its own
  empty journal. This is the trap the in-process scheduler exists to avoid.
* **Odds API quota is finite.** Player props cost one request per event. The
  scanner caches responses for `api.cache_ttl` seconds (default 120) — raise it
  if you're burning through a free tier.
* **The web service never places bets.** Nothing in the HTTP surface can write
  to the journal. Settling and grading stay on the CLI, where a person is.
* **Cold starts re-fetch.** The HTTP caches live on the volume, so mounting it
  also stops every deploy from re-downloading a season of game logs.
* **`--sample` and a fetching pass don't mix.** Fetched logs can never be
  written into `data/sample`; the store raises rather than overwrite the
  fixture. Don't run the deployment with `--sample` anyway — it prices real
  form against invented lines.

## Deploying somewhere else

Nothing here is Railway-specific beyond `railway.toml`. Any platform that runs
a container works the same way: set `PMBOT_DATA_DIR` to a persistent mount, set
`PMBOT_API_TOKEN`, expose `$PORT`, point health checks at `/healthz`.

```bash
docker build -t pmbot .
docker run -p 8080:8080 -v pmbot-data:/data \
  -e PMBOT_API_TOKEN=... -e PMBOT_ODDS_API_KEY=... \
  -e PMBOT_DAILY_AT=16:00 -e PMBOT_DAILY_SPORTS=nba,nfl \
  pmbot
```

If your platform *can* share storage between a long-running service and a
scheduled job, use `pmbot cron --sport nba --sport nfl` as the scheduled
command instead and leave `PMBOT_DAILY_AT` unset.
