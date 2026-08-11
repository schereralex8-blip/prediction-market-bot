# Deploying pmbot on Railway

Two services from one repo:

| service | start command | what it does |
|---|---|---|
| **web** | `pmbot serve --host 0.0.0.0` | read-only dashboard + JSON API |
| **cron** | `pmbot cron --sport nba` | scheduled refresh: game logs, closing lines, scan |

Both run the same image. Both must mount **the same volume**, or they won't
see the same journal.

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
| `PMBOT_MC_SIMS` | no | lower it (e.g. `8000`) if a big slate is slow on a small instance |
| `PORT` | no | injected by Railway |

Generate a token that isn't guessable:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 4. Add the cron service

New Service → same repo → then set:

- **Start command:** `pmbot cron --sport nba`
- **Cron schedule:** e.g. `0 16 * * *` (16:00 UTC daily)
- **Volume:** mount the *same* volume at `/data`
- **Variables:** same as the web service

Each run refreshes game logs for everyone on the board, stamps closing lines on
bets that are still missing them, and prints the slate it would bet.

It does **not** log bets to the journal unless you pass `--log`. That's
deliberate: a journal is a record of wagers a person actually placed, and one
that fills itself with bets you never made destroys the only honest measurement
you have. Turn it on only if the deployment really is placing them.

Skip steps with `--skip-fetch`, `--skip-close`, `--skip-scan`.

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
  instance; a second replica either can't mount it or corrupts it.
* **Odds API quota is finite.** Player props cost one request per event. The
  scanner caches responses for `api.cache_ttl` seconds (default 120) — raise it
  if you're burning through a free tier.
* **The web service never places bets.** Nothing in the HTTP surface can write
  to the journal. Settling and grading stay on the CLI, where a person is.
* **Cold starts re-fetch.** The HTTP caches live on the volume, so mounting it
  also stops every deploy from re-downloading a season of game logs.

## Deploying somewhere else

Nothing here is Railway-specific beyond `railway.toml`. Any platform that runs
a container works the same way: set `PMBOT_DATA_DIR` to a persistent mount, set
`PMBOT_API_TOKEN`, expose `$PORT`, point health checks at `/healthz`.

```bash
docker build -t pmbot .
docker run -p 8080:8080 -v pmbot-data:/data \
  -e PMBOT_API_TOKEN=... -e PMBOT_ODDS_API_KEY=... pmbot
```
