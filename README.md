# daily2odds

Football probability research and a free website using Dixon–Coles goal modelling, Elo features and LightGBM.

Live: https://daily2odds.playhuud.com/ — https://daily2odds.com/ redirects there.

Includes chronological training/calibration, seven-day forecasts, saved bookmaker quotes, combination analysis and prediction history. Probabilities are estimates, not guarantees. Development backtests do not establish independent performance or profitability. Current-season injury/suspension adjustments are unavailable.

## Layout

- `sites/touchline-studio/`: Daily2Odds React/Vinext website. The folder name is retained for pipeline compatibility.
- `src/football/`: historical ingestion, features, models, evaluation, forecasting, settlement and saved history.
- `football_*.py`: command-line workflows and the separate private Streamlit interface.
- `refresh_touchline_odds.py`: budget-capped scheduled bookmaker collection.
- `tests/`: pipeline regression tests. Website checks are in `sites/touchline-studio/tests/`.
- `FOOTBALL*.md`: implementation and evaluation documentation.

Databases, trained model binaries, subscriber records, provider keys, logs and private SSH configuration are excluded. Website `data/*.json` files are previously published sample snapshots, not live data.

## Website

Requires Node.js 22.13 or newer.

```sh
cd sites/touchline-studio
npm ci
npm run build
npm start -- --port 8502
```

A local D1-compatible SQLite binding stores runtime caches and optional email signup. Public JSON snapshots provide a fallback. On a fresh development database, apply the SQL migrations in `drizzle/` before testing signup or cache persistence.

## Football pipeline

Requires Python 3.10 or newer. LightGBM on macOS may require `brew install libomp`.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Set provider keys locally; never commit .env.
.venv/bin/python football_ingest.py --first 2016
.venv/bin/python -u football_commercial_nightly.py
```

Build the website before the commercial workflow imports caches. Training uses an eight-year chronological window and separate recent calibration. Historical coverage varies. The commercial database is isolated from the original database. Visitors read saved analysis and never trigger provider requests or training. API subscription limits and daily credit caps apply.

## Validation

```sh
.venv/bin/python -m pytest tests -q
cd sites/touchline-studio
npx tsc --noEmit
node --experimental-strip-types tests/team-search.test.mjs
node --experimental-strip-types tests/combination-search.test.mjs
node --experimental-strip-types tests/combination-dates.test.mjs
node --experimental-strip-types tests/upcoming-fixtures.test.mjs
node --experimental-strip-types tests/history-support.test.mjs
node tests/today-combination.test.mjs
```

## Deployment

The live site runs in a separate Docker container behind Caddy on the owner's VPS with its own persistent SQLite volume. See `sites/touchline-studio/ops/DEPLOYMENT.md`. Training and scheduled odds collection still run on the owner's Mac and sync public caches; they have not moved to the VPS.

Before building Docker, populate `sites/touchline-studio/seed/forecasts.json`, `prices.json` and `history.json` from reviewed public snapshots. Demo snapshots are available in the corresponding website `data/` files. Production should use the latest successful public caches. Never seed subscriber data.

Optional SSH cache-sync settings belong in the ignored `data/football/commercial/vps_publish.json` (`destination` and `identity`). Keep SSH private keys outside this repository. The VPS creates its own internal cache-import token in an untracked `.env`; it is never sent to browsers.
