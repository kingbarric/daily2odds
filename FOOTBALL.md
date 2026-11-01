# Football prediction and paper accumulator workflow

> Latest odds integration and trained probability-first experiment: see [FOOTBALL_ODDS_API.md](FOOTBALL_ODDS_API.md). With ODDS_API_KEY configured, auto can use The Odds API; earlier public-only instructions below describe the prior workflow.


> Current implementation update: see [FOOTBALL_DEVELOPMENT.md](FOOTBALL_DEVELOPMENT.md) for executed 2024/25–2025/26 development backtests, public-source five-day forecasts, the new default `--provider auto`, £10 paper stakes and separate paper database. The API-Football key is now optional for the public fallback; current-season access on its free plan was tested and rejected. Earlier setup examples below describe the original authenticated-provider workflow.


This extends the existing historical pipeline and Dixon–Coles model. No additional prediction model was introduced. There is **no demonstrated profitable edge**. Existing evaluation reports are retained as legacy research; their historical estimated odds times do not prove executable prices.

## Install and run

From this repository, using Python 3.11+ (tested with 3.13):

```sh
.venv/bin/python -m pip install -r requirements.txt
```

LightGBM on macOS still requires the existing `libomp` dependency. The forward workflow uses Dixon–Coles, not LightGBM. The existing database is `data/football/football.db`; do not delete or recreate it.

Set `API_FOOTBALL_KEY` in `.env` to your API-Football API-Sports key. This is a data-provider key, never a bookmaker login. Provider entitlement to the current season/odds must be confirmed using your account; it was not possible to verify entitlement without a configured key. Requests have timeouts, bounded pagination, explicit error handling and no automatic retries. The local daily cap counts attempted HTTP requests, persists through restarts, and does not count other applications using your provider account.

```sh
# One cycle: refresh results, ingest next five days, collect prices, freeze forecasts,
# search tickets and reserve paper stakes only for qualifying tickets.
.venv/bin/python football_forward.py --team-map config/football_teams.example.json

# Same cycle, collect forecasts without reserving paper stakes:
.venv/bin/python football_forward.py --forecast-only

# Persistent worker: scheduled every eight hours until stopped.
.venv/bin/python football_forward.py --watch --interval 28800

# Read-only dashboard:
.venv/bin/python -m streamlit run football_dashboard.py
```

The worker must remain running (or run the single-cycle command from your existing service/cron). No operating-system scheduled task is silently installed. It defaults to 100 requests per UTC day, at most 20 fixtures' odds per cycle, 1 paper unit per ticket and a one-time 1,000-unit paper bankroll. These are configurable CLI options. Result refreshes, pagination and five-day fixture discovery also consume requests; exhausted budgets stop data collection until the next UTC day. A provider outage never substitutes sample data. Already-persisted results settle before network calls; newly received results settle before unrelated fixture discovery.

Historical canonical club names are matched exactly. The optional JSON maps API names to existing historical names. The supplied file is an example subset, not a complete alias catalogue. Unknown names warn and skip affected fixtures; invalid alias targets fail rather than merge clubs incorrectly. Copy and extend the example as required. No fuzzy matching is used.

For another database, use `--db /absolute/path/football.db` for the worker and `FOOTBALL_DB_PATH=/absolute/path/football.db` for the dashboard. Worker schema changes are additive, applied at startup; the dashboard opens SQLite read-only and never migrates it.

## Historical timing corrections

- CSVs lack verified individual quote timestamps. New imports leave `odds_snapshots.captured_at_utc` NULL and set `captured_at_estimated=1`. The collection schedule is only an estimate retained on matches where it precedes a known kickoff. Early holiday kickoffs are **never assigned fabricated earlier quote times**.
- Existing stored quotes and legacy guessed timestamps are preserved. `db.odds_matrix()` excludes estimated/unknown timestamps by default, and verified pre-closing prices must precede a known kickoff. `known_by` additionally restricts quote availability.
- Head-to-head features include only meetings strictly before the forecast's decision date, not merely before match day. Same-day results are excluded.
- Elo applies results by actual match date, not forecast group; postponed matches cannot contribute their results early. Dixon–Coles itself filters future results and withholds forecasts on failed numerical convergence.
- Backtest cache names changed so pre-correction walks are not reused. Existing `football_walkforward.md`, `football_baselines.md` and `football_data_quality.md` are not overwritten by the corrected entry points.

A strict historical odds backtest stops when no verified quote timestamps exist. An explicitly requested sensitivity run remains possible:

```sh
.venv/bin/python football_backtest.py --allow-estimated-odds
```

It writes `reports/football_walkforward_corrected.md`, labels the timing limitation, and is **not evidence of executable returns**. Corrected ingestion writes `reports/football_data_quality_corrected.md`. Historical ingestion remains optional; no re-ingestion is needed to exclude legacy estimated quotes safely.

## Forecasts and uncertainty

The forward model keeps the existing evaluated settings: country-specific Dixon–Coles, 365-day half-life for 1X2/double chance, 730 days for goal markets, 3,650-day history window. It uses only completed matches before the current UTC day. Newly received finished fixture results are merged into training in memory, deduplicated against historical rows; historical tables are not rewritten. Same-day results and results received after the fitting cutoff are excluded. New/unmapped clubs without sufficient country history are skipped.

Each forecast records its actual UTC creation time, kickoff, model/settings version, exact training-row hash, training cutoff, probabilities and conservative estimates. Updates append new versions. SQL triggers reject UPDATE/DELETE on forecasts, live odds and ledger entries. A fixture that starts while fitting is skipped before insertion. Rescheduled fixtures require a fresh forecast.

Conservative estimates use the 95% Wilson lower hit-rate estimate for the same market/selection and 10-percentage-point probability bin, capped at the model's probability. Only previously frozen, settled forward forecasts received before the new decision count. One earliest forecast per match enters calibration, chosen **before** assigning its bin. At least 100 distinct matches per bin are required; otherwise the conservative probability is zero and the workflow returns **NO BET**. Forecasts continue accumulating even while no tickets qualify. These bin estimates are not a confidence guarantee for a particular match, and distribution changes can invalidate historical calibration.

## Ticket rules

- Only genuine provider bookmaker quotes with provider update timestamps and actual local receipt timestamps; no synthetic odds or probabilities-as-prices.
- Source update and receipt must be in the past and no more than 60 minutes old. Forecasts older than 24 hours are not used. Always use the latest collected quote per bookmaker/market/selection, not an older more favourable quote.
- 2–5 legs from different matches, one bookmaker per ticket; combined decimal odds **4.75–5.25**, inclusive.
- Nominal joint probability is the product. Conservative product is additionally multiplied by `0.95 ** (legs - 1)` to flag/penalise unknown shared-context dependence. This is a heuristic, **not a proved joint lower bound**.
- Require conservative EV of at least 2% in the workflow, where EV = probability × combined odds − 1. Return up to three tickets with disjoint matches, ranked by conservative then nominal EV, or **NO BET**.
- Bounded beam search: up to 100 eligible candidates, beam width 1,000. It evaluates combinations, but does not promise exhaustive/global optimality.
- Paper placement checks immutable forecast/odds references and available cash in a `BEGIN IMMEDIATE` transaction. A match can have paper exposure only once, even across concurrent workers and later quote revisions. Repeating an identical placement is idempotent. Paper stake reservation is automatic; real-money betting is absent.

The dashboard shows upcoming fixtures, ranked single estimates, proposed tickets, immutable prediction history, frozen-forward diagnostics and bankroll. Displayed nominal/conservative EV is an estimate, not realised performance or a recommendation.

## Paper settlement policy

Only full-time regulation scores settle these markets; extra time and penalties are excluded. The provider adapter uses the regulation `score.fulltime` fields for AET/PEN fixtures. Matches still playing remain pending. Cancelled and abandoned fixtures void; postponed fixtures remain pending for 48 hours after the ticket's original kickoff, then void. A reschedule more than 48 hours beyond original kickoff voids that leg. Voided legs pay at 1.00; all-void tickets return the stake. A lost leg settles a ticket as lost without waiting for unrelated legs.

This explicit paper policy is not a claim about any bookmaker's specific rules. Repeated settlement cannot credit twice. Later result corrections do not rewrite settled ticket/ledger records; reconciliation requires explicit manual review. Cash, reserved stakes, equity at stake cost, profit, settled ROI, hit rate and realised drawdown persist across restarts. Displayed zero ROI before settlement means no settled sample, not established break-even performance.

## Validation

See `reports/football_implementation_validation.md` for commands, results and preservation checks. Automated tests use temporary databases and recorded provider responses; they never make real-money bets or mutate the production historical database.
