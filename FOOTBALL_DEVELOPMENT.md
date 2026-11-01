# Football development run and current forward workflow

> Latest odds integration and trained probability-first experiment: see [FOOTBALL_ODDS_API.md](FOOTBALL_ODDS_API.md). With ODDS_API_KEY configured, auto can use The Odds API; earlier public-only instructions below describe the prior workflow.


This extends the existing SQLite history, Dixon–Coles, Elo, LightGBM, beam optimiser and paper store. No new prediction model was introduced. The original database and all pre-existing football reports are preserved. New artifacts are in `reports/football_development_2024_2025_integrity_v3/`.

## Executed development backtests

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python football_development.py \
  --output reports/football_development_2024_2025_integrity_v3
.venv/bin/python football_development_report.py
```

For a new run, omit `--output` to create a timestamped report directory, then pass that path to the report command with `--folder`. `--phase markets` and `--phase accumulators` allow incremental execution. Source history is opened read-only. Versioned source-fingerprinted caches are separate from earlier caches. Existing evaluation reports are never replaced by these commands.

The run evaluates 8,040 matches in 2024/25 and 2025/26. The extra 2023/24 season warms up chronological calibration. Both target seasons are **development backtests**: previous tuning/model/strategy investigations included them. No fresh holdout claim is made. 2026/27 is excluded from this run but was inspected in earlier reports; independent validation is reserved for prospective frozen forecasts after `data/football/forward_policy_lock_v1.json`.

Dixon–Coles retains existing 365-day result/730-day goal half-lives and a 3,650-day history window. Elo retains K20 and previous-five-season probability mapping. LightGBM uses its existing form/Elo/H2H feature variant, 4 leaves/400 trees and sigmoid calibration; each target season trains through s−2 and calibrates on s−1. No model blending, new algorithms or hyperparameter search was added. Elo/LightGBM cannot supply goal-market probabilities; unsupported combinations remain explicit rather than fabricated.

Market/group scores and per-selection binary diagnostics are separate. 1X2 scores are multiclass. Double chance consists of overlapping binary events, never a three-class probability vector. Market-group hit rate selects the largest model probability; selection-level accuracy classifies each event at 0.5. Do not compare these different accuracy definitions as evidence of betting value. Calibration includes models and the margin-removed bookmaker benchmark.

Only 1X2 and Over/Under 2.5 have actual historical prices in this source. Double-chance bookmaker event probabilities may be derived from de-vigged 1X2 for scoring only: they are not obtainable odds. Other unpriced markets have probability diagnostics but no fabricated ROI. Bookmaker comparisons use matched available-price rows; historical timing remains unverified.

The strict accumulator run requires actual verified receipt before its scheduled decision and kickoff. There are zero such historical rows, so strict results are **NO BET, £0 stakes, undefined ROI**. A separate estimated-timing sensitivity run explicitly assumes availability at the publisher's schedule; its stored legs retain NULL verified receipt timestamps and labelled assumed times. These returns are hypothetical, not evidence that the prices were executable.

Tickets use £10 stakes, 2–5 distinct matches, one bookmaker, 4.75–5.25 decimal odds and at most three match-disjoint tickets per decision. Market averages/maxima and exchange quotes are excluded as bettable prices. Candidate truncation (60) and beam width (250) bound search; global optimality is not guaranteed. Conservative estimates require 100 previously settled match observations per league/selection/probability bin and use Wilson95 lower hit rates capped at model probability. Results from the current decision day cannot enter calibration. Individual conservative EV must be positive and ticket conservative EV at least 2% after the 5% per-additional-leg dependence haircut.

The dependence haircut is an explicit heuristic, not a guaranteed joint probability bound. `dependence_stress.csv` adds a 10% haircut and Fréchet event-overlap stress from the conservative marginal estimates, without changing selected tickets or treating these diagnostics as separately validated strategies. Odds already include bookmaker margins; margin is not deducted a second time from payouts.

Payouts round to pennies HALF_UP. Historical result availability is conservatively no earlier than the next UTC date or kickoff+3h. Realised drawdown uses simultaneous settlement batches. Losing streaks use settlement cutoff then deterministic ticket ID for ties. ROI confidence intervals bootstrap decision-date clusters and do not correct for selection bias, unverified prices, or small-sample instability. Full winning/losing tickets, legs, price provenance, probabilities, calibration counts, stakes and payouts are recorded.

## Current forward workflow

```sh
# One real paper cycle; no new API key required for this public-source fallback.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python football_forward.py \
  --provider auto --stake 10

# Persistent scheduled worker: every eight hours until stopped.
.venv/bin/python football_forward.py --provider auto --watch --stake 10

# Read-only dashboard:
.venv/bin/python -m streamlit run football_dashboard.py
```

Default `auto` combines daily ESPN public soccer scoreboards with football-data.co.uk's current quoted-price batch. It queries each UTC day intersecting the next five days, explicitly filters the ten configured league UIDs, skips unconfirmed kickoff times/unknown aliases, and reuses canonical fixture IDs across sources. Cup matches are not inferred from team names. Upstream completeness and stability are not contractually guaranteed; oversized/truncated scoreboard responses fail closed. Raw source responses and actual collection times are archived in `data/football/forward_sources/`.

Public bookmaker prices retain actual receipt and batch publication separately; batch publication is not proof of an individual bookmaker update. Missing source publication timestamps yield no usable quotes; source age over 60 minutes cannot qualify. Do not manufacture fresh odds from a newly downloaded stale batch. Publisher quotes must match the same canonical teams and kickoff as the schedule source. Market coverage and file freshness can limit tickets even while fixture predictions work.

`--provider public-csv` uses only the narrower publisher batch. `--provider api-football` retains the authenticated adapter for accounts with current-season entitlement. The supplied API-Football free key was authenticated, but the provider rejects current-season access; it allows 2022–2024 only. `auto` does not send that key to any public-source host.

`--db` points to the existing **read-only historical database**. `--paper-db` defaults to `data/football/paper_forward.db`, protecting original historical bytes. The dashboard defaults to that paper database, or use `FOOTBALL_DB_PATH` to choose another paper file. No historical re-ingestion is needed. The first paper deposit is £1,000 and is idempotent; the CLI stake defaults to £10. No bookmaker login, real funds, or automatic real-money bets exist.

Forecasts/quotes are append-only and immutable. The deployed existing Dixon–Coles model was not switched to the luckiest sensitivity result. Forward uncertainty uses only earlier frozen, settled forecasts from the same league, so cold-start quality can legitimately produce NO BET. Completed fixtures settle automatically on subsequent cycles using regulation scores, explicit void/postponement rules and idempotent ledger credits. Extra-time results without reliable regulation scores are withheld. See `FOOTBALL.md` for the paper settlement policy.

The worker must stay running or be invoked by your existing process scheduler. This implementation does not silently install an OS background service. One live cycle has been executed; report artifacts include its actual forecasts and NO BET result. No settled forward performance or statistically credible profitable edge is claimed.
