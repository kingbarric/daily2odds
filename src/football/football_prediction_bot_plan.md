# Football Prediction & Accumulator Optimiser — Implementation Brief

## Objective
Build a **paper-only** football prediction system that trains on historical matches, forecasts fixtures for the next five days, scores multiple markets, and optionally constructs accumulators targeting **4.75–5.25 decimal odds**. It must record predictions before kickoff and objectively measure calibration, accuracy and hypothetical returns. **No guarantee of profit or winning tickets.**

## Stack (free, local)
- **Python 3.11+**, `pandas`, `numpy`, `scipy`, `requests`, `pydantic`
- **Historical data:** [Football-Data.co.uk](https://www.football-data.co.uk/downloadm.php) season CSVs; check coverage, permitted use and licensing. Start with **2016/17–2025/26**; use 2026/27 for forward tests.
- **Upcoming fixtures/results:** [football-data.org](https://www.football-data.org/) or [API-Football](https://www.api-football.com/); verify current free-tier limits and league coverage before implementation.
- **Models:** Elo (custom Python); Dixon–Coles/Poisson ([penaltyblog](https://docs.pena.lt/y/), verify current API); [LightGBM](https://github.com/microsoft/LightGBM); [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html) calibration. XGBoost is **optional later**, not required for MVP.
- **Storage:** SQLite for MVP (PostgreSQL optional later); SQLAlchemy; APScheduler or cron for scheduled jobs.
- **Dashboard:** Streamlit + Plotly; pytest for tests.
- **Optimiser:** Python beam search or constrained enumeration; no LLM required for prediction.

## Input data
Store match ID, league, season, kickoff UTC, home/away team IDs, final scores, half-time scores where available, shots/cards where available, bookmaker and market odds **with source and timestamp**. Normalize team names/IDs; handle promotions, missing data and changing column schemas. Never treat post-kickoff or closing odds as pre-kickoff data unless their timestamp proves availability.

## Build process (implement in this order)
1. **Ingest:** Download and validate 10 completed seasons of CSVs for selected leagues. Create normalized `teams`, `matches`, `odds_snapshots` tables; deduplicate and report missing values.
2. **Leakage-safe features:** Process matches in chronological order. Compute pre-match Elo, rolling 5/10 match form, home/away splits, goals scored/conceded, rest days and season indicators. Use only information available before each kickoff. Older results should carry less weight.
3. **Baselines:** Implement home/draw/away historical-frequency baseline, Elo baseline and Dixon–Coles goal model. Produce 1X2, over/under 1.5/2.5/3.5 and BTTS probabilities from a coherent score distribution.
4. **ML:** Train LightGBM on pre-match features to estimate 1X2 probabilities. Keep a simple baseline and evaluate whether ML improves out-of-sample log loss/Brier score. Calibrate probabilities on a separate chronological validation set. Do not blend models without measured improvement.
5. **Backtest:** Use expanding-window or walk-forward season splits; e.g. early seasons for training, later seasons for tuning, and 2025/26 as a held-out test. No random match splitting. Benchmark against bookmaker implied probabilities after removing overround where possible.
6. **Forward predictions:** Fetch fixtures for the next five days (for example, 27 matches); calculate predictions daily and **freeze timestamped predictions before kickoff**. Later updates are separate versioned forecasts.
7. **Markets:** Generate candidate picks per match: 1X2, double chance, totals and BTTS; use actual obtainable decimal odds. Do not fabricate prices or calculate odds from model probabilities and call them bookmaker prices. Exclude markets without odds or reliable model probabilities.
8. **Ticket optimiser:** Enumerate/beam-search combinations from different matches, target combined odds **4.75–5.25**, max 2–5 legs initially. Rank by estimated joint win probability and expected value (`EV = joint_probability * combined_odds - 1`). For picks on different matches, independence is an approximation; account for shared contextual correlations or flag uncertainty. Prevent mutually exclusive or overlapping selections on the same match. Output **up to three tickets**, preferably diversified, and **NO BET** when quality thresholds fail. Do not force a 5.00 ticket.
9. **Settlement:** Fetch final results, settle every pick and ticket, and record paper stakes, returns, ROI, hit rate, drawdown, log loss, Brier score, calibration and sample sizes. Handle postponements, voids and bookmaker-specific settlement rules explicitly.
10. **Dashboard:** Tabs for upcoming fixtures, ranked single picks, proposed ~5.00 accumulators, prediction history, model diagnostics and paper bankroll. Show data freshness and prediction timestamp.

## Example output (illustrative only)
| Pick | Model probability | Available odds |
|---|---:|---:|
| Match A — Over 1.5 | 84% | 1.30 |
| Match B — Home win | 76% | 1.45 |
| Match C — Under 3.5 | 81% | 1.35 |

The optimiser must find a valid combination near 5.00 **only when prices and model quality justify it**. A 5.00 ticket has a **20% break-even hit rate**; a model claiming 25% gives theoretical +25% stake EV, but only if the probability is correctly calibrated. High single-pick accuracy alone does not imply positive betting returns.

## Acceptance criteria
- Reproducible install and one-command historical ingestion.
- Tests verifying no future-data leakage, valid probabilities summing to 1, correct odds multiplication, market settlement, and no duplicated matches in tickets.
- Walk-forward evaluation report with bookmaker baseline and uncertainty/sample-size disclosure.
- Daily scheduler stores immutable predictions and resolves results later.
- Dashboard displays real data or explicitly marks sample/mock data.
- Paper-only mode: **no bookmaker login, deposits or automatic real-money bets**.

## Initial deliverables for coding agent
1. Repo scaffold, `requirements.txt` or `pyproject.toml`, `.env.example`, README.
2. Historical CSV importer + SQLite schema + tests.
3. Elo and Dixon–Coles baseline + walk-forward evaluation.
4. Fixtures adapter, prediction snapshot job and paper settlement.
5. Odds-based accumulator optimiser + Streamlit dashboard.

**Start coding immediately with ingestion and baselines.** If a provider key is unavailable, implement its adapter interface and test with recorded fixtures; do not block the historical pipeline. Document assumptions and licensing/coverage limitations instead of inventing data.
