# Local football website

Open http://127.0.0.1:8501 in Chrome while the server is running.

To restart from the project directory:

```sh
.venv/bin/python -m streamlit run football_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

This is a local website. Keep it bound to loopback; it has no public authentication or hosting configuration. API keys remain in the existing project environment and are not displayed or saved in job configurations.

## Controls

- **Train:** choose 3, 5 or 8 seasons; optionally execute the existing LightGBM/odds comparison on development seasons 2024/25 and 2025/26. Calibration always uses a separate earlier season. Window lengths exclude that calibration season. Original historical data starts in 2016, so the early requested eight-season fold has seven actual seasons. The page reports this explicitly.
- **Predict:** choose an ordered date range, up to 32 historical days or today plus the next four days for forward forecasts. Historical jobs refit the chosen window on earlier seasons rather than applying a model trained on later outcomes. Predictions are persisted before actual results are joined. Future jobs use the selected trained artifact, current schedules and provider odds subject to the existing shared credit caps. A missing or stale price never becomes a qualifying bet. Unsupported date ranges return an explanatory job failure.
- **Explore results:** filter selection probability, league, market, dates, model expected total goals and decimal odds. Optionally show only priced selections or the highest-probability selection per fixture. Download the filtered CSV. Historical rows include actual score and whether the selection won. Saved original forecasts and the 28 October blind replay are available without a new provider request.

Dixon–Coles retains its existing recency weighting and goal models. The selectable season window changes the existing LightGBM training sample, not the goal-model architecture. Model expected goals are not shot-based observed xG. Double chance and totals/BTTS are included alongside match winners.

Jobs run in allowlisted background Python workers, with one job at a time. Closing the tab does not cancel training. Jobs persist in `data/football/app_jobs/jobs.sqlite`; results and model artifacts are in unique `reports/football_app/<job-id>/` directories. The historical database is read-only. Existing reports are preserved. Each forward job has its own experimental paper ledger; these are **not** a continuous combined bankroll. Ongoing scheduled settlement still uses the existing forward worker, rather than a browser tab. No real bets are placed.

## Interpretation

Per-selection probability is not an accumulator's joint probability. The existing optimiser enforces one selection per fixture, actual same-bookmaker prices, 4.75–5.25 total odds and uncertainty/dependence checks. With insufficient forward calibration evidence or no fresh qualifying offers it returns NO BET. Filtering the display does not rerun or change frozen tickets.

Historical prices lack verified receipt timestamps. Historical odds features and replay results are development analysis, not proof that prices were executable at the simulated time. This website does not claim a profitable strategy or independent validation.

## Saved predicted games and checking scores

Every completed prediction run is listed under **Explore results → Predicted games → Result set**. Reopen the same run and click **Check results**. This starts a background results-only job: it does not retrain, regenerate probabilities or fetch paid odds. Final regulation-time scores and won/lost outcomes are added when available; future or unavailable games stay pending. Historical runs read the original database; forward runs use the existing public scoreboard adapter and settle any existing paper tickets in that run. Source coverage and publication delays can affect availability.

Actual-result observations are stored separately in append-only `actual_results.sqlite` beside the run. Original prediction CSVs, model artifacts and immutable forecast probabilities are preserved. The displayed table and filtered CSV download include the latest saved observations when you return. The latest check time is displayed. Checking is manual via the button, not an automatic scheduled service. Legacy static reports remain preserved; the button is available on completed application prediction runs.

Tick **Select** beside a displayed row and enter **Stake (£)** to calculate a hypothetical accumulator return. The return includes stake; profit is shown separately. Example: £50 at combined decimal odds 3.00 gives £150 return and £100 profit if every leg wins. One selection per match and valid prices are required. Missing prices are never invented. Displayed prices can be historical, stale or from different bookmakers: this calculator does not guarantee an executable ticket, include fees/bonuses/void rules, place bets, or debit the paper bankroll. Checkbox selections apply to visible rows and can reset when filters/result sets change.
