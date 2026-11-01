# Historical odds training and live multi-market odds

Historical fixtures/results and pre-closing bookmaker odds remain football-data.co.uk. The existing LightGBM now has a trained development variant that adds margin-removed market probabilities and overround to the existing past-only features. Closing prices and target-game scores are excluded from inputs. The archive's per-price timestamps remain unverified: this is development sensitivity, not verified executable-return evidence.

The trained artifact, metadata and chronological comparisons are in `reports/football_odds_training_20261029/`. For each 2024/25 and 2025/26 development prediction, booster training ends at s−2 and sigmoid calibration uses s−1. On matched price-covered rows the odds inputs improved log loss/Brier versus the original feature-only model, but did not beat the bookmaker benchmark. The final current-season artifact trains through 2024/25 and calibrates on 2025/26; current known results update rolling form/Elo and the existing Dixon–Coles goal model.

Model scope is explicit: odds-aware LightGBM supplies 1X2 and derived double chance. Existing Dixon–Coles supplies Over/Under 1.5/2.5/3.5 and BTTS from historical goal rates. Model expected goals are goal-rate estimates, not observed shot-based xG. There is no numerical averaging of the models, no new prediction algorithm, and no claim of a profitable edge.

## Reproduce

```sh
# Choose a NEW output path for another training run to preserve the saved artifact.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python football_train_odds.py \
  --output reports/football_odds_training_new_run

# Actual experiment used this trained model and a separate paper database:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python football_forward.py \
  --provider odds-api \
  --odds-model reports/football_odds_training_20261029/lightgbm_with_odds.joblib \
  --paper-db data/football/paper_odds_api_20261029_v2.db \
  --rank-by probability --stake 10

# Static probability-first report, no stakes or network calls:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python football_odds_report.py
```

The supplied `ODDS_API_KEY` is stored in `.env` and is never printed, included in saved snapshots/metadata, or exposed in credential-bearing exception URLs. `--provider auto` now prefers The Odds API when this key is configured, while explicit `scoreboard-csv`, `public-csv` and `api-football` remain available. No service/subscription was purchased. No OS background worker was left running.

## Markets and limits

League batch requests use the ten configured soccer keys, `regions=uk`, `markets=h2h,totals`, decimal prices and the next five days. Exchanges are excluded because commission/liquidity rules are not represented. Full 1X2 prices require a draw outcome; two-way/draw-no-bet prices are not silently relabelled. Only actual goal lines 1.5/2.5/3.5 are mapped. Half-time, cards, corners and unsupported markets are ignored.

Alternate totals, BTTS and double chance use event-specific requests, capped at three events per cycle. Available markets vary by bookmaker; no missing double-chance price is manufactured from probabilities. Explicit team aliases are checked against existing canonical clubs, and sportsbook quotes attach only when canonical teams, league/season and kickoff match the fixture identity. Older source times cannot be replaced by a newly fetched receipt time. Missing/future timestamps, in-play quotes and invalid prices are excluded.

Pricing requests have a 30-credit cycle cap, a persistent 30-credit UTC-day cap and a conservative 450-credit calendar-month local cap, plus provider-reported remaining-credit checks. The usage ledger is shared across paper databases in `data/football/odds_api/usage.db`. Unknown/failed requests retain their reserved cost conservatively. The zero-cost sports endpoint verifies current coverage/quota. No automatic retries or paid plan changes occur.

Genuine raw odds responses can be reused for 30 minutes to avoid spending credits on repeated cycles or alias corrections. Reuse preserves the ORIGINAL capture timestamp and provider market/bookmaker update timestamp. A narrower cached query window may have no prices for newly entering fixtures; it is never labelled complete five-day quote coverage. The final live report records actual source coverage and API warnings. Local day/month caps can withhold additional event requests even while the provider still has unused credits.

The first paid test used 2 credits. The full initial collection used another 29; reprocessing the archived responses after alias corrections used no additional paid credits. The last provider quota check reported **31 used / 469 remaining**. Continuous all-league odds polling requires managing this finite allowance; the default budget fails closed rather than consuming the entire account silently.

## Probability-first analysis

`--rank-by probability` prioritises conservative joint probability, then nominal joint probability, among tickets that still satisfy the 4.75–5.25 odds and quality/EV rules. The existing EV-first mode remains available with `--rank-by ev`. Search is bounded, so global optimality is not guaranteed. Within a ticket every leg is a different match and all prices come from one bookmaker.

The report separately shows:

- Highest estimated-probability selections with the best real available price for that same event/selection.
- High-probability analysis candidates that also have positive nominal model EV. These do not bypass the uncertainty gate and are not represented as approved bets.
- Breakdowns for double chance, totals, BTTS and 1X2, plus reproducible model expected goals.
- All current forecasts and per-market CSV downloads.

Higher probability does not automatically mean fair odds. Many very likely outcomes are priced too low to offer positive model EV. The latest experimental cycle saved 104 new forecasts, matched prices on 103 fixtures and found 6,355 fresh quote candidates. It returned **NO BET** because prospective calibration evidence for this experimental strategy is insufficient; no £10 stakes were forced. The £1,000 paper balance is unchanged.

This is a NEW experiment, kept in its own paper database, not a revision of earlier frozen forecasts or a continuation falsely labelled as independent validation of the older strategy. Earlier databases and reports remain intact. See the training metadata, current source/quota summary and test log alongside the HTML report for evidence.

Final implementation audit corrected Betway’s bookmaker identity (it is not Bwin) and the provider’s literal away-team-or-draw X2 labels. The final run uses paper_odds_api_20261029_v2.db; earlier draft experiments are retained for audit and are not used for the current report.
