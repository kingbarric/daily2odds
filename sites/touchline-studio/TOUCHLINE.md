# Touchline — separate commercial-facing analytics website

The original Python/Streamlit website, football database, reports and credentials are untouched. This project is a separate Vinext/Cloudflare Worker website with its own D1 signup database.

## Included

- Bespoke responsive interface, animated SVG stadium/pitch artwork, model-network diagram, club ticket graphics and radar-style snapshot explanation.
- Free browsing with optional email-only club registration; explicit consent, persistent D1 storage, same-origin validation, duplicate handling and a private removal link.
- Competition, market, date and leading-probability filters; 5/10/20 match views; team search; model expected-goals sorting; browser-local bookmarks.
- Transparent saved model output, market probabilities, expected goal rates and archived prices when present. Expand a card to see its date/source limitations.
- Support modal with PayPal and Apple Pay slots. No real payment is collected while links are absent. Apple Pay requires an enabled checkout provider.
- Keyboard-accessible controls, Escape dismissal, modal focus handling and reduced-motion support.

## Current data

104 fixtures from the original 29 October 2026 five-day forecast run `211aa6439fb546c39d8e537ce54ceea9` are exported in `data/forecasts.json`. These are real saved model outputs, not newly generated predictions. Source: `reports/football_app/211aa6439fb546c39d8e537ce54ceea9/predictions.csv` in the parent project.

The analysis animation describes integrity checks, filtering and retrieval of that saved output. It does not claim on-demand LightGBM training or live bookmaker freshness. The snapshot is clearly dated; past fixture results are not automatically fetched by this separate website. The site's 5/10/20 count is an informational match-view limit, not an optimised betting ticket.

Future data refresh requires a new reviewed export and rebuild/deployment. Python models and provider credentials are not shipped to the website. No public training endpoint is exposed. No betting or bookmaker account access exists.

## Support configuration

Set runtime `PAYPAL_SUPPORT_URL` to the owner's HTTPS PayPal support/payment link, and `APPLE_PAY_SUPPORT_URL` to an HTTPS checkout that supports Apple Pay. Until these are supplied, buttons accurately show “Payment link not connected yet”. Card data is handled by the payment provider, not this website.

## Signup

This is optional registration of interest, not account authentication. No confirmation or marketing emails are sent automatically. No account is needed to explore matches. Production D1 signup data is separate from the local preview database. The removal token is shown only when a new signup succeeds; duplicates do not reveal an existing user's token. Removal requires explicit POST confirmation. No public subscriber list endpoint exists.

## Development

```sh
node node_modules/typescript/bin/tsc --noEmit
node scripts/check-data.mjs
node /Users/ericbarima/.codex/plugins/cache/openai-curated-remote/sites/1.1.2/scripts/build-site.mjs
npm start -- --port 8502
```

D1 schema/migrations are in `db/schema.ts` and `drizzle/`. Migrations are applied by Sites publication; local preview needs the starter's local migration command once per pending migration.

## Validation

TypeScript checks and production build passed. Dataset checks passed. HTTP page/forecast endpoints responded 200. Signup integration checks verified required consent, persistent insertion, duplicates, cross-origin rejection and token-based removal using synthetic local-only QA data, which was removed. Browser QA verified Premier League filtering, the analysis stages, five returned match cards, model details, and mobile layout. Original site health returned 200 on port 8501.

## Publication state

Site registration succeeded (`appgprj_6ae3499aaab08191bfa1214bd8363bb6`), but no source was uploaded and no version was deployed. Automatic approval review rejected the source workflow because it would send the new site's code and locally derived 104-match forecast export to an external Sites-hosted repository without explicit payload/destination approval. Local implementation, build, preview and tests are complete. Approval is required before retrying that publication action. No credentials were written to source files.

Final responsive QA: desktop 1440×1000 and phone 390×844; filtered cards, bookmarks and expanded model details rendered correctly. Browser viewport override was reset. The separate preview is retained on port 8502 for review. Your original dashboard is retained on port 8501.

## Light probability table and scheduled cache

The main view now uses a light high-contrast table with larger text, top-25/50/100 limits, league/market/date/team/probability filters and expandable model context. Default ordering is descending model probability with one outcome per match. Bracketed likelihood bands describe estimates only: higher ≥80%, moderate ≥60%, lower <60%. They are not validated confidence intervals. No accumulator recommender is included.

Premier League 1X2 quotes were fetched from UK bookmakers on 29 October 2026 at 20:11:32 UTC/London: 10 fixtures, 537 individual offers, one credit within the existing daily cap. Genuine prices include bookmaker, source/capture time and kickoff. Missing prices remain unavailable. “Fresh prices only” requires valid source/capture ordering, a source age ≤60 minutes and a future kickoff; old prices become stale.

The owner-side `refresh_touchline_odds.py` rotates leagues at 06:00 (E0/E1/E2), 10:00 (E3/SP1/D1), 14:00 (I1/F1) and 18:00 (N1/P1), Europe/London. Standard 1X2/totals collection is at most 20 credits/day within the shared 30/day cap. It preserves cached prices/timestamps on failures or exhausted budgets and skips previously completed slots. It mirrors the JSON cache into the separate local D1 table. No source is published or uploaded.

Codex app automation “Refresh Touchline odds cache” is active at those times and needs the local host/app available. Visitor requests only read cached data. The model dataset remains the saved 104-fixture export; scheduled odds collection does not retrain models or extend fixture forecasts. The external publication approval block remains unresolved.

Validation: top-100 sorting, unique-fixture selection, league/threshold filters, missing and stale prices, likelihood boundaries, 10-league schedule coverage, credit design, failure preservation and slot idempotence tested. Five consecutive website cache reads consumed zero provider credits. Full Python suite: 315 passed, one existing warning. TypeScript checks and production build passed. Both original and commercial local servers respond successfully.

## Compact results and selected-price calculator

Default view is 20 priced forecasts in a bounded, keyboard-focusable scroll panel with sticky desktop headers. “Show 20 more” expands to 40/60/80/100 using cached data only. Separate columns show decimal odds, bookmaker and quote collection time. The preserved 614-quote archive across 103 fixtures fills genuine historical price coverage alongside the newer saved cache; archived quotes lack a verified source-update field here and cannot pass the fresh-price filter. No invented prices are used.

Probability bars animate once from zero width to their actual estimated values, with matching likelihood colours and accessible numeric meter values. Reduced-motion preference disables these animations.

Each priced outcome has an optional checkbox. Selecting it opens a non-modal floating calculator showing selected rows and the product of their displayed decimal odds. Duplicate-match selections and missing/invalid prices are blocked. Clear/remove/close/reopen controls work; filtering does not silently clear selected rows, and selected numeric prices remain fixed as checked. The panel identifies stale prices and mixed bookmakers. It performs arithmetic only and submits no bet or ticket.

Verification: TypeScript and production build passed. Tests verified the 20-row limit and calculator multiplication, empty selection, stale warnings, mixed bookmakers, invalid prices and duplicate-match guards. Browser verification: 1.10 × 1.06 = 1.166; removing 1.06 restored 1.100; clear restored zero; Show 20 more changed the view from 20 to 40; reset restored 20. The API returned 614 actual saved quote outcomes without consuming provider credits. Current forecast probabilities and the original private dashboard were not modified.
