# Light forecast-first redesign

- Production build and TypeScript checks passed.
- Forecast ranking checks passed: top-100 descending probability, one outcome per fixture, competition/threshold/team filters, fresh-vs-stale price rules and likelihood bands.
- Full Python suite: 315 passed, one existing joblib warning, 23.88s.
- Actual provider collection: 10 Premier League fixtures, 537 quotes, 1 credit, 29 October 2026 at 20:11:32 London/UTC. Current shared daily usage 30, provider balance 468.
- Five consecutive website reads consumed zero provider credits.
- Scheduled updater tested against the exhausted budget: previous prices retained, cache imported into separate local D1, repeated collection slot skipped. Daily planned cost 20 credits within existing shared cap of 30.
- Codex app schedule active: 06:00 / 10:00 / 14:00 / 18:00 Europe/London, id refresh-touchline-odds-cache. Local host/app must be available. This updates price caches, not the original model forecast export or a hosted deployment.
- Desktop and mobile browser checks showed high-contrast light controls, ranked probability cards and genuine labelled bookmaker prices; temporary viewport override reset.
- Original dashboard health 200; all nine historical data/report preservation hashes unchanged.
- Publication not retried; existing explicit approval requirement still applies. No betting execution or accumulator recommendations added.
