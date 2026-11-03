# Daily2Odds VPS deployment

Live: https://daily2odds.playhuud.com/

Porkbun A record: `daily2odds` -> `51.38.71.193`, TTL 600.

The existing OVH host runs a separate `daily2odds-web` container from `~/daily2odds/compose.yaml`. No host port is published. Caddy reaches port 3000 on the existing `truearena-edge` network. PlayHuud application services and databases are separate.

Runtime: prebuilt Vinext worker served by Node and local workerd/Miniflare, without the Wrangler development server. Persistent D1-compatible SQLite data lives in Docker volume `daily2odds_daily2odds-data`. Subscribers start in their own new database; local subscriber data was not transferred. The image carries only the compiled public site, dependencies and public cache seeds, not football databases, model artifacts, API keys or account credentials.

Caddy's pre-change backup is `~/truearena/infra/Caddyfile.prod.before-daily2odds-20261031T223339Z`. Only a new hostname block was appended after successful validation. TLS is managed by existing Caddy. `/__cache*` is blocked at Caddy. A generated server-only token protects the internal cache importer; it remains in the VPS `.env` and is never sent to browsers.

The existing local Codex scheduled jobs still run on the owner's Mac: training daily at 01:15 London; odds at 06:00, 10:00, 14:00 and 18:00 London. `src/football/commercial_vps.py` syncs only public `forecasts`, `prices`, `history` payloads using SSH and the container's internal importer. Config: `data/football/commercial/vps_publish.json`. The Mac and Codex must be available to run those jobs. The hosted website stays online when the Mac is offline, serving the last saved data. No provider calls occur on visitor searches.

To rebuild, refresh the public `seed/*.json`, build the site locally, send only Dockerfile, .dockerignore, package.json, package-lock.json, dist/, ops/, seed/ to `~/daily2odds`, then `docker compose build && docker compose up -d`. Do not transfer `.env`, `.wrangler`, local database state or private project directories. Preserve the persistent volume. Existing cache records are not overwritten by seeds on restart. Back up the volume before changing storage or schema.

Verification: HTTPS root/terms/support/forecasts/history; static CSS and JS; cache importer hidden publicly; standalone health; PlayHuud root 200 and backend health UP; public payload sync successful. No payment transaction was performed.

## Registered-domain redirect (1 November 2026)

`daily2odds.com` and `www.daily2odds.com` issue HTTPS 301 redirects to `https://daily2odds.playhuud.com{uri}`, preserving paths and query strings. Porkbun apex ALIAS points to `daily2odds.playhuud.com`; explicit `www` CNAME points to `daily2odds.com`. Existing email MX/TXT and wildcard records were preserved. Caddy backup: `~/truearena/infra/Caddyfile.prod.before-daily2odds-domain-20261101T215524Z`. Both valid HTTPS redirects were verified, including `/terms?source=check`; the original Daily2Odds and PlayHuud roots still returned 200. No application rebuild or data changes were needed.

## Continuous results and matchday board (3 November 2026)

`ops/match-feed.mjs` polls public ESPN competition scoreboards every two minutes
on the VPS, independently of the owner's Mac. Queries cover known fixtures that
are due to start or have unresolved results in the previous seven days; three
requests run at a time with ten-second timeouts. Exact league, ordered canonical
teams and kickoff must match. Only explicit completed status with verified
regulation scores settles predictions. Extra-time/penalty games need first- and
second-period scores; unavailable results remain open.

Public score updates and original forecast records persist atomically in
`/data/public-feed.json` on the existing volume. Subscriber storage is separate
and untouched. Forecast/history imports update the in-memory public feeds
immediately, preserving the earliest saved pre-kickoff probabilities. The Node
adapter serves cached JSON directly for `/api/forecasts` and `/api/history`,
avoiding repeated large-response transfers through Miniflare/workerd. Browsers
share one poll per endpoint per minute, pause background-tab polling, refresh
when refocused and retain the last good payload on failures.

The matchday board has Today, Upcoming and History views. Today's final scores
stay visible until midnight Europe/London; older matches remain in History.
New published forecasts appear without a page reload. Training/new forecast
generation still follows the existing owner-side nightly workflow; the result
poller does not create model predictions or spend odds-provider credits.

`/healthz` now probes the worker/database with a bounded timeout. A thirty-second
watchdog exits the service after two failed probes so Docker's existing restart
policy can recover it. It also recycles the service above 650 MiB cgroup usage,
below the unchanged 768 MiB hard cap. This mitigates the observed worker memory
failure; it does not claim to fix the upstream allocator.

When packaging releases, include the public `seed/scoreboard-config.json` and
`seed/archived-prices.json` as well as the three existing public payload seeds.
No provider credentials are needed for the scoreboard poller. Use a stopped,
consistent volume backup before replacing the running service; preserve the old
image tag for rollback.

## Layout pass (3 November 2026, 22:45 London)

One content grid (1200px plus a 32/24/20px gutter) now sets the left and right
edge of the header, hero, every section and the footer; a single heading scale
and card style replace the earlier per-section values. The results board counts
each game's top pick, because all 14 outcomes per game always settle exactly 7
matched. Only `dist/` changed; `ops/` and `seed/` were identical to the running
release. Release tarballs exclude `dist/server/.wrangler` (local preview
temporary files).

Rollback image: `daily2odds:before-design-20261103T224544Z` (new image tagged
`daily2odds:design-20261103`). Stopped-volume backup:
`~/daily2odds/backups/data-20261103T224544Z.tar.gz`; source backup alongside.
