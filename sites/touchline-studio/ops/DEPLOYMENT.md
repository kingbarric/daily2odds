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
