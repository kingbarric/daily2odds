"""Run on the VPS after the isolated service passes its health check."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
path=Path.home()/'truearena/infra/Caddyfile.prod'
original=path.read_text()
block='''
# Daily2Odds — isolated container; separate persistent database.
daily2odds.playhuud.com {
    encode zstd gzip
    @private path /__cache*
    respond @private 404
    request_body {
        max_size 64KB
    }
    header X-Content-Type-Options nosniff
    header Referrer-Policy strict-origin-when-cross-origin
    reverse_proxy daily2odds-web:3000
}
'''
if 'daily2odds.playhuud.com {' in original:
    print('Proxy already configured');raise SystemExit(0)
candidate=original+block
# Validate inside the existing container, with its existing environment.
subprocess.run(['docker','exec','-i','truearena-prod-caddy-1','caddy','validate','--config','-','--adapter','caddyfile'],input=candidate,text=True,check=True)
backup=path.with_name('Caddyfile.prod.before-daily2odds-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
backup.write_text(original)
try:
    # Keep the bind-mounted file's inode.
    path.write_text(candidate)
    subprocess.run(['docker','exec','truearena-prod-caddy-1','caddy','reload','--config','/etc/caddy/Caddyfile','--adapter','caddyfile'],check=True)
except BaseException:
    path.write_text(original)
    subprocess.run(['docker','exec','truearena-prod-caddy-1','caddy','reload','--config','/etc/caddy/Caddyfile','--adapter','caddyfile'],check=True)
    raise
print('Proxy configured; backup:',backup)
