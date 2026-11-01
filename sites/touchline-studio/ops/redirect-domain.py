"""Run on the existing VPS to redirect the owner's newly registered domain."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
path=Path.home()/'truearena/infra/Caddyfile.prod'
original=path.read_text()
if 'daily2odds.com, www.daily2odds.com {' in original:
    print('Domain redirect already configured');raise SystemExit(0)
block='''
# Additional registered domain; preserve paths and query strings.
daily2odds.com, www.daily2odds.com {
    redir https://daily2odds.playhuud.com{uri} permanent
}
'''
candidate=original+block
subprocess.run(['docker','exec','-i','truearena-prod-caddy-1','caddy','validate','--config','-','--adapter','caddyfile'],input=candidate,text=True,check=True)
backup=path.with_name('Caddyfile.prod.before-daily2odds-domain-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
backup.write_text(original)
try:
    path.write_text(candidate)
    subprocess.run(['docker','exec','truearena-prod-caddy-1','caddy','reload','--config','/etc/caddy/Caddyfile','--adapter','caddyfile'],check=True)
except BaseException:
    path.write_text(original)
    subprocess.run(['docker','exec','truearena-prod-caddy-1','caddy','reload','--config','/etc/caddy/Caddyfile','--adapter','caddyfile'],check=True)
    raise
print('Redirect configured; backup:',backup)
