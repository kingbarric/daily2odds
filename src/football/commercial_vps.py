"""Push only explicitly public caches to the owner's configured VPS over SSH."""
import json
import subprocess
from pathlib import Path
CONFIG=Path(__file__).resolve().parents[2]/'data/football/commercial/vps_publish.json'

def publish_cache(payload,key):
    if key not in ('forecasts','prices','history'):raise ValueError('Unknown public cache')
    if not CONFIG.exists():return
    config=json.loads(CONFIG.read_text())
    subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i',config['identity'],config['destination'],'docker exec -i daily2odds-web node ops/cache-import.mjs '+key],input=json.dumps(payload,allow_nan=False),text=True,capture_output=True,check=True,timeout=90)
