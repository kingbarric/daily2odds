"""Scheduled owner-side bookmaker cache collection. Never invoked by site visitors.

Four London-time slots rotate 20 configured competitions at 06:00,10:00,14:00,18:00.
1X2 plus rotating totals cost at most 28 credits/day; the shared 30/day cap remains.
Original model predictions and the historical database are never modified.
"""
import json
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import os
from src.football.db import DB_PATH
from src.football.odds_api import OddsAPI,SPORTS
from src.football import odds_api as odds_module
from src.football.commercial_leagues import LEAGUES as COMMERCIAL_LEAGUES,TEAM_ALIASES
from src.football.providers import ProviderError

ROOT=Path(__file__).resolve().parent
SITE=ROOT/'sites/touchline-studio'
GROUPS=[['E0','E1','E2','E3','SP1'],['D1','I1','F1','N1','P1'],['D2','SP2','I2','F2','SC0'],['SC1','SC2','B1','T1','G1']]
SUPPORTED_SPORTS={**SPORTS,'soccer_germany_bundesliga2':'D2','soccer_spain_segunda_division':'SP2','soccer_italy_serie_b':'I2','soccer_france_ligue_two':'F2','soccer_spl':'SC0','soccer_belgium_first_div':'B1','soccer_turkey_super_league':'T1','soccer_greece_super_league':'G1'}


def main():
    now=datetime.now(timezone.utc);local=now.astimezone(ZoneInfo('Europe/London'))
    slot=min(3,max(0,(local.hour-6)//4));stamp=f'{local.date()}:{slot}'
    state=DB_PATH.parent/'touchline_collection.sqlite'
    with sqlite3.connect(state,timeout=30) as c:
        c.execute('CREATE TABLE IF NOT EXISTS slots(slot TEXT PRIMARY KEY,status TEXT,updated TEXT)')
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT status,updated FROM slots WHERE slot=?',(stamp,)).fetchone()
        if row and (row[0]=='completed' or (now-datetime.fromisoformat(row[1])).total_seconds()<900):
            print('This collection slot has already run or is active; using saved prices.');return
        c.execute('INSERT OR REPLACE INTO slots VALUES (?,?,?)',(stamp,'running',now.isoformat()))
    if not os.getenv('ODDS_API_KEY'):raise ValueError('Odds provider key is not configured')
    history=DB_PATH.parent/'commercial/football.db'
    if not history.exists():history=DB_PATH
    with sqlite3.connect(f'file:{history.resolve()}?mode=ro',uri=True) as c:names={r[0] for r in c.execute('SELECT name FROM teams')}
    api=OddsAPI(os.environ['ODDS_API_KEY'],canonical_teams=names,snapshot_dir=DB_PATH.parent/'odds_api/snapshots',budget_db=DB_PATH.parent/'odds_api/usage.db',cycle_budget=7)
    path=SITE/'data/prices.json';cache=json.loads(path.read_text()) if path.exists() else {'matches':{}}
    updated=[];messages=[]
    totals={GROUPS[slot][(local.day+i)%len(GROUPS[slot])] for i in range(2)}
    original_sports=dict(odds_module.SPORTS);original_aliases=dict(odds_module.ALIASES_ODDS)
    odds_module.SPORTS.update(SUPPORTED_SPORTS);odds_module.ALIASES_ODDS.update(TEAM_ALIASES)
    try:
        for sport,league in SUPPORTED_SPORTS.items():
            if league not in GROUPS[slot]:continue
            try:
                payload,captured=api._get(f'/sports/{sport}/odds',dict(regions='uk',markets='h2h,totals' if league in totals else 'h2h',oddsFormat='decimal',dateFormat='iso',commenceTimeFrom=now.strftime('%Y-%m-%dT%H:%M:%SZ'),commenceTimeTo=(now+timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')),2 if league in totals else 1)
                for event in payload:
                    fixture,quotes=api.normalise(event,captured)
                    if not fixture:continue
                    best={}
                    for q in quotes:
                        key=q['market']+':'+q['selection']
                        if key not in best or q['odds']>best[key]['odds']:
                            best[key]=dict(odds=q['odds'],bookmaker=q['bookmaker'],capturedAt=q['captured_at_utc'],sourceUpdatedAt=q['source_updated_at'],kickoff=fixture['kickoff_utc'])
                    # Replace this fixture's collected markets; retain original source times.
                    cache['matches'].setdefault(fixture['provider_id'],{}).update(best)
                updated.append(league)
            except ProviderError:
                messages.append(f'{league}: collection unavailable or shared budget exhausted; previous cache retained')
    finally:
        odds_module.SPORTS.clear();odds_module.SPORTS.update(original_sports)
        odds_module.ALIASES_ODDS.clear();odds_module.ALIASES_ODDS.update(original_aliases)
    if updated:
        cache.update(capturedAt=now.isoformat(),coverage='Rolling scheduled UK bookmaker cache: match result (1X2) across 18 supported competitions and rotating total-goal markets. Twenty competitions have model forecasts; Scottish Championship/League One have no current odds-provider support. Original per-price timestamps retained.',provider='The Odds API / UK bookmakers',quota=api.quota)
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(cache,indent=2));temporary.replace(path)
    # Bounded chunks avoid D1's SQL-size limit as the 20-league cache grows.
    from src.football.commercial_nightly import mirror_payload
    mirror_payload(cache,'prices',base=DB_PATH.parent,site=SITE)
    with sqlite3.connect(state) as c:c.execute('UPDATE slots SET status=?,updated=? WHERE slot=?',('completed',now.isoformat(),stamp))
    print(json.dumps(dict(slot=stamp,refreshed_competitions=updated,messages=messages,source='scheduled saved cache only')))

if __name__=='__main__':main()
