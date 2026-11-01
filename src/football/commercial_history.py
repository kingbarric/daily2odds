"""Publish saved forecast history without fitting models or rewriting forecasts."""
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def outcome(key,h,a):
    market,selection=key.split(':')
    if market=='1x2':return selection==('home' if h>a else 'away' if a>h else 'draw')
    if market=='double_chance':return {'home_draw':h>=a,'draw_away':a>=h,'home_away':h!=a}[selection]
    if market=='btts':return (h>0 and a>0)==(selection=='yes')
    if market.startswith('ou_'):return (h+a>float(market[3:]))==(selection=='over')
    return None

def build_history(root=ROOT):
    root=Path(root);records={};results={}
    db=root/'data/football/commercial/observed_results.db'
    if db.exists():
        with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as conn:
            for ident,kickoff,h,a in conn.execute("SELECT provider_id,kickoff_utc,hg,ag FROM fixtures WHERE status='finished'"):
                results[(ident,kickoff)]=(h,a)
    # Preserve the earliest actually saved pre-kickoff forecast for each match.
    for path in sorted((root/'reports/football_nightly').glob('*/forecasts.json')):
        data=json.loads(path.read_text())
        for m in data['matches']:
            if not m.get('kickoff') or m['asOf']>=m['kickoff']:continue
            ident=m['id'];prior=records.get(ident)
            if prior and prior['asOf']<=m['asOf']:continue
            score=results.get((ident,m['kickoff']))
            records[ident]=dict(id=ident,date=m['date'],home=m['home'],away=m['away'],league=m['league'],asOf=m['asOf'],kind='Saved pre-match forecast',rows=[dict(market=k,probability=p,score=f'{score[0]}–{score[1]}' if score else None,correct=outcome(k,*score) if score else None) for k,p in m['probabilities'].items()])
    # Existing result-blind replay is explicitly separate from genuine forward history.
    replay=root/'reports/football_blind_replay_20261028/selected_markets_with_actuals.csv'
    if replay.exists():
        for r in csv.DictReader(replay.open()):
            if r['mode']!='strict_live_freshness':continue
            ident='replay:'+r['provider_id'];entry=records.setdefault(ident,dict(id=ident,date=r['kickoff_utc'][:10],home=r['home'],away=r['away'],league=r['provider_id'].split(':')[1],asOf=r['simulated_as_of_utc'],kind='Retrospective result-blind replay',rows=[]))
            entry['rows'].append(dict(market=r['market']+':'+r['selection'],probability=float(r['probability']),score=r['actual_score'] or None,correct={'True':True,'False':False}.get(r['selection_won'])))
    return dict(updatedAt=datetime.now(timezone.utc).isoformat(),matches=sorted(records.values(),key=lambda m:(m['date'],m['id']),reverse=True))

def publish_history():
    from .commercial_nightly import mirror_payload,atomic_json,SITE
    payload=build_history();atomic_json(SITE/'data/history.json',payload);mirror_payload(payload,'history');return payload
