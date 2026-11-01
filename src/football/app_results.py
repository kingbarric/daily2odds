"""Append-only actual results beside immutable saved predictions; arithmetic only."""
from datetime import datetime,timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sqlite3
import pandas as pd
from . import markets
from .db import DB_PATH
from .forward_store import ForwardStore
from .scoreboard_provider import ScoreboardCSV


def connect(output):
    conn=sqlite3.connect(Path(output)/'actual_results.sqlite',timeout=30)
    conn.row_factory=sqlite3.Row
    conn.execute('CREATE TABLE IF NOT EXISTS checks(id INTEGER PRIMARY KEY, checked_at TEXT, observations TEXT)')
    return conn


def overlay(frame,output):
    path=Path(output)/'actual_results.sqlite'
    if not path.exists():return frame
    result=frame.copy();latest={}
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True) as c:
        for (raw,) in c.execute('SELECT observations FROM checks ORDER BY id'):
            for record in json.loads(raw):latest[record['provider_id']]=record
    for col in ('actual_score','selection_won','match_status','results_checked_at'):
        if col not in result:result[col]=None
        result[col]=result[col].astype(object)
    for index,row in result.iterrows():
        actual=latest.get(row.provider_id)
        if not actual:continue
        result.at[index,'match_status']=actual['status']
        result.at[index,'results_checked_at']=actual['checked_at']
        if actual['status']=='finished':
            result.at[index,'actual_score']=f"{actual['hg']}–{actual['ag']}"
            result.at[index,'selection_won']=bool(markets.won(row.market,row.selection,actual['hg'],actual['ag']))
        elif actual['status'] in ('cancelled','abandoned'):
            result.at[index,'actual_score']=actual['status'].title()
            result.at[index,'selection_won']=None
    return result


def check_results(output,provider_factory=ScoreboardCSV,clock=None,history_path=DB_PATH):
    """Only outcomes/settlement are updated; no retraining or odds API requests."""
    output=Path(output);now=(clock or (lambda:datetime.now(timezone.utc)))()
    frozen=pd.read_csv(output/'predictions.csv');observations=[]
    paper=output/'paper.db'
    if paper.exists():
        store=ForwardStore(paper)
        try:
            fixtures=store.fixtures()
            with sqlite3.connect(f'file:{Path(history_path).resolve()}?mode=ro',uri=True) as c:
                names={r[0] for r in c.execute('SELECT name FROM teams')}
            # Query only matches whose recorded kickoff is due; future games remain pending.
            due=[f for f in fixtures if pd.Timestamp(f['kickoff_utc'])<=pd.Timestamp(now)]
            if due:
                provider=provider_factory(canonical_teams=names,known_fixtures=fixtures,
                    snapshot_dir=Path(history_path).parent/'forward_sources')
                incoming=provider.results([f['provider_id'] for f in due])
                known={f['provider_id']:f for f in due}
                for r in incoming:
                    old=known.get(r['provider_id'])
                    if old is None or any(r[k]!=old[k] for k in ('home','away','league')):continue
                    if r['status']=='finished' and (r.get('hg') is None or r.get('ag') is None):continue
                    # A later partial response must not erase a previously final result.
                    if old['status']=='finished' and r['status']!='finished':continue
                    observations.append({k:r[k] for k in ('provider_id','status','hg','ag')})
                    store.upsert_fixtures([r],now)
            store.settle(now)
            bankroll=store.bankroll()
            settlement=dict(checked_at=now.isoformat(),bankroll=bankroll,tickets=store.tickets())
            # Separate display cache; immutable original ticket export is retained.
            temp=output/'settlement_snapshot.tmp'
            temp.write_text(json.dumps(settlement,indent=2,default=str))
            temp.replace(output/'settlement_snapshot.json')
        finally:store.close()
    else:
        # Historical jobs already contain actuals, but can be refreshed from the RO database.
        with sqlite3.connect(f'file:{Path(history_path).resolve()}?mode=ro',uri=True) as c:
            rows=c.execute('SELECT match_id,fthg,ftag FROM matches WHERE fthg IS NOT NULL AND ftag IS NOT NULL')
            wanted=set(frozen.provider_id)
            observations=[dict(provider_id=mid,status='finished',hg=hg,ag=ag) for mid,hg,ag in rows if mid in wanted]
        bankroll=None
    for r in observations:r['checked_at']=now.isoformat()
    with connect(output) as c:
        c.execute('INSERT INTO checks(checked_at,observations) VALUES (?,?)',(now.isoformat(),json.dumps(observations)))
    refreshed=overlay(frozen,output)
    final=refreshed[refreshed.actual_score.notna()&~refreshed.actual_score.isin(['Pending','Unavailable','Cancelled','Abandoned'])] if 'actual_score' in refreshed else refreshed.iloc[:0]
    return dict(output=str(output.resolve()),checked_at=now.isoformat(),
        finished_matches=int(final.provider_id.nunique()),observations=len(observations),bankroll=bankroll,
        message='Results checked. Unfinished or unavailable matches remain pending. Original predictions unchanged.')


def payout(rows,stake):
    """Hypothetical accumulator payout using selected decimal odds, no betting action."""
    amount=Decimal(str(stake))
    if not amount.is_finite() or amount<0 or amount>Decimal('1000000'):raise ValueError('Enter a stake between £0 and £1,000,000')
    if not rows:return dict(selections=0,combined_odds=Decimal('0'),payout=Decimal('0'),profit=Decimal('0'))
    ids=[r['provider_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Select at most one selection per match')
    combined=Decimal('1')
    for r in rows:
        odd=Decimal(str(r['odds']))
        if not odd.is_finite() or odd<=1:raise ValueError('Each selected game needs an available decimal price greater than 1')
        combined*=odd
        if combined>Decimal('1e18'):raise ValueError('Combined odds exceed the calculator limit')
    gross=(amount*combined).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    return dict(selections=len(rows),combined_odds=combined,payout=gross,profit=gross-amount)
