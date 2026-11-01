from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import json
import sqlite3
import pandas as pd
import pytest
from src.football.app_results import payout,check_results,overlay
from src.football.forward_store import ForwardStore


def test_payout_arithmetic_and_invalid_selections():
    rows=[dict(provider_id='a',odds=1.5),dict(provider_id='b',odds=2)]
    result=payout(rows,50)
    assert result['payout']==Decimal('150.00') and result['profit']==100
    assert payout([],50)['payout']==0
    for odds in (None,float('nan'),1,0,float('inf')):
        with pytest.raises((ValueError,ArithmeticError)):payout([dict(provider_id='a',odds=odds)],50)
    with pytest.raises(ValueError,match='one selection'):payout(rows+[rows[0]],50)
    with pytest.raises(ValueError):payout(rows,-1)


def test_check_adds_final_score_without_changing_forecast(tmp_path):
    now=datetime(2026,10,29,22,tzinfo=timezone.utc)
    fixture=dict(provider_id='fd:E0:2026:A|B',league='E0',season=2026,home='A',away='B',kickoff_utc='2026-10-29T18:00:00+00:00',status='scheduled',hg=None,ag=None)
    future=dict(fixture,provider_id='fd:E0:2026:A|C',away='C',kickoff_utc='2026-10-30T18:00:00+00:00')
    store=ForwardStore(tmp_path/'paper.db');store.upsert_fixtures([fixture,future],datetime(2026,10,29,12,tzinfo=timezone.utc));store.close()
    history=tmp_path/'history.db'
    with sqlite3.connect(history) as c:
        c.execute('CREATE TABLE teams(name TEXT)');c.executemany('INSERT INTO teams VALUES (?)',[('A',),('B',),('C',)])
    frame=pd.DataFrame([dict(provider_id=f['provider_id'],market='ou_1.5',selection='over',probability=.8,odds=1.5,actual_score='Pending') for f in (fixture,future)])
    path=tmp_path/'predictions.csv';frame.to_csv(path,index=False);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    queried=[]
    class Provider:
        def __init__(self,**kwargs):pass
        def results(self,ids):
            queried.extend(ids)
            return [dict(fixture,status='finished',hg=2,ag=1)]
    result=check_results(tmp_path,provider_factory=Provider,clock=lambda:now,history_path=history)
    assert queried==[fixture['provider_id']]
    assert result['finished_matches']==1
    joined=overlay(frame,tmp_path)
    assert joined.iloc[0].actual_score=='2–1' and joined.iloc[0].selection_won is True
    assert joined.iloc[1].actual_score=='Pending'
    assert joined.probability.tolist()==[.8,.8]
    check_results(tmp_path,provider_factory=Provider,clock=lambda:now,history_path=history)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    with sqlite3.connect(tmp_path/'actual_results.sqlite') as c:assert c.execute('SELECT count(*) FROM checks').fetchone()[0]==2
