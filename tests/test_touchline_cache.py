import json
import sqlite3
from types import SimpleNamespace
import refresh_touchline_odds as refresh
from src.football.providers import ProviderError


def test_schedule_covers_all_leagues_within_shared_cap():
    leagues=[league for group in refresh.GROUPS for league in group]
    assert len(leagues)==len(set(leagues))==20
    assert set(leagues)==set(refresh.COMMERCIAL_LEAGUES)
    assert len(leagues)+2*len(refresh.GROUPS)==28<30


def test_budget_outage_keeps_previous_cache_and_slot_is_idempotent(tmp_path,monkeypatch):
    site=tmp_path/'site';(site/'data').mkdir(parents=True)
    previous=dict(capturedAt='2026-10-29T00:00:00+00:00',coverage='saved',matches={'old':{'1x2:home':{'odds':1.5}}})
    cache=site/'data/prices.json';cache.write_text(json.dumps(previous));before=cache.read_bytes()
    history=tmp_path/'football.db'
    with sqlite3.connect(history) as c:c.execute('CREATE TABLE teams(name TEXT)')
    monkeypatch.setattr(refresh,'SITE',site);monkeypatch.setattr(refresh,'DB_PATH',history);monkeypatch.setenv('ODDS_API_KEY','synthetic-test-value')
    calls=[]
    class NoBudget:
        quota={}
        def __init__(self,*args,**kwargs):assert kwargs['cycle_budget']==7
        def _get(self,*args,**kwargs):calls.append(args[0]);raise ProviderError('budget exhausted')
    monkeypatch.setattr(refresh,'OddsAPI',NoBudget)
    imports=[]
    monkeypatch.setattr(refresh.subprocess,'run',lambda *args,**kwargs:imports.append(args) or SimpleNamespace(returncode=0))
    refresh.main();assert calls and len(imports)>=2 and cache.read_bytes()==before
    count=len(calls);import_count=len(imports);refresh.main();assert len(calls)==count and len(imports)==import_count
    assert not (tmp_path/'touchline_cache_import.sql').exists()
