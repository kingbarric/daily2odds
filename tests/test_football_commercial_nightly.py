from datetime import datetime,timezone
from types import SimpleNamespace
import json
import sqlite3
import pandas as pd
import pytest
from src.football import commercial_nightly as nightly
from src.football.commercial_leagues import league_scope,LEAGUES,EXTRA,ESPN_IDS
from src.football import data


def test_commercial_registry_is_isolated_and_restored():
    before=dict(data.LEAGUES)
    with pytest.raises(RuntimeError):
        with league_scope():
            assert len(data.LEAGUES)==20 and set(EXTRA)<=set(data.LEAGUES)
            raise RuntimeError('test cleanup')
    assert data.LEAGUES==before
    assert len(LEAGUES)==20 and len(ESPN_IDS)==10


def test_nightly_training_never_uses_today_or_future_targets():
    now=datetime(2026,10,29,23,tzinfo=timezone.utc)
    frame=pd.DataFrame({'date':pd.to_datetime(['2020-01-01','2022-01-01','2026-05-01','2026-08-01','2026-10-28','2026-10-29','2026-10-30','2027-01-01'])})
    train,cal=nightly.training_masks(frame,now)
    assert not (train&cal).any()
    assert frame.loc[train,'date'].max()<frame.loc[cal,'date'].min()
    assert not train.iloc[5:].any() and not cal.iloc[5:].any()
    assert cal.iloc[4] and train.iloc[0]  # 2020 is inside the requested eight-year window
    old=pd.DataFrame({'date':pd.to_datetime(['2018-01-01'])})
    assert not nightly.training_masks(old,now)[0].any()


def test_large_snapshot_is_chunked_and_reassembled(tmp_path,monkeypatch):
    monkeypatch.setattr(nightly,'BASE',tmp_path);monkeypatch.setattr(nightly,'SITE',tmp_path)
    path=tmp_path/'cache.db'
    with sqlite3.connect(path) as c:c.execute('CREATE TABLE forecast_cache(key TEXT PRIMARY KEY,payload TEXT,updated_at TEXT)')
    statements=[]
    def execute(args,**kwargs):
        sql=open(args[-1]).read();statements.append(sql)
        assert len(sql)<100000
        with sqlite3.connect(path) as c:c.executescript(sql)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(nightly.subprocess,'run',execute)
    payload={'matches':[], 'context':"quoted ' source "*15000}
    nightly.mirror_snapshot(payload)
    with sqlite3.connect(path) as c:
        manifest=json.loads(c.execute("SELECT payload FROM forecast_cache WHERE key='forecasts'").fetchone()[0])
        raw=''.join(c.execute('SELECT payload FROM forecast_cache WHERE key=?',(f"forecasts:{manifest['version']}:{i}",)).fetchone()[0] for i in range(manifest['count']))
    assert json.loads(raw)==payload and len(statements)>2
    assert "VALUES ('forecasts'," in statements[-1]


def test_failed_chunk_import_keeps_previous_current_snapshot(tmp_path,monkeypatch):
    monkeypatch.setattr(nightly,'BASE',tmp_path);monkeypatch.setattr(nightly,'SITE',tmp_path)
    path=tmp_path/'cache.db'
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE forecast_cache(key TEXT PRIMARY KEY,payload TEXT,updated_at TEXT)')
        c.execute("INSERT INTO forecast_cache VALUES ('forecasts','previous','old')")
    calls=[]
    def fail_second(args,**kwargs):
        calls.append(open(args[-1]).read())
        return SimpleNamespace(returncode=1 if len(calls)==2 else 0)
    monkeypatch.setattr(nightly.subprocess,'run',fail_second)
    with pytest.raises(RuntimeError):nightly.mirror_snapshot({'large':'x'*120000})
    assert not any("VALUES ('forecasts'," in sql for sql in calls)
    with sqlite3.connect(path) as c:assert c.execute("SELECT payload FROM forecast_cache WHERE key='forecasts'").fetchone()[0]=='previous'


def test_unavailable_current_injuries_are_not_treated_as_full_strength(monkeypatch):
    monkeypatch.setenv('API_FOOTBALL_KEY','synthetic-test-key')
    monkeypatch.setattr(nightly.requests,'get',lambda *a,**k:SimpleNamespace(json=lambda:{'errors':{'plan':'Current season unavailable'},'response':[]}))
    news=nightly.check_news(datetime(2026,10,29,tzinfo=timezone.utc))
    assert news['status']=='unavailable_current_season' and news['applied'] is False
