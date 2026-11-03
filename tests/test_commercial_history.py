import json
from src.football.commercial_history import build_history,outcome

def test_market_results():
    assert outcome('double_chance:draw_away',0,2)
    assert not outcome('1x2:home',0,2)
    assert outcome('ou_2.5:under',0,2)
    assert not outcome('btts:yes',0,2)

def test_history_keeps_earliest_pre_match_forecast(tmp_path):
    folder=tmp_path/'reports/football_nightly';folder.mkdir(parents=True)
    def save(version,time,p):
        dest=folder/version;dest.mkdir()
        (dest/'forecasts.json').write_text(json.dumps({'matches':[dict(id='m',date='2026-10-28',home='A',away='B',league='E0',asOf=time,kickoff='2026-10-28T18:00:00Z',probabilities={'1x2:home':p})]}))
    save('a','2026-10-28T10:00:00Z',.6)
    save('b','2026-10-28T12:00:00Z',.8)
    save('c','2026-10-28T20:00:00Z',.99)
    match=build_history(tmp_path)['matches'][0]
    assert match['rows'][0]['probability']==.6
    assert match['rows'][0]['correct'] is None
    assert match['kickoff']=='2026-10-28T18:00:00Z'

def test_settlement_matches_equivalent_kickoff_timezones(tmp_path):
    import sqlite3
    folder=tmp_path/'reports/football_nightly/a';folder.mkdir(parents=True)
    match=dict(id='m',date='2026-10-28',home='A',away='B',league='E0',asOf='2026-10-28T17:30:00+01:00',kickoff='2026-10-28T18:00:00Z',probabilities={'1x2:draw':.7})
    (folder/'forecasts.json').write_text(json.dumps({'matches':[match]}))
    db=tmp_path/'data/football/commercial/observed_results.db';db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE fixtures (provider_id TEXT,kickoff_utc TEXT,hg INTEGER,ag INTEGER,status TEXT)')
        conn.execute('INSERT INTO fixtures VALUES (?,?,?,?,?)',('m','2026-10-28T18:00:00+00:00',0,0,'finished'))
    row=build_history(tmp_path)['matches'][0]['rows'][0]
    assert row['score']=='0–0' and row['correct'] is True

def test_team_context_excludes_future_results():
    import pandas as pd
    from src.football.commercial_nightly import team_context
    history=pd.DataFrame([dict(date=pd.Timestamp('2026-10-29'),home='A',away='B',hg=2,ag=1),dict(date=pd.Timestamp('2026-11-01'),home='A',away='B',hg=9,ag=0)])
    context=team_context(history,{'home':'A','away':'B'},pd.Timestamp('2026-10-30',tz='UTC'))
    assert context['home']['historicalMatches']==1
    assert context['home']['recent'][0]['scored']==2
    assert context['away']['recent'][0]['result']=='L'
