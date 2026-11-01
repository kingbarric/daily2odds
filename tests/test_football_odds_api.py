from datetime import datetime,timedelta,timezone
from copy import deepcopy
import json
import requests
import pytest
from src.football.odds_api import OddsAPI
from src.football.providers import ProviderError

NOW=datetime(2026,10,29,12,tzinfo=timezone.utc)

def event():
    return dict(id='event1',sport_key='soccer_epl',home_team='Chelsea',away_team='Manchester United',
        commence_time=(NOW+timedelta(days=1)).isoformat(),bookmakers=[dict(key='williamhill',title='William Hill',
        last_update=(NOW-timedelta(minutes=2)).isoformat(),markets=[
            dict(key='h2h',outcomes=[dict(name='Chelsea',price=2),dict(name='Draw',price=3.5),dict(name='Manchester United',price=4)]),
            dict(key='alternate_totals',outcomes=[dict(name='Over',point=1.5,price=1.2),dict(name='Under',point=1.5,price=4),dict(name='Under',point=3.5,price=1.4)]),
            dict(key='btts',outcomes=[dict(name='Yes',price=1.8),dict(name='No',price=2.1)]),
            dict(key='double_chance',outcomes=[dict(name='Chelsea or Draw',price=1.3),dict(name='Draw or Manchester United',price=1.9)])])])

class Response:
    status_code=200
    def __init__(self,payload):self.payload=payload;self.headers={'x-requests-remaining':'498','x-requests-used':'2','x-requests-last':'2'}
    def json(self):return self.payload
class Session:
    def __init__(self,payload):self.payload=payload
    def get(self,*args,**kwargs):return Response(self.payload)

def api(**kwargs):return OddsAPI('test-key',canonical_teams={'Chelsea','Man United'},clock=lambda:NOW,**kwargs)


def test_normalises_all_supported_markets_without_synthetic_prices():
    f,q=api().normalise(event(),NOW)
    assert f['provider_id']=='fd:E0:2026:Chelsea|Man United'
    assert {x['market'] for x in q}=={'1x2','ou_1.5','ou_3.5','btts','double_chance'}
    assert all(x['bookmaker']=='william_hill' for x in q)
    assert next(x for x in q if x['market']=='double_chance' and x['selection']=='home_draw')['odds']==1.3
    assert all(x['source_updated_at']!=x['captured_at_utc'] for x in q)


def test_no_live_odds_future_updates_two_way_or_exchange_prices():
    e=event();e['commence_time']=NOW.isoformat();assert api().normalise(e,NOW)==(None,[])
    e=event();e['bookmakers'][0]['last_update']=(NOW+timedelta(minutes=1)).isoformat()
    with pytest.raises(ProviderError,match='Future'):api().normalise(e,NOW)
    e=event();e['bookmakers'][0]['markets']=e['bookmakers'][0]['markets'][:1]
    e['bookmakers'][0]['markets'][0]['outcomes'].pop(1)
    assert api().normalise(e,NOW)[1]==[]
    e=event();e['bookmakers'][0]['key']='betfair_ex_uk';assert api().normalise(e,NOW)[1]==[]


def test_missing_timestamp_unknown_team_and_unsupported_lines_are_not_guessed():
    e=event();del e['bookmakers'][0]['last_update'];assert api().normalise(e,NOW)[1]==[]
    e=event();e['home_team']='Unknown United';assert api().normalise(e,NOW)==(None,[])
    e=event();e['bookmakers'][0]['markets']=[dict(key='totals',outcomes=[dict(name='Over',point=2.75,price=2)])]
    assert api().normalise(e,NOW)[1]==[]


def test_credit_budget_persists_and_secret_is_not_in_saved_metadata(tmp_path):
    path=tmp_path/'budget.db'
    a=api(session=Session([]),budget_db=path,daily_budget=2,snapshot_dir=tmp_path/'snapshots')
    a._get('/sports/soccer_epl/odds',dict(regions='uk',markets='h2h,totals'),2)
    b=api(session=Session([]),budget_db=path,daily_budget=2)
    with pytest.raises(ProviderError,match='budget'):b._get('/sports/soccer_epl/odds',{},2)
    assert all('test-key' not in p.read_text() for p in (tmp_path/'snapshots').glob('*.json'))


def test_network_errors_do_not_leak_credential_bearing_url():
    class Broken:
        def get(self,*args,**kwargs):raise requests.ConnectionError('https://example?apiKey=test-key')
    with pytest.raises(ProviderError) as result:api(session=Broken())._get('/sports',{},0)
    assert 'test-key' not in str(result.value)


def test_cached_quotes_keep_original_capture_time_and_do_not_spend_again(tmp_path):
    a=api(session=Session([event()]),snapshot_dir=tmp_path/'snapshots',budget_db=tmp_path/'budget.db',daily_budget=2)
    _,captured=a._get('/sports/soccer_epl/odds',dict(regions='uk',markets='h2h,totals'),2)
    b=api(session=Session([]),snapshot_dir=tmp_path/'snapshots',budget_db=tmp_path/'budget.db',daily_budget=2)
    payload,reused=b._get('/sports/soccer_epl/odds',dict(regions='uk',markets='h2h,totals'),2)
    assert reused==captured and payload[0]['id']=='event1' and b.cache_hits==1 and b.reserved==0


def test_betway_is_not_mislabelled_as_bwin():
    e=event();e['bookmakers'][0]['key']='betway'
    assert {q['bookmaker'] for q in api().normalise(e,NOW)[1]}=={'betway'}


def test_away_or_draw_label_is_the_actual_x2_quote():
    e=event();e['bookmakers'][0]['markets']=[dict(key='double_chance',outcomes=[
        dict(name='Chelsea or Draw',price=1.3),dict(name='Manchester United or Draw',price=1.9),dict(name='Chelsea or Manchester United',price=1.2)])]
    q=api().normalise(e,NOW)[1]
    assert {r['selection'] for r in q}=={'home_draw','draw_away','home_away'}
    assert next(r['odds'] for r in q if r['selection']=='draw_away')==1.9
