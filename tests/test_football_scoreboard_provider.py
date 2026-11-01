from datetime import datetime,timezone,timedelta
from copy import deepcopy
import pytest
from src.football.scoreboard_provider import ScoreboardCSV
from src.football.public_provider import PublicCSV
from src.football.providers import ProviderError

NOW=datetime(2026,10,28,12,tzinfo=timezone.utc)
def event(league='700',completed=False):
    return dict(id='1',uid=f's:600~l:{league}~e:1',date='2026-10-29T20:00Z',
        competitions=[dict(status=dict(period=2,type=dict(name='STATUS_FULL_TIME' if completed else 'STATUS_SCHEDULED',state='post' if completed else 'pre',completed=completed)),
            competitors=[dict(homeAway='home',score='2',team={'displayName':'Manchester United','name':'Manchester United'}),
                         dict(homeAway='away',score='1',team={'displayName':'Fulham','name':'Fulham'})])])
class Response:
    def __init__(self,p):self.p=p
    def raise_for_status(self):pass
    def json(self):return self.p
class Session:
    def __init__(self,events):self.events=events;self.calls=0
    def get(self,*args,**kwargs):self.calls+=1;return Response({'events':self.events})
class Quotes:
    latest_metadata={}
    def fixtures(self,*args):return []
    def odds(self,*args):return []

def provider(events):
    return ScoreboardCSV(session=Session(events),clock=lambda:NOW,canonical_teams={'Man United','Fulham'},public_provider=Quotes())


def test_explicit_league_and_team_alias_and_no_invented_odds():
    p=provider([event(),event('99999')]);f=p.fixtures(NOW.date(),NOW.date()+timedelta(days=1))
    assert len(f)==1 and f[0]['league']=='E0' and f[0]['home']=='Man United'
    assert f[0]['provider_id']==PublicCSV._id('E0',2026,'Man United','Fulham')
    assert p.odds(f[0]['provider_id'])==[]
    # Each date is fetched only once in the bounded five-minute cache.
    p.fixtures(NOW.date(),NOW.date()+timedelta(days=1))
    assert p.session.calls==2


def test_unknown_team_is_skipped_and_reported():
    e=event();e['competitions'][0]['competitors'][0]['team']['displayName']='Unknown club'
    e['competitions'][0]['competitors'][0]['team']['name']='Unknown club'
    p=provider([e])
    with pytest.warns(UserWarning,match='alias'):
        assert p.fixtures(NOW.date(),NOW.date())==[]
    assert p.latest_metadata['skipped_unmapped_teams']==['Unknown club']


def test_regulation_settlement_and_extra_time_without_period_scores_withheld():
    p=provider([event(completed=True)])
    f=p._normalise(event(completed=True));assert f['hg']==2 and f['ag']==1
    e=event(completed=True);e['competitions'][0]['status']['period']=4
    with pytest.warns(UserWarning,match='Regulation'):
        assert p._normalise(e) is None
    for c in e['competitions'][0]['competitors']:
        c['score']='9';c['linescores']=[{'period':1,'value':1},{'period':2,'value':0},{'period':3,'value':8}]
    f=p._normalise(e);assert f['hg']==1 and f['ag']==1


def test_truncated_scoreboard_is_not_reported_as_complete():
    p=provider([event()]*1000)
    with pytest.raises(ProviderError,match='Incomplete'):
        p.fixtures(NOW.date(),NOW.date())
