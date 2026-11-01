from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from src.football.public_provider import PublicCSV
from src.football.providers import ProviderError

NOW=datetime(2026,10,28,12,tzinfo=timezone.utc)
CSV='Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,B365CH\nI1,29/10/2026,20:00,Inter,Milan,2,3.5,4,2.1,1.9\n'
class Response:
    def __init__(self,text=CSV,modified='Wed, 28 Oct 2026 11:50:00 GMT'):
        self.content=text.encode();self.headers={'Last-Modified':modified} if modified else {}
    def raise_for_status(self):pass
class Session:
    def __init__(self,response=None):self.response=response or Response()
    def get(self,*args,**kwargs):return self.response


def test_real_receipt_and_batch_publication_are_distinct_and_no_closing_odds(tmp_path):
    p=PublicCSV(session=Session(),clock=lambda:NOW,canonical_teams={'Inter','Milan'},snapshot_dir=tmp_path)
    fixtures=p.fixtures(NOW.date(),(NOW+timedelta(days=5)).date())
    assert len(fixtures)==1 and fixtures[0]['kickoff_utc']=='2026-10-29T20:00:00+00:00'
    quotes=p.odds(fixtures[0]['provider_id'])
    assert len(quotes)==3 and {q['bookmaker'] for q in quotes}=={'bet365'}
    assert all(q['captured_at_utc']==NOW.isoformat() and q['source_updated_at']!=q['captured_at_utc'] for q in quotes)
    assert len(list(tmp_path.glob('*.csv')))==1
    assert p.latest_metadata['timestamp_scope'].endswith('not individual bookmaker update')


def test_missing_publication_does_not_invent_source_update():
    p=PublicCSV(session=Session(Response(modified=None)),clock=lambda:NOW)
    f=p.fixtures(NOW.date(),(NOW+timedelta(days=5)).date())
    assert len(f)==1 and p.odds(f[0]['provider_id'])==[]


def test_future_publication_is_rejected():
    p=PublicCSV(session=Session(Response(modified='Wed, 28 Oct 2026 13:00:00 GMT')),clock=lambda:NOW)
    with pytest.raises(ProviderError,match='future'):p.fixtures(NOW.date(),NOW.date())


def test_public_result_adapter_uses_actual_scores_and_preserves_original_database():
    result='Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\nI1,29/10/2026,20:00,Inter,Milan,2,1\n'
    p=PublicCSV(session=Session(Response(result)),clock=lambda:NOW+timedelta(days=2))
    f=dict(provider_id='fd:I1:2026:Inter|Milan',league='I1',season=2026,home='Inter',away='Milan',status='scheduled',kickoff_utc='2026-10-29T20:00:00+00:00')
    p.known[f['provider_id']]=f
    assert p.results([f['provider_id']])[0]['hg']==2
    assert p.results([f['provider_id']])[0]['status']=='finished'
