"""Free publisher fixture/result fallback reusing football-data.co.uk's existing data.

Coverage is the publisher's CURRENTLY PUBLISHED fixture batch, not a guaranteed
complete rolling five-day schedule. Quotes retain actual HTTP receipt and file
Last-Modified separately. The latter is a batch publication timestamp, not proof
of an individual bookmaker price update; stale batches cannot qualify for tickets.
"""
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from io import StringIO
import json
from pathlib import Path
import uuid
import warnings

import pandas as pd
import requests

from . import data
from .providers import ProviderError
from .workflow import instant


class PublicCSV:
    FIXTURES_URL='https://www.football-data.co.uk/fixtures.csv'

    def __init__(self, *, session=None,clock=None,canonical_teams=None,known_fixtures=(),
                 snapshot_dir=None,timeout=25):
        self.session=session or requests.Session()
        self.clock=clock or (lambda:datetime.now(timezone.utc))
        self.canonical=set(canonical_teams) if canonical_teams is not None else None
        self.known={f['provider_id']:f for f in known_fixtures}
        self.snapshot_dir=Path(snapshot_dir) if snapshot_dir else None
        self.timeout=timeout;self.quotes={};self.latest_metadata={}

    def _download(self,url):
        try:
            response=self.session.get(url,timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            raise ProviderError('Public fixture/result feed request failed') from None
        captured=instant(self.clock());modified=None
        value=response.headers.get('Last-Modified')
        if value:
            try:
                modified=parsedate_to_datetime(value).astimezone(timezone.utc)
            except (TypeError,ValueError):
                raise ProviderError('Invalid public feed publication timestamp') from None
            if modified>captured:
                raise ProviderError('Public feed publication timestamp is in the future')
        raw=response.content.decode('utf-8-sig')
        self.latest_metadata=dict(source=url,captured_at_utc=captured.isoformat(),
                                  batch_published_at_utc=modified.isoformat() if modified else None,
                                  timestamp_scope='HTTP receipt and batch publication; not individual bookmaker update',
                                  coverage='publisher fixture batch; not guaranteed complete five-day schedule')
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True,exist_ok=True)
            name=captured.strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8]
            (self.snapshot_dir/(name+'.csv')).write_text(raw)
            (self.snapshot_dir/(name+'.json')).write_text(json.dumps(self.latest_metadata,indent=2))
        return raw,captured,modified

    @staticmethod
    def _id(league,season,home,away):
        return f'fd:{league}:{season}:{home}|{away}'

    def fixtures(self,start,end):
        start=pd.Timestamp(start).date();end=pd.Timestamp(end).date()
        raw,captured,modified=self._download(self.FIXTURES_URL)
        try:
            frame=pd.read_csv(StringIO(raw))
        except Exception:
            raise ProviderError('Invalid public fixture CSV') from None
        required={'Div','Date','Time','HomeTeam','AwayTeam'}
        if not required<=set(frame.columns):
            raise ProviderError('Missing public fixture columns')
        records=[];self.quotes={}
        for _,row in frame.iterrows():
            league=str(row.Div)
            if league not in data.LEAGUES:
                continue
            try:
                local=pd.to_datetime(str(row.Date)+' '+str(row.Time),dayfirst=True,errors='raise')
                kickoff=local.tz_localize('Europe/London',ambiguous='raise',nonexistent='raise').tz_convert('UTC').to_pydatetime()
            except (ValueError,TypeError):
                warnings.warn('Skipped public fixture without reliable kickoff',UserWarning);continue
            if not start<=kickoff.date()<=end or kickoff<=captured:
                continue
            home,away=data.team_name(row.HomeTeam),data.team_name(row.AwayTeam)
            if self.canonical is not None and (home not in self.canonical or away not in self.canonical):
                warnings.warn(f'Skipped unknown canonical teams: {home} / {away}',UserWarning);continue
            season=local.year if local.month>=7 else local.year-1
            identifier=self._id(league,season,home,away)
            fixture=dict(provider_id=identifier,league=league,season=season,home=home,away=away,
                         kickoff_utc=kickoff.isoformat(),status='scheduled',hg=None,ag=None)
            records.append(fixture);self.known[identifier]=fixture
            quotes=[]
            if modified is not None:
                for col in frame.columns:
                    hit=data._COL_1X2.match(col) or data._COL_OU.match(col)
                    if not hit:
                        continue
                    prefix,closing,selection=hit.groups()
                    book=data.BOOKMAKERS[prefix]
                    if closing or book in ('market_avg','market_max','betfair_exchange'):
                        continue
                    try:
                        price=float(row[col])
                    except (ValueError,TypeError):
                        continue
                    if not (1<price<1e10):
                        continue
                    quotes.append(dict(provider_id=identifier,bookmaker=book,
                                       market='1x2' if selection in 'HDA' else 'ou_2.5',
                                       selection=data._SEL[selection],odds=price,
                                       captured_at_utc=captured.isoformat(),source_updated_at=modified.isoformat()))
            self.quotes[identifier]=quotes
        return records

    def odds(self,fixture_id):
        return self.quotes.get(fixture_id,[])

    def results(self,ids):
        selected=[self.known[i] for i in ids if i in self.known]
        groups={(f['league'],int(f['season'])) for f in selected};result=[]
        for league,season in groups:
            url=data.URL.format(code=data.season_code(season),div=league)
            raw,_,_=self._download(url)
            frame=pd.read_csv(StringIO(raw))
            for fixture in selected:
                if fixture['league']!=league or int(fixture['season'])!=season:
                    continue
                match=frame[(frame.HomeTeam.map(data.team_name)==fixture['home']) &
                            (frame.AwayTeam.map(data.team_name)==fixture['away'])]
                if len(match)!=1:
                    continue
                row=match.iloc[0]
                if pd.isna(row.FTHG) or pd.isna(row.FTAG):
                    continue
                goals=[float(row.FTHG),float(row.FTAG)]
                if any(not g.is_integer() or g<0 for g in goals):
                    raise ProviderError('Invalid public regulation score')
                date=pd.to_datetime(str(row.Date),dayfirst=True)
                if pd.notna(row.get('Time')):
                    kickoff=pd.to_datetime(str(row.Date)+' '+str(row.Time),dayfirst=True).tz_localize('Europe/London').tz_convert('UTC').isoformat()
                else:
                    kickoff=fixture['kickoff_utc']
                result.append(dict(fixture,status='finished',hg=int(goals[0]),ag=int(goals[1]),kickoff_utc=kickoff))
        return result
