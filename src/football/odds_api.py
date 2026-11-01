"""The Odds API v4: current UK decimal quotes and exact source timestamps.

Historical training remains football-data.co.uk. API secrets are never persisted
in snapshots, metadata or exception messages. Quota is bounded per collection and
by a persistent local UTC-day/month budget, in addition to provider headers.
"""
from datetime import datetime,timedelta,timezone
import json
import math
from pathlib import Path
import sqlite3
import uuid
import warnings
from zoneinfo import ZoneInfo

import requests

from .providers import ProviderError
from .public_provider import PublicCSV
from .scoreboard_provider import ScoreboardCSV,ALIASES
from .workflow import instant

SPORTS={'soccer_epl':'E0','soccer_efl_champ':'E1','soccer_england_league1':'E2',
        'soccer_england_league2':'E3','soccer_spain_la_liga':'SP1','soccer_germany_bundesliga':'D1',
        'soccer_italy_serie_a':'I1','soccer_france_ligue_one':'F1','soccer_netherlands_eredivisie':'N1',
        'soccer_portugal_primeira_liga':'P1'}
BOOKS={'williamhill':'william_hill','betfred_uk':'betfred','betfair_sb_uk':'betfair',
       'ladbrokes_uk':'ladbrokes','paddypower':'paddy_power','skybet':'skybet','betway':'betway',
       'betvictor':'betvictor'}
ALIASES_ODDS={**ALIASES,'Manchester City':'Man City','Leeds United':'Leeds','Newcastle United':'Newcastle',
    'West Ham United':'West Ham','Wolverhampton Wanderers':'Wolves','Brighton and Hove Albion':'Brighton',
    'Brighton & Hove Albion':'Brighton','Hull City':'Hull','Ipswich Town':'Ipswich',
    'AFC Bournemouth':'Bournemouth','Inter Milan':'Inter','AC Milan':'Milan','Napoli':'Napoli',
    'Atletico Madrid':'Ath Madrid','Athletic Bilbao':'Ath Bilbao','RCD Espanyol':'Espanol','Deportivo Alaves':'Alaves',
    'Real Betis':'Betis','Celta de Vigo':'Celta','Paris Saint Germain':'Paris SG','Paris Saint-Germain':'Paris SG',
    'Bayern Munich':'Bayern Munich','Borussia Dortmund':'Dortmund','Borussia Monchengladbach':"M'gladbach",
    'Eintracht Frankfurt':'Ein Frankfurt','FC Cologne':'FC Koln','1. FC Koln':'FC Koln','SC Freiburg':'Freiburg',
    'Bayer Leverkusen':'Leverkusen','FSV Mainz 05':'Mainz','VfB Stuttgart':'Stuttgart','VfL Wolfsburg':'Wolfsburg',
    'FC Augsburg':'Augsburg','FC St. Pauli':'St Pauli','PSV Eindhoven':'PSV Eindhoven','AZ Alkmaar':'AZ Alkmaar',
    'Sporting Lisbon':'Sp Lisbon','Sporting Braga':'Sp Braga','Vitoria Guimaraes':'Guimaraes',
    'Academico Viseu':'Academico Viseu'}

# Provider-specific literal aliases, checked against the existing canonical names.
ALIASES_ODDS.update({
 'Coventry City':'Coventry','Wrexham AFC':'Wrexham','Birmingham City':'Birmingham','Stoke City':'Stoke',
 'Preston North End':'Preston','Swansea City':'Swansea','Blackburn Rovers':'Blackburn','Bolton Wanderers':'Bolton',
 'Cardiff City':'Cardiff','Charlton Athletic':'Charlton','Derby County':'Derby','Queens Park Rangers':'QPR',
 'West Bromwich Albion':'West Brom','Bradford City':'Bradford','Huddersfield Town':'Huddersfield',
 'Cambridge United':'Cambridge','Plymouth Argyle':'Plymouth','Wigan Athletic':'Wigan','Bromley FC':'Bromley',
 'Wycombe Wanderers':'Wycombe','Doncaster Rovers':'Doncaster','Leicester City':'Leicester','Mansfield Town':'Mansfield',
 'Stockport County FC':'Stockport','Wimbledon':'AFC Wimbledon','Chesterfield FC':'Chesterfield','Grimsby Town':'Grimsby',
 'Accrington Stanley':'Accrington','Northampton Town':'Northampton','Crewe Alexandra':'Crewe','Tranmere Rovers':'Tranmere',
 'Cheltenham Town':'Cheltenham','Colchester United':'Colchester','Oldham Athletic':'Oldham','Exeter City':'Exeter',
 'Shrewsbury Town':'Shrewsbury','Swindon Town':'Swindon','Rotherham United':'Rotherham',
 'Real Racing Club de Santander':'Santander','Elche CF':'Elche','Deportivo La Coruña':'La Coruna','CA Osasuna':'Osasuna',
 'SC Paderborn':'Paderborn','FC Schalke 04':'Schalke 04','TSG Hoffenheim':'Hoffenheim','1. FC Köln':'FC Koln',
 'Hamburger SV':'Hamburg','Atalanta BC':'Atalanta','RC Lens':'Lens','Le Mans FC':'Le Mans','AS Monaco':'Monaco',
 'SC Cambuur':'Cambuur','FC Zwolle':'Zwolle','FC Twente Enschede':'Twente','SC Telstar':'Telstar',
 'Moreirense FC':'Moreirense','CF Estrela':'Estrela','Rio Ave FC':'Rio Ave','CS Maritimo':'Maritimo',
 'Famalicão':'Famalicao','Vitória SC':'Guimaraes'})


class OddsAPI:
    BASE='https://api.the-odds-api.com/v4'
    def __init__(self,key,*,canonical_teams,session=None,clock=None,snapshot_dir=None,
                 budget_db=None,cycle_budget=30,daily_budget=30,monthly_budget=450):
        if not isinstance(key,str) or not key.strip():raise ValueError('ODDS_API_KEY required')
        self.key=key;self.canonical=set(canonical_teams);self.session=session or requests.Session()
        self.clock=clock or (lambda:datetime.now(timezone.utc));self.snapshot_dir=Path(snapshot_dir) if snapshot_dir else None
        self.budget_db=str(budget_db) if budget_db else None
        self.cycle_budget=cycle_budget;self.daily_budget=daily_budget;self.monthly_budget=monthly_budget
        self.reserved=0;self.quota={};self.warnings=[];self.event_ids={};self.receipts={};self.cache_hits=0
        if self.budget_db:
            Path(self.budget_db).parent.mkdir(parents=True,exist_ok=True)
            with sqlite3.connect(self.budget_db) as c:
                c.execute('CREATE TABLE IF NOT EXISTS odds_api_credit_usage(id TEXT PRIMARY KEY, day TEXT, month TEXT, reserved INTEGER, actual INTEGER)')

    def _reserve(self,cost):
        if self.reserved+cost>self.cycle_budget:raise ProviderError('The Odds API cycle credit budget exhausted')
        remaining=self.quota.get('remaining')
        if remaining is not None and remaining<cost:raise ProviderError('The Odds API remaining credits insufficient')
        token=uuid.uuid4().hex;now=instant(self.clock());day=now.date().isoformat();month=day[:7]
        if self.budget_db:
            with sqlite3.connect(self.budget_db,timeout=30) as c:
                c.execute('BEGIN IMMEDIATE')
                used_day=c.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM odds_api_credit_usage WHERE day=?',(day,)).fetchone()[0]
                used_month=c.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM odds_api_credit_usage WHERE month=?',(month,)).fetchone()[0]
                if used_day+cost>self.daily_budget or used_month+cost>self.monthly_budget:
                    raise ProviderError('The Odds API local day/month credit budget exhausted')
                c.execute('INSERT INTO odds_api_credit_usage VALUES (?,?,?,?,NULL)',(token,day,month,cost))
        self.reserved+=cost;return token

    def _get(self,path,params,cost):
        # Reuse genuine source receipts for at most 30 minutes; never relabel a
        # cached quote as newly captured. This also permits alias corrections
        # without spending the customer's credits on identical raw data.
        if cost and self.snapshot_dir and self.snapshot_dir.exists():
            ignored={'commenceTimeFrom','commenceTimeTo'}
            wanted={k:v for k,v in params.items() if k not in ignored}
            now=instant(self.clock())
            for file in sorted(self.snapshot_dir.glob('*-metadata.json'),reverse=True):
                metadata=json.loads(file.read_text())
                captured=instant(metadata['captured_at_utc'])
                if not timedelta(0)<=now-captured<=timedelta(minutes=30):continue
                previous={k:v for k,v in metadata.get('parameters',{}).items() if k not in ignored}
                data=file.with_name(file.name.replace('-metadata.json','.json'))
                if metadata.get('endpoint')==path and previous==wanted and data.is_file():
                    self.cache_hits+=1
                    return json.loads(data.read_text()),captured
        token=self._reserve(cost)
        try:
            response=self.session.get(self.BASE+path,params={**params,'apiKey':self.key},timeout=25)
            if response.status_code!=200:
                raise ProviderError(f'The Odds API HTTP {response.status_code}; no request URL logged')
            payload=response.json()
        except (requests.RequestException,ValueError):
            raise ProviderError('The Odds API request failed; URL and credentials withheld') from None
        actual=cost
        for header,label in [('x-requests-remaining','remaining'),('x-requests-used','used'),('x-requests-last','last')]:
            value=response.headers.get(header)
            if value is not None:
                try:self.quota[label]=int(value)
                except (TypeError,ValueError):pass
        actual=self.quota.get('last',cost)
        if self.budget_db:
            with sqlite3.connect(self.budget_db) as c:c.execute('UPDATE odds_api_credit_usage SET actual=? WHERE id=?',(actual,token))
        captured=instant(self.clock())
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True,exist_ok=True)
            stem=captured.strftime('%Y%m%dT%H%M%S%fZ')+'-'+token[:8]
            (self.snapshot_dir/(stem+'.json')).write_text(json.dumps(payload))
            (self.snapshot_dir/(stem+'-metadata.json')).write_text(json.dumps(dict(endpoint=path,parameters=params,captured_at_utc=captured.isoformat(),quota=self.quota),indent=2))
        return payload,captured

    def sports(self):
        payload,_=self._get('/sports',{},0)
        if not isinstance(payload,list):raise ProviderError('Invalid The Odds API sports response')
        return {r['key'] for r in payload if r.get('active') and r.get('key') in SPORTS}

    def _team(self,name):
        candidate=ALIASES_ODDS.get(name,name)
        if candidate not in self.canonical:
            self.warnings.append(f'Unmapped The Odds API team: {name}')
            return None
        return candidate

    def normalise(self,event,captured):
        sport=event.get('sport_key');league=SPORTS.get(sport)
        if not league:return None,[]
        try:
            kickoff=instant(event['commence_time'])
            if kickoff<=captured:return None,[]
            home=self._team(event['home_team']);away=self._team(event['away_team'])
            if not home or not away:return None,[]
            local=kickoff.astimezone(ZoneInfo('Europe/London'));season=local.year if local.month>=7 else local.year-1
            identifier=PublicCSV._id(league,season,home,away)
            fixture=dict(provider_id=identifier,league=league,season=season,home=home,away=away,kickoff_utc=kickoff.isoformat(),status='scheduled',hg=None,ag=None)
            self.event_ids[identifier]=(sport,str(event['id']))
            rows=[]
            for book in event.get('bookmakers',[]):
                book_key=book['key']
                # Exchange prices require commission/available-liquidity rules not
                # provided here; don't silently compare them to fixed-odds books.
                if book_key.startswith(('betfair_ex','matchbook','smarkets')):continue
                for market in book.get('markets',[]):
                    scope=market.get('key')
                    if scope not in ('h2h','h2h_3_way','totals','alternate_totals','btts','double_chance'):continue
                    updated=market.get('last_update') or book.get('last_update')
                    if not updated:continue
                    source=instant(updated)
                    if source>captured or source>=kickoff:raise ProviderError('Future/in-play source timestamp rejected')
                    outcomes=market.get('outcomes',[])
                    # 1X2 requires the draw outcome, so draw-no-bet/two-way odds
                    # cannot accidentally train or price a 90-minute 1X2 pick.
                    if scope in ('h2h','h2h_3_way') and {o.get('name') for o in outcomes}!={event['home_team'],event['away_team'],'Draw'}:continue
                    for outcome in outcomes:
                        selection=None;family=None;name=outcome.get('name')
                        if scope in ('h2h','h2h_3_way'):
                            family='1x2';selection={event['home_team']:'home',event['away_team']:'away','Draw':'draw'}.get(name)
                        elif scope in ('totals','alternate_totals'):
                            line=outcome.get('point')
                            if line in (1.5,2.5,3.5) and name in ('Over','Under'):
                                family=f'ou_{line}';selection=name.lower()
                        elif scope=='btts' and name in ('Yes','No'):
                            family='btts';selection=name.lower()
                        elif scope=='double_chance':
                            # Only literal documented outcome labels; never derive
                            # a bookmaker price from sums of model probabilities.
                            choices={f"{event['home_team']} or Draw":'home_draw',f"Draw or {event['away_team']}":'draw_away',
                                     f"{event['home_team']} or {event['away_team']}":'home_away',
                                     'Home/Draw':'home_draw','Draw/Away':'draw_away','Home/Away':'home_away',
                                     f"Draw or {event['home_team']}":'home_draw',f"{event['away_team']} or Draw":'draw_away',
                                     f"{event['away_team']} or {event['home_team']}":'home_away',
                                     '1X':'home_draw','X2':'draw_away','12':'home_away'}
                            family='double_chance';selection=choices.get(name)
                        if not selection:continue
                        price=outcome.get('price')
                        if isinstance(price,bool) or not isinstance(price,(int,float)) or not math.isfinite(price) or price<=1:
                            raise ProviderError('Invalid decimal bookmaker odds')
                        rows.append(dict(provider_id=identifier,bookmaker=BOOKS.get(book_key,'odds_api:'+book_key),
                                         market=family,selection=selection,odds=float(price),captured_at_utc=captured.isoformat(),source_updated_at=source.isoformat()))
            # Distinct markets may repeat the same line. Keep only newest quote,
            # not whichever old revision has the more favourable price.
            latest={}
            for row in rows:
                k=(row['bookmaker'],row['market'],row['selection'])
                if k not in latest or row['source_updated_at']>=latest[k]['source_updated_at']:latest[k]=row
            return fixture,list(latest.values())
        except (KeyError,ValueError,TypeError,AttributeError):
            raise ProviderError('Malformed The Odds API event; no raw request logged') from None

    def collect(self,start,end,*,extra_events=3):
        self.reserved=0;self.warnings=[];self.event_ids={}
        active=self.sports();records={};quotes={};events=[]
        now=instant(self.clock())
        for sport in SPORTS:
            if sport not in active:continue
            params=dict(regions='uk',markets='h2h,totals',oddsFormat='decimal',dateFormat='iso',
                        commenceTimeFrom=now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        commenceTimeTo=(now+timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ'))
            try:
                payload,captured=self._get(f'/sports/{sport}/odds',params,2)
            except ProviderError as exc:
                self.warnings.append(str(exc));continue
            if not isinstance(payload,list):raise ProviderError('Invalid league odds response')
            for event in payload:
                fixture,rows=self.normalise(event,captured)
                if not fixture:continue
                kickoff=instant(fixture['kickoff_utc'])
                if not start<=kickoff.date()<=end or kickoff<=now:continue
                records[fixture['provider_id']]=fixture;quotes[fixture['provider_id']]=rows;events.append((kickoff,event))
        enriched=set()
        # Reprocess all still-fresh event responses already paid for, even if an
        # alias correction changes which fixtures are now earliest in the window.
        if self.snapshot_dir and self.snapshot_dir.exists():
            for file in sorted(self.snapshot_dir.glob('*-metadata.json'),reverse=True):
                m=json.loads(file.read_text());path=m.get('endpoint','');captured=instant(m['captured_at_utc'])
                if '/events/' not in path or not timedelta(0)<=now-captured<=timedelta(minutes=30):continue
                if path in enriched:continue
                data=file.with_name(file.name.replace('-metadata.json','.json'))
                if not data.exists():continue
                fixture,rows=self.normalise(json.loads(data.read_text()),captured)
                if fixture and fixture['provider_id'] in records:
                    quotes.setdefault(fixture['provider_id'],[]).extend(rows);enriched.add(path);self.cache_hits+=1
        for _,event in sorted(events,key=lambda item:item[0])[:extra_events]:
            sport=event['sport_key'];identifier=event['id']
            if f'/sports/{sport}/events/{identifier}/odds' in enriched:continue
            try:
                payload,captured=self._get(f'/sports/{sport}/events/{identifier}/odds',
                    dict(regions='uk',markets='alternate_totals,btts,double_chance',oddsFormat='decimal',dateFormat='iso'),3)
                fixture,rows=self.normalise(payload,captured)
                if fixture:quotes.setdefault(fixture['provider_id'],[]).extend(rows)
            except ProviderError as exc:self.warnings.append(str(exc))
        for identifier,rows in quotes.items():
            latest={}
            for row in rows:
                k=(row['bookmaker'],row['market'],row['selection'])
                if k not in latest or (row['source_updated_at'],row['captured_at_utc'])>(latest[k]['source_updated_at'],latest[k]['captured_at_utc']):latest[k]=row
            quotes[identifier]=list(latest.values())
        return records,quotes


class OddsAPIProvider:
    odds_preloaded=True
    def __init__(self,api,*,schedule,extra_events=3):
        self.api=api;self.schedule=schedule;self.extra_events=extra_events;self.quotes={};self.latest_metadata={}
    def fixtures(self,start,end):
        fixtures={f['provider_id']:f for f in self.schedule.fixtures(start,end)}
        records,quotes=self.api.collect(start,end,extra_events=self.extra_events)
        self.quotes={}
        for identifier,fixture in records.items():
            if identifier not in fixtures:fixtures[identifier]=fixture
            if fixtures[identifier]['status']=='scheduled' and instant(fixtures[identifier]['kickoff_utc'])==instant(fixture['kickoff_utc']):
                self.quotes[identifier]=quotes.get(identifier,[])
        if hasattr(self.schedule,'known'):self.schedule.known.update(fixtures)
        self.latest_metadata=dict(source='The Odds API v4',region='uk',format='decimal',quota=self.api.quota,
                                  historical_source='football-data.co.uk; pre-closing timestamp verification remains explicit',
                                  fixtures_with_quotes=len(self.quotes),quote_count=sum(map(len,self.quotes.values())),
                                  extra_event_limit=self.extra_events,cache_hits=self.api.cache_hits,warnings=self.api.warnings,
                                  schedule_metadata=self.schedule.latest_metadata)
        return list(fixtures.values())
    def odds(self,identifier):return self.quotes.get(identifier,[])
    def results(self,ids):return self.schedule.results(ids)
