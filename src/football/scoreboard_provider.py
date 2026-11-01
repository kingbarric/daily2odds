"""Public daily schedule coverage plus the existing publisher's real quoted prices.

ESPN's public scoreboard is an unauthenticated upstream interface, not a paid
coverage/SLA guarantee. League UIDs are explicitly mapped; cup events cannot be
inferred from team names. Team aliases are explicit, never fuzzy matched.
Both sources share canonical fixture IDs to prevent duplicate match exposure.
"""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import re
import uuid
import warnings
from zoneinfo import ZoneInfo

import requests

from .public_provider import PublicCSV
from .providers import ProviderError
from .workflow import instant

LEAGUE_IDS={'700':'E0','3914':'E1','3915':'E2','3916':'E3','740':'SP1','720':'D1','730':'I1','710':'F1','725':'N1','715':'P1'}
ALIASES={
    'Manchester United':'Man United','Nottingham Forest':"Nott'm Forest",'Tottenham Hotspur':'Tottenham',
    'Bristol Rovers':'Bristol Rvs','Burton Albion':'Burton','Lincoln City':'Lincoln','Norwich City':'Norwich',
    'Oxford United':'Oxford','Peterborough United':'Peterboro','Salford City':'Salford',
    'Sheffield Wednesday':'Sheffield Weds','York City':'York',
    'AS Roma':'Roma','Internazionale':'Inter','Alavés':'Alaves','Athletic Club':'Ath Bilbao',
    'Atlético Madrid':'Ath Madrid','Celta Vigo':'Celta','Deportivo':'La Coruna','Espanyol':'Espanol',
    'Málaga':'Malaga','Racing Santander':'Santander','Rayo Vallecano':'Vallecano','Real Sociedad':'Sociedad',
    'Borussia Mönchengladbach':"M'gladbach",'Eintracht Frankfurt':'Ein Frankfurt','FC Cologne':'FC Koln',
    'Le Havre AC':'Le Havre','Paris Saint-Germain':'Paris SG','ADO Den Haag':'Den Haag',
    'FC Groningen':'Groningen','FC Twente':'Twente','FC Utrecht':'Utrecht','Fortuna Sittard':'For Sittard',
    'NEC Nijmegen':'Nijmegen','PEC Zwolle':'Zwolle','Académico de Viseu':'Academico Viseu',
    'Braga':'Sp Braga','FC Famalicao':'Famalicao','FC Porto':'Porto','Sporting CP':'Sp Lisbon',
    'Vitória de Guimaraes':'Guimaraes',
}


class ScoreboardCSV:
    URL='https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard'

    def __init__(self,*,session=None,clock=None,canonical_teams=None,known_fixtures=(),snapshot_dir=None,public_provider=None):
        self.session=session or requests.Session();self.clock=clock or (lambda:datetime.now(timezone.utc))
        self.canonical=set(canonical_teams) if canonical_teams is not None else set()
        self.known={f['provider_id']:f for f in known_fixtures}
        self.snapshot_dir=Path(snapshot_dir) if snapshot_dir else None
        self.public=public_provider or PublicCSV(session=self.session,clock=self.clock,canonical_teams=canonical_teams,
                                               known_fixtures=known_fixtures,snapshot_dir=snapshot_dir)
        self.cache={};self.latest_metadata={};self.quote_ids={};self.skipped=set()

    def _payload(self,day):
        day=str(day);now=instant(self.clock())
        if day in self.cache and now-self.cache[day][0]<=timedelta(minutes=5):
            return self.cache[day][1]
        try:
            response=self.session.get(self.URL,params={'dates':day.replace('-',''),'limit':1000},timeout=25)
            response.raise_for_status();payload=response.json()
        except (requests.RequestException,ValueError):
            raise ProviderError('Public daily scoreboard unavailable') from None
        if not isinstance(payload,dict) or not isinstance(payload.get('events'),list) or len(payload['events'])>=1000:
            raise ProviderError('Incomplete or invalid public scoreboard batch')
        captured=instant(self.clock());self.cache[day]=(captured,payload)
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True,exist_ok=True)
            name=captured.strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8]
            (self.snapshot_dir/(name+'-scoreboard.json')).write_text(json.dumps(payload))
            (self.snapshot_dir/(name+'-metadata.json')).write_text(json.dumps(dict(source=self.URL,day=day,captured_at_utc=captured.isoformat())))
        return payload

    def _team(self,team):
        label=team.get('displayName','')
        if label in ALIASES and ALIASES[label] in self.canonical:
            return ALIASES[label]
        candidates={team.get(k,'').strip() for k in ('name','displayName','shortDisplayName','location')}&self.canonical
        if len(candidates)==1:
            return candidates.pop()
        self.skipped.add(label)
        warnings.warn(f'No unambiguous historical alias for scoreboard team {label!r}; fixture skipped',UserWarning)
        return None

    def _normalise(self,event):
        uid=re.search(r'~l:(\d+)(?:~|$)',event.get('uid',''))
        if not uid or uid[1] not in LEAGUE_IDS:
            return None
        league=LEAGUE_IDS[uid[1]]
        competitions=event.get('competitions',[])
        if len(competitions)!=1:
            return None
        game=competitions[0]
        if game.get('timeValid') is False:
            warnings.warn('Scoreboard fixture has no confirmed kickoff time; skipped',UserWarning)
            return None
        teams={c['homeAway']:c for c in game.get('competitors',[]) if c.get('homeAway') in ('home','away')}
        if set(teams)!={'home','away'}:
            return None
        home=self._team(teams['home']['team']);away=self._team(teams['away']['team'])
        if not home or not away:return None
        kickoff=instant(event['date'])
        local=kickoff.astimezone(ZoneInfo('Europe/London'))
        season=local.year if local.month>=7 else local.year-1
        info=game.get('status',event.get('status',{}));typ=info.get('type',{})
        name=str(typ.get('name',''));state=typ.get('state')
        hg=ag=None
        if 'POSTPON' in name or 'SUSPEND' in name:status='postponed'
        elif 'CANCEL' in name:status='cancelled'
        elif 'ABANDON' in name:status='abandoned'
        elif typ.get('completed'):
            status='finished'
            extra=info.get('period',2)>2 or re.search(r'\b(AET|ET|PEN)\b|extra|penalt',name+' '+str(typ.get('detail',''))+' '+str(typ.get('shortDetail','')),re.I)
            goals=[]
            for side in ('home','away'):
                c=teams[side]
                if extra:
                    lines={int(x.get('period',0)):x.get('value') for x in c.get('linescores',[])}
                    if 1 not in lines or 2 not in lines:
                        warnings.warn('Regulation score unavailable for extra-time fixture; settlement withheld',UserWarning);return None
                    value=float(lines[1])+float(lines[2])
                else:
                    value=float(c['score'])
                if value<0 or not value.is_integer():raise ProviderError('Invalid scoreboard regulation goals')
                goals.append(int(value))
            hg,ag=goals
        elif state=='in':status='live'
        elif state=='pre':status='scheduled'
        else:return None
        return dict(provider_id=PublicCSV._id(league,season,home,away),league=league,season=season,
                    home=home,away=away,kickoff_utc=kickoff.isoformat(),status=status,hg=hg,ag=ag)

    def _records(self,day):
        result=[]
        try:
            for e in self._payload(day)['events']:
                row=self._normalise(e)
                if row:result.append(row)
        except (KeyError,ValueError,TypeError,AttributeError):
            raise ProviderError('Malformed public scoreboard event') from None
        return result

    def fixtures(self,start,end):
        start=datetime.fromisoformat(str(start)).date();end=datetime.fromisoformat(str(end)).date()
        days=(end-start).days+1
        if not 1<=days<=7:raise ValueError('scoreboard date window must be 1–7 days')
        records={}
        for n in range(days):
            for row in self._records((start+timedelta(days=n)).isoformat()):
                kickoff=instant(row['kickoff_utc'])
                if start<=kickoff.date()<=end and kickoff>instant(self.clock()):records[row['provider_id']]=row
        # Fresh publisher prices must match the same canonical teams AND kickoff.
        try:
            public=self.public.fixtures(start,end)
        except ProviderError:
            warnings.warn('Publisher prices unavailable; schedule forecasts continue with NO BET',UserWarning);public=[]
        self.quote_ids={}
        for row in public:
            identifier=row['provider_id']
            if identifier not in records:records[identifier]=row
            if instant(records[identifier]['kickoff_utc'])==instant(row['kickoff_utc']):self.quote_ids[identifier]=identifier
        self.known.update(records)
        self.latest_metadata=dict(source=self.URL,captured_at_utc=instant(self.clock()).isoformat(),
                                  queried_utc_days=days,configured_leagues=list(LEAGUE_IDS.values()),
                                  source_completeness='all returned configured-league events; upstream completeness not guaranteed',
                                  skipped_unmapped_teams=sorted(self.skipped),odds_source=self.public.latest_metadata)
        return list(records.values())

    def odds(self,fixture_id):
        return self.public.odds(self.quote_ids[fixture_id]) if fixture_id in self.quote_ids else []

    def results(self,ids):
        wanted={i for i in ids if i in self.known}
        days={instant(self.known[i]['kickoff_utc']).date().isoformat() for i in wanted}
        result=[]
        for day in sorted(days):
            result.extend(r for r in self._records(day) if r['provider_id'] in wanted)
        return result
