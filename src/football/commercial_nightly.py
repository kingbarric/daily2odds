"""Isolated 20-league training + seven-day snapshots, using existing model classes."""
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import contextlib
import hashlib
import json
import os
import sqlite3
import subprocess
import uuid
import warnings
import joblib
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from sqlalchemy import create_engine
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from . import db,data,historical,ml
from .commercial_leagues import LEAGUES,EXTRA,SLUGS,ESPN_IDS,TEAM_ALIASES,league_scope
from .odds_training import historical_odds,predictor
from .workflow import predict_fixtures,instant,merge_forward_history
from .scoreboard_provider import ScoreboardCSV,ALIASES
from . import scoreboard_provider
import unicodedata
from .public_provider import PublicCSV

ROOT=Path(__file__).resolve().parents[2]
BASE=db.DB_PATH.parent/'commercial'
SITE=ROOT/'sites/touchline-studio'


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(payload,indent=2,default=str,allow_nan=False));tmp.replace(path)


def training_masks(df,asof,years=8,calibration_days=120):
    cutoff=pd.Timestamp(asof).tz_convert('Europe/London').tz_localize(None).normalize()
    cal_start=cutoff-pd.Timedelta(days=calibration_days)
    training=(df.date>=cutoff-pd.DateOffset(years=years))&(df.date<cal_start)
    calibration=(df.date>=cal_start)&(df.date<cutoff)
    return training,calibration


def refresh_history(progress,now):
    target=BASE/'football.db';raw=BASE/'raw';raw.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        progress('Copying historical database into the isolated commercial workspace')
        with sqlite3.connect(f'file:{db.DB_PATH.resolve()}?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:src.backup(dst)
    season=now.year-(now.month<7)
    jobs=[]
    for year in range(2016,season+1):
        for league in LEAGUES:
            if year!=season and league not in EXTRA:continue
            path=raw/f'{data.season_code(year)}_{league}.csv'
            if path.exists() and year!=season:continue
            jobs.append((year,league,path))
    def fetch(item):
        year,league,path=item
        r=requests.get(data.URL.format(code=data.season_code(year),div=league),timeout=35)
        if r.status_code==404:return item,False
        r.raise_for_status();temp=path.with_suffix('.tmp');temp.write_bytes(r.content);temp.replace(path)
        return item,True
    progress(f'Downloading league-season updates ({len(jobs)} files; historical files cached)')
    failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,item) for item in jobs]
        for n,future in enumerate(as_completed(futures),1):
            try:item,ok=future.result()
            except requests.RequestException as e:failures.append(type(e).__name__)
            if n%20==0:progress(f'Historical downloads: {n}/{len(jobs)}')
    engine=db.engine(target);counts={}
    try:
        for year in range(2016,season+1):
            for league in LEAGUES:
                path=raw/f'{data.season_code(year)}_{league}.csv'
                if not path.exists():continue
                with engine.connect() as c:
                    exists=c.exec_driver_sql('SELECT count(*) FROM matches WHERE league=? AND season=?',(league,year)).scalar()
                if exists and year!=season:continue
                try:
                    matches,odds,issues=data.parse_file(path,league,year)
                    matches=matches[matches.match_date<pd.Timestamp(now).tz_convert('Europe/London').tz_localize(None).normalize()].copy()
                    odds=odds[odds.match_id.isin(matches.match_id)]
                    if matches.empty:continue
                    with engine.begin() as c:db.ingest(c,matches,odds,data.SOURCE)
                    progress(f'Imported {league} {year}: {len(matches)} completed matches')
                except (ValueError,KeyError) as e:failures.append(f'{league} {year}: {type(e).__name__}')
        with engine.connect() as c:counts=dict(c.exec_driver_sql('SELECT league,count(*) FROM matches GROUP BY league').all())
    finally:engine.dispose()
    return target,counts,failures



def refresh_recent_history(history,now,progress):
    from .forward_store import ForwardStore
    store=ForwardStore(BASE/'observed_results.db')
    provider=ScoreboardCSV(canonical_teams=set(history.home)|set(history.away),snapshot_dir=BASE/'results')
    original_team=provider._team
    def resolve(team):
        label=team.get('displayName','');alias=TEAM_ALIASES.get(label)
        return alias if alias in provider.canonical else original_team(team)
    provider._team=resolve
    old=dict(scoreboard_provider.LEAGUE_IDS);scoreboard_provider.LEAGUE_IDS.update(ESPN_IDS)
    try:
        progress('Refreshing completed results from the previous seven days for current team strength')
        today=now.astimezone(ZoneInfo('Europe/London')).date()
        for ago in range(1,8):
            try:
                payload=provider._payload((today-timedelta(days=ago)).isoformat())
                finished=[]
                for event in payload['events']:
                    fixture=provider._normalise(event)
                    if fixture and fixture['status']=='finished':finished.append(fixture)
                store.upsert_fixtures(finished,datetime.now(timezone.utc))
            except (ProviderError,ValueError):continue
        received=datetime.now(timezone.utc)
        return merge_forward_history(history,store.fixtures(),received)
    finally:
        scoreboard_provider.LEAGUE_IDS.clear();scoreboard_provider.LEAGUE_IDS.update(old);store.close()

def train_existing(path,output,now,progress):
    progress('Building past-only features for the existing LightGBM model across 20 leagues')
    history=historical.load(path,last_season=now.year)
    history=refresh_recent_history(history,now,progress)
    cutoff=pd.Timestamp(now).tz_convert('Europe/London').tz_localize(None).normalize()
    history=history[history.date<cutoff].reset_index(drop=True)
    odds=historical_odds(path,history);X=ml.features(history).join(odds)
    valid=odds.notna().all(1);d=history[valid].reset_index(drop=True);X=X[valid].reset_index(drop=True)
    train,cal=training_masks(d,now)
    if train.sum()<1000 or cal.sum()<300:raise ValueError('Insufficient chronological training/calibration data')
    progress(f'Training existing LightGBM: {train.sum()} training rows, {cal.sum()} separate recent calibration rows')
    booster=lgb.LGBMClassifier(**ml.BASE,num_leaves=4,n_estimators=400,random_state=0).fit(X[train],d.loc[train,'result'])
    fitted=CalibratedClassifierCV(FrozenEstimator(booster),method='sigmoid',ensemble=False).fit(X[cal],d.loc[cal,'result'])
    meta=dict(algorithm='Existing LightGBM, 4 leaves x400; sigmoid calibration; Dixon-Coles goals and Elo features retained',training_seasons=sorted(int(x) for x in d.loc[train,'season'].unique()),
        calibration_season=int(d.loc[cal,'season'].max()),prediction_season=now.year-(now.month<7),
        training_rows=int(train.sum()),calibration_rows=int(cal.sum()),training_latest_date=str(d.loc[train,'date'].max()),calibration_latest_date=str(d.loc[cal,'date'].max()),
        cutoff=str(cutoff),training_window_years=8,calibration_days=120,leagues=sorted(d['div'].unique()),
        source_hash=hashlib.sha256(d[['match_id','date','hg','ag']].to_csv(index=False).encode()).hexdigest(),
        status='development_model_not_independent_validation',injury_features='Not trained: current verified player availability unavailable; no numerical injury adjustment')
    artifact=output/'lightgbm_with_odds.joblib';joblib.dump(dict(model=fitted,columns=list(X.columns),metadata=meta),artifact)
    atomic_json(output/'training_metadata.json',meta)
    return history,artifact,meta


def team_context(history,fixture,asof):
    cutoff=pd.Timestamp(asof).tz_convert('UTC').tz_localize(None)
    past=history[pd.to_datetime(history.date)<cutoff].sort_values('date')
    result={}
    for side in ('home','away'):
        team=fixture[side];games=past[(past.home==team)|(past.away==team)].tail(5)
        recent=[]
        for _,r in games.iterrows():
            home=r.home==team;scored=int(r.hg if home else r.ag);conceded=int(r.ag if home else r.hg)
            recent.append(dict(date=str(r.date)[:10],opponent=r.away if home else r.home,scored=scored,conceded=conceded,result='W' if scored>conceded else 'D' if scored==conceded else 'L'))
        result[side]=dict(team=team,historicalMatches=int(((past.home==team)|(past.away==team)).sum()),recent=recent)
    return result


def cached_price_provider():
    cache=json.loads((SITE/'data/prices.json').read_text())
    class Cached:
        def odds(self,identifier):
            return [dict(market=k.split(':')[0],selection=k.split(':')[1],odds=p['odds'],bookmaker=p['bookmaker'],captured_at_utc=p.get('capturedAt'),source_updated_at=p.get('sourceUpdatedAt'))
                    for k,p in cache.get('matches',{}).get(identifier,{}).items() if p.get('capturedAt') and p.get('sourceUpdatedAt')]
    return Cached()


def fixtures_seven_days(history,now,output,progress):
    """Daily public all-scoreboards; explicit league identities, never team-inferred cups."""
    canonical=set(history.home)|set(history.away)
    mapping=ESPN_IDS
    class CommercialScoreboard(ScoreboardCSV):
        def _team(self,team):
            def folded(value):
                return ''.join(c for c in unicodedata.normalize('NFKD',value).casefold() if c.isalnum())
            labels=[team.get(k,'') for k in ('name','displayName','shortDisplayName','location')]
            explicit={TEAM_ALIASES.get(label,ALIASES.get(label,label)) for label in labels}&self.canonical
            if len(explicit)==1:return explicit.pop()
            # Unique accent/case/punctuation identity only; never fuzzy similarity.
            candidates={name for name in self.canonical if folded(name) in {folded(x) for x in labels if x}}
            if len(candidates)==1:return candidates.pop()
            self.skipped.add(team.get('displayName',''));return None
    provider=CommercialScoreboard(canonical_teams=canonical,snapshot_dir=BASE/'fixtures')
    records={};start=now.astimezone(ZoneInfo('Europe/London')).date()
    old=dict(scoreboard_provider.LEAGUE_IDS);scoreboard_provider.LEAGUE_IDS.update(mapping)
    try:
        for offset in range(7):
            day=start+timedelta(days=offset);progress(f'Fetching schedules for {day} across 20 verified league identities')
            for event in provider._payload(day.isoformat())['events']:
                fixture=provider._normalise(event)
                if fixture and fixture['status']=='scheduled' and instant(fixture['kickoff_utc'])>now:records[fixture['provider_id']]=fixture
        absent=set(LEAGUES)-{f['league'] for f in records.values()}
        for code in sorted(absent):
            for offset in range(7):
                day=start+timedelta(days=offset)
                try:
                    response=provider.session.get(f'https://site.api.espn.com/apis/site/v2/sports/soccer/{SLUGS[code]}/scoreboard',params={'dates':day.strftime('%Y%m%d'),'limit':1000},timeout=25)
                    response.raise_for_status();payload=response.json()
                    if not any(l.get('slug')==SLUGS[code] for l in payload.get('leagues',[])):continue
                    for event in payload.get('events',[]):
                        f=provider._normalise(event)
                        if f and f['league']==code and f['status']=='scheduled' and instant(f['kickoff_utc'])>now:records[f['provider_id']]=f
                except (requests.RequestException,ValueError):continue
    finally:
        scoreboard_provider.LEAGUE_IDS.clear();scoreboard_provider.LEAGUE_IDS.update(old)
    return list(records.values()),provider.skipped


def check_news(now):
    # One entitlement probe per night; no fake assumptions about current injuries.
    key=os.getenv('API_FOOTBALL_KEY')
    if not key:return dict(status='unavailable',reason='No configured team-news provider',applied=False)
    try:
        r=requests.get('https://v3.football.api-sports.io/injuries',params={'league':39,'season':now.year-(now.month<7)},headers={'x-apisports-key':key},timeout=25)
        payload=r.json()
        if payload.get('errors'):
            return dict(status='unavailable_current_season',reason='Provider account does not permit current-season injury/suspension data',applied=False,checkedAt=now.isoformat())
        return dict(status='context_available_not_modelled',reason='Availability feed accessible, but player-impact coefficients require historical validation before changing probabilities',applied=False,checkedAt=now.isoformat())
    except (requests.RequestException,ValueError):return dict(status='provider_unavailable',reason='Team-news check failed; no injury adjustment applied',applied=False,checkedAt=now.isoformat())


def mirror_payload(payload,key='forecasts',base=None,site=None):
    if key not in ('forecasts','prices','history'):raise ValueError('Unknown cache key')
    base=Path(base) if base else BASE;site=Path(site) if site else SITE
    base.mkdir(parents=True,exist_ok=True)
    # D1 limits SQL statement size. Write bounded chunks first and flip the
    # current manifest only after every chunk is safely stored.
    import tempfile
    raw=json.dumps(payload,separators=(',',':'),allow_nan=False)
    version=uuid.uuid4().hex;chunks=[raw[i:i+40000] for i in range(0,len(raw),40000)]
    def execute(sql):
        with tempfile.NamedTemporaryFile(mode='w',suffix='.sql',dir=base,delete=False) as f:f.write(sql);filename=f.name
        try:
            result=subprocess.run(['node','--import','./scripts/sites-env.mjs','./node_modules/wrangler/bin/wrangler.js','d1','execute','DB','--local','--config','dist/server/wrangler.json','--persist-to','.wrangler/state','--file',filename],cwd=site,capture_output=True,text=True)
            if result.returncode:raise RuntimeError('Forecast cache import failed; training artifacts retained')
        finally:Path(filename).unlink(missing_ok=True)
    for n,chunk in enumerate(chunks):
        escaped=chunk.replace("'","''")
        execute(f"INSERT INTO forecast_cache(key,payload,updated_at) VALUES ('{key}:{version}:{n}','{escaped}','{datetime.now(timezone.utc).isoformat()}');")
    manifest=json.dumps(dict(chunked=True,version=version,count=len(chunks))).replace("'","''")
    from .commercial_vps import publish_cache
    if site.resolve()==(Path(__file__).resolve().parents[2]/'sites/touchline-studio').resolve():
        publish_cache(payload,key)
    execute(f"INSERT INTO forecast_cache(key,payload,updated_at) VALUES ('{key}','{manifest}','{datetime.now(timezone.utc).isoformat()}') ON CONFLICT(key) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at;")


def mirror_snapshot(payload):
    return mirror_payload(payload,'forecasts')

def run(reuse_model=None):
    BASE.mkdir(parents=True,exist_ok=True);now=datetime.now(timezone.utc);run_id=now.strftime('%Y%m%dT%H%M%SZ');output=ROOT/'reports/football_nightly'/run_id;output.mkdir(parents=True,exist_ok=True)
    lock=BASE/'nightly.lock'
    import fcntl
    with lock.open('w') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('Nightly training already running; previous forecasts remain available.');return
        def progress(message):
            print(message,flush=True);atomic_json(BASE/'latest_job.json',dict(status='running',run=run_id,message=message,updatedAt=datetime.now(timezone.utc).isoformat()))
        try:
            with league_scope():
                if reuse_model:
                    path=BASE/'football.db';artifact=Path(reuse_model)
                    metadata=joblib.load(artifact)['metadata']
                    history=historical.load(path,last_season=now.year)
                    history=refresh_recent_history(history,now,progress)
                    with sqlite3.connect(path) as c:counts=dict(c.execute('SELECT league,count(*) FROM matches GROUP BY league').fetchall())
                    failures=[]
                else:
                    path,counts,failures=refresh_history(progress,now)
                    history,artifact,metadata=train_existing(path,output,now,progress)
                fixtures,skipped=fixtures_seven_days(history,now,output,progress)
                progress(f'Running existing goal and outcome models on {len(fixtures)} fixtures')
                provider=cached_price_provider();diagnostics={}
                # Capture the actual input clock after training, keeping price future/age checks intact.
                asof=datetime.now(timezone.utc)
                probabilities,version=predictor(artifact,provider)(history,fixtures,asof)
                predict_fixtures(history,fixtures,asof,diagnostics=diagnostics)
                news=check_news(asof);rows=[]
                for f in fixtures:
                    mid=f['provider_id']
                    if mid not in probabilities or instant(f['kickoff_utc'])<=asof:continue
                    g=diagnostics[mid]
                    rows.append(dict(id=mid,date=instant(f['kickoff_utc']).astimezone(ZoneInfo('Europe/London')).date().isoformat(),league=f['league'],home=f['home'],away=f['away'],asOf=asof.isoformat(),kickoff=f['kickoff_utc'],
                        homeGoals=g['expected_home_goals'],awayGoals=g['expected_away_goals'],probabilities=probabilities[mid],prices={},modelSource=provider.prediction_sources.get(mid,'Dixon-Coles'),teamNewsStatus=news['status'],teamContext=team_context(history,f,asof)))
                if not rows:raise ValueError('No valid seven-day forecasts generated; previous public snapshot retained')
                dates=[(asof.astimezone(ZoneInfo('Europe/London')).date()+timedelta(days=i)).isoformat() for i in range(7)]
                payload=dict(snapshot=dates[0],generatedAt=asof.isoformat(),windowStart=dates[0],windowEnd=dates[-1],status='nightly_forecast',source='Existing models retrained on completed earlier results; cached odds only',
                    leagues={code:dict(name=info[1],country=info[0],historicalMatches=counts.get(code,0),forecastFixtures=sum(r['league']==code for r in rows)) for code,info in LEAGUES.items()},teamNews=news,training=metadata,
                    coverageWarnings=failures+['Current injury/suspension impacts not applied; no validated current player-availability input.'],skippedTeams=sorted(skipped),matches=rows)
                atomic_json(output/'forecasts.json',payload)
                # Freeze each version before switching the current public cache.
                mirror_snapshot(payload);atomic_json(BASE/'latest_forecasts.json',payload)
                from .commercial_history import publish_history
                publish_history()
                atomic_json(BASE/'latest_job.json',dict(status='completed',run=run_id,output=str(output),forecasts=len(rows),leagues=len(counts),windowStart=dates[0],windowEnd=dates[-1],teamNews=news))
                print(json.dumps(dict(status='completed',forecasts=len(rows),historicalLeagues=len(counts),teamNews=news)),flush=True)
        except Exception as exc:
            atomic_json(BASE/'latest_job.json',dict(status='failed',run=run_id,error=type(exc).__name__,message='Nightly run failed; previous public forecasts retained.'))
            print('Nightly run failed:',type(exc).__name__,flush=True);raise
