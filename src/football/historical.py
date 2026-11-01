"""Versioned, chronological forecasts from existing models for development replays.

No tuning, model blending, synthetic odds, or writes to the source SQLite database.
2026/27 is excluded entirely from these development runs.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from . import db, markets, models, ml
from .data import LEAGUES, capture_times_utc

VERSION = 'integrity-v3-development-2024-2025'


def load(path, last_season=2025):
    engine = create_engine(f'sqlite:///file:{Path(path).resolve()}?mode=ro&uri=true')
    try:
        with engine.connect() as conn:
            df = db.load_matches(conn, seasons=range(2016, last_season+1))
    finally:
        engine.dispose()
    df = df[df.hg.notna() & df.ag.notna()].reset_index(drop=True)
    # Estimated collection schedule is only a conservative simulation convention.
    # Preserve it separately: it is not a verified individual quote receipt.
    cap = capture_times_utc(df.date)
    kick = pd.to_datetime(df.kickoff_utc)
    df['estimated_schedule_utc'] = cap.where(kick.notna() & (cap < kick))
    return df


def fingerprint(df):
    fields=['match_id','date','decision','div','season','home','away','hg','ag','kickoff_utc']
    return hashlib.sha256(df[fields].to_csv(index=False).encode()).hexdigest()


def generate(df, cache_dir, start_season=2023):
    """Returns {model: wide probabilities}, with full source-frame alignment.

    The extra 2023/24 season is calibration warm-up only. Test target seasons are
    2024/25 and 2025/26, both explicitly development data. Fits use dates strictly
    before each decision day. Convergence failures withhold that group's forecast.
    """
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    key=f'{VERSION}-{fingerprint(df)}-{start_season}'
    file=cache_dir/(key+'.pkl')
    if file.exists():
        payload=pd.read_pickle(file)
        if payload['fingerprint'] != fingerprint(df):
            raise ValueError('cache source fingerprint mismatch')
        return payload['models'],payload['failures']
    failures=[]
    cols=[c for m in markets.MARKETS for c in markets.columns(m)]
    dc=pd.DataFrame(np.nan,index=df.index,columns=cols)
    countries=df['div'].map(lambda d:LEAGUES[d][0])
    for country in countries.unique():
        country_rows=df[countries==country]
        fits={365:models.DixonColes(half_life=365,window_days=3650),
              730:models.DixonColes(half_life=730,window_days=3650)}
        groups=country_rows[country_rows.season>=start_season].groupby('decision',sort=True)
        for day,targets in groups:
            train=country_rows[country_rows.date<day]
            for half_life,model in fits.items():
                try:
                    model.fit(train,day)
                    p=model.markets(targets)
                    use=[c for c in cols if (c.startswith(('1x2:','double_chance:'))) == (half_life==365)]
                    dc.loc[targets.index,use]=p[use].to_numpy()
                except (ValueError,FloatingPointError) as exc:
                    failures.append(dict(country=country,decision=str(day),half_life=half_life,reason=str(exc)))
        print(f'Dixon-Coles complete: {country}',flush=True)
    diff=models.elo_diff(df,k=20,by='decision')
    seasons=range(start_season,int(df.season.max())+1)
    elo=markets.from_1x2(models.elo_probs(df,diff,seasons,train_years=5,burn_in=0)).set_index(df.index)
    print('Elo complete',flush=True)
    X=ml.features(df)
    predictions,_=ml.walk(X,df.result.to_numpy(),df.season.to_numpy(),seasons,2017,
                          dict(num_leaves=4,n_estimators=400))
    gbm=markets.from_1x2(predictions['sigmoid']).set_index(df.index)
    print('LightGBM complete: existing form/Elo/H2H features, 4 leaves x400, chronological sigmoid calibration',flush=True)
    result={'Dixon-Coles':dc,'Elo':elo,'LightGBM':gbm}
    # Catch corrupt distributions before saving or using a cached run.
    for name,frame in result.items():
        for m in markets.EXCLUSIVE:
            c=markets.columns(m)
            if not set(c)<=set(frame.columns):
                continue
            p=frame[c].to_numpy();ok=np.isfinite(p).all(1)
            if ((p[ok]<0)|(p[ok]>1)).any() or not np.allclose(p[ok].sum(1),1):
                raise ValueError(f'invalid {name} {m} distribution')
    pd.to_pickle(dict(fingerprint=fingerprint(df),models=result,failures=failures,version=VERSION),file)
    return result,failures


def load_odds(path, match_ids):
    engine=create_engine(f'sqlite:///file:{Path(path).resolve()}?mode=ro&uri=true')
    try:
        with engine.connect() as conn:
            odds=pd.read_sql("SELECT * FROM odds_snapshots WHERE snapshot='pre_closing'",conn)
    finally:
        engine.dispose()
    odds=odds[odds.match_id.isin(set(match_ids))].copy()
    odds['captured_at_utc']=pd.to_datetime(odds.captured_at_utc)
    odds['timestamp_verified']=~odds.captured_at_estimated.astype(bool)&odds.captured_at_utc.notna()
    return odds
