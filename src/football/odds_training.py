"""Existing LightGBM with historical pre-closing market features (development only).

Closing odds and target scores never enter X. Exact historical price timestamps
remain unverified: no training result is promoted to verified executable returns.
The score distribution for totals/BTTS remains the existing Dixon–Coles model.
"""
import hashlib
import json
from pathlib import Path
from datetime import timedelta

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sqlalchemy import create_engine

from . import ml,evaluate,markets
from .workflow import instant,predict_fixtures

ODDS_FEATURES=['odds_home_probability','odds_draw_probability','odds_away_probability','odds_overround']


def historical_odds(path,df):
    engine=create_engine(f'sqlite:///file:{Path(path).resolve()}?mode=ro&uri=true')
    try:
        with engine.connect() as c:
            # Explicit pre-closing predicate: closing columns cannot silently win
            # a fallback or become a model input.
            o=pd.read_sql("SELECT match_id,selection,odds FROM odds_snapshots WHERE bookmaker='market_avg' AND market='1x2' AND snapshot='pre_closing'",c)
    finally:engine.dispose()
    prices=o.pivot(index='match_id',columns='selection',values='odds').reindex(index=df.match_id,columns=['home','draw','away']).to_numpy(float)
    return price_features(prices,index=df.index)


def price_features(prices,index=None):
    prices=np.asarray(prices,float)
    inv=1/prices;overround=inv.sum(1);p=inv/overround[:,None]
    good=np.isfinite(prices).all(1)&(prices>1).all(1)
    p[~good]=np.nan;overround[~good]=np.nan
    return pd.DataFrame(np.column_stack([p,overround]),columns=ODDS_FEATURES,index=index)


def train(df,odds_features,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    base=ml.features(df);augmented=base.join(odds_features)
    available=odds_features.notna().all(1).to_numpy()
    # Keep identical availability rows for both model variants and bookmaker.
    d=df[available].reset_index(drop=True);X=augmented[available].reset_index(drop=True);Xbase=base[available].reset_index(drop=True)
    rows=[];predictions={}
    params=dict(num_leaves=4,n_estimators=400)
    for label,inputs in [('existing_features',Xbase),('existing_features_plus_preclosing_odds',X)]:
        runs,_=ml.walk(inputs,d.result.to_numpy(),d.season.to_numpy(),[2024,2025],2017,params)
        p=runs['sigmoid'];predictions[label]=p
        for season in [2024,2025]:
            mask=(d.season.to_numpy()==season)&np.isfinite(p).all(1)
            scores=evaluate.scores(p[mask],d.result.to_numpy()[mask])
            rows.append(dict(model=label,season=season,**scores))
    odds_p=X[ODDS_FEATURES[:3]].to_numpy()
    for season in [2024,2025]:
        mask=d.season.to_numpy()==season
        rows.append(dict(model='bookmaker_margin_removed',season=season,**evaluate.scores(odds_p[mask],d.result.to_numpy()[mask])))
    # Final current-season artifact: chronological booster/calibrator separation.
    # Rolling form/Elo for new fixtures may incorporate later known results, while
    # no current target outcome is used to train this fixed season's head.
    season=int(d.season.max());training=(d.season>=2017)&(d.season<=season-2);cal=d.season==season-1
    if training.sum()<1000 or cal.sum()<300:raise ValueError('insufficient chronological training/calibration data')
    booster=lgb.LGBMClassifier(**ml.BASE,**params,random_state=0).fit(X[training],d.result.to_numpy()[training])
    fitted=CalibratedClassifierCV(FrozenEstimator(booster),method='sigmoid',ensemble=False).fit(X[cal],d.result.to_numpy()[cal])
    metadata=dict(algorithm='existing LightGBM; 4 leaves x400; sigmoid calibration',
        status='development_experiment_not_independent_validation',prediction_season=season,
        training_seasons=sorted(int(v) for v in d[training].season.unique()),calibration_season=season-1,
        training_rows=int(training.sum()),calibration_rows=int(cal.sum()),
        historical_source='football-data.co.uk market_avg pre_closing 1X2 only',
        timestamp_policy='historical per-price timestamps unverified; no claim of executable odds',
        market_scope='LightGBM 1X2 and derived double chance; existing Dixon-Coles remains totals and BTTS model',
        live_input_note='mean available fresh UK bookmaker decimal prices, normalised after averaging; bookmaker mix may differ from historical market_avg',
        source_hash=hashlib.sha256(d[['match_id','date','hg','ag']].to_csv(index=False).encode()+odds_features.to_csv(index=False).encode()).hexdigest())
    joblib.dump(dict(model=fitted,columns=list(X.columns),metadata=metadata),output/'lightgbm_with_odds.joblib')
    table=pd.DataFrame(rows);table.to_csv(output/'training_evaluation.csv',index=False)
    for label,p in predictions.items():
        frame=d[['match_id','season','div','date','home','away','hg','ag','result']].copy()
        frame[['home_probability','draw_probability','away_probability']]=p
        frame[frame.season.isin([2024,2025])].to_csv(output/(label+'_development_predictions.csv'),index=False)
    (output/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(table.to_string(index=False),flush=True)
    return metadata


def live_price_features(provider,fixtures,as_of,*,max_source_age=timedelta(hours=1)):
    rows=[]
    for f in fixtures:
        quotes=provider.odds(f['provider_id']);by_book={}
        for q in quotes:
            if q['market']!='1x2':continue
            age=instant(as_of)-instant(q['source_updated_at'])
            if age<timedelta(0) or (max_source_age is not None and age>max_source_age):continue
            if instant(q['captured_at_utc'])>instant(as_of):continue
            by_book.setdefault(q['bookmaker'],{})[q['selection']]=q['odds']
        complete=[[g[s] for s in ['home','draw','away']] for g in by_book.values() if set(g)=={'home','draw','away'}]
        rows.append(np.mean(complete,axis=0) if complete else [np.nan]*3)
    return price_features(rows)


def predictor(model_path,provider,*,max_source_age=timedelta(hours=1)):
    artifact=joblib.load(model_path)  # trusted locally produced artifact only
    def predict(history,fixtures,as_of):
        original,version=predict_fixtures(history,fixtures,as_of)
        provider.prediction_sources={f['provider_id']:'DC-all-markets-no-fresh-odds' for f in fixtures}
        records=[]
        for f in fixtures:
            kickoff=pd.Timestamp(f['kickoff_utc'])
            records.append(dict(match_id=f['provider_id'],date=kickoff.tz_convert('Europe/London').tz_localize(None).normalize(),
                                decision=pd.Timestamp(as_of).tz_localize(None).normalize(),div=f['league'],season=int(f['season']),
                                home=f['home'],away=f['away'],hg=np.nan,ag=np.nan,result=0,_future=True))
        if not records:return original,version
        hist=history.copy();hist['_future']=False
        joined=pd.concat([hist,pd.DataFrame(records)],ignore_index=True).sort_values(['date','div','home'],kind='stable').reset_index(drop=True)
        X=ml.features(joined);positions=joined.index[joined._future]
        mapping={r['match_id']:r for r in records}
        ordered=[mapping[mid] for mid in joined.loc[positions,'match_id']]
        odds=live_price_features(provider,[next(f for f in fixtures if f['provider_id']==r['match_id']) for r in ordered],as_of,max_source_age=max_source_age)
        valid=odds.notna().all(1).to_numpy()
        targets=X.loc[positions].reset_index(drop=True).join(odds)
        if valid.any():
            p=artifact['model'].predict_proba(targets.loc[valid,artifact['columns']])
            for f,prob in zip([r for r,ok in zip(ordered,valid) if ok],p):
                identifier=f['match_id']
                if identifier not in original:continue
                provider.prediction_sources[identifier]='odds-LightGBM-1x2-DC-goals'
                for key,value in markets.from_1x2(np.asarray([prob])).iloc[0].items():original[identifier][key]=float(value)
        return original,version+':experimental-odds-lgbm-1x2-dc:'+artifact['metadata']['source_hash'][:20]
    return predict
