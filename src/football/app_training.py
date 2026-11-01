"""Rolling-window comparisons using the existing fixed LightGBM architecture."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from . import ml, evaluate, historical
from .odds_training import historical_odds


def split_masks(df, target_season, years):
    if years not in (3, 5, 8):
        raise ValueError('Training window must be 3, 5 or 8 seasons')
    train = df.season.between(target_season-years-1, target_season-2)
    cal = df.season.eq(target_season-1)
    test = df.season.eq(target_season)
    return train, cal, test


def fit(X, df, target_season, years):
    train, cal, _ = split_masks(df, target_season, years)
    if train.sum() < 500 or cal.sum() < 100:
        raise ValueError('Insufficient earlier training/calibration rows for this date')
    booster = lgb.LGBMClassifier(**ml.BASE, num_leaves=4, n_estimators=400,
                                random_state=0).fit(X.loc[train], df.loc[train, 'result'])
    model = CalibratedClassifierCV(FrozenEstimator(booster), method='sigmoid', ensemble=False)
    model.fit(X.loc[cal], df.loc[cal, 'result'])
    return model, dict(requested_training_seasons=years,
        training_seasons=sorted(int(v) for v in df.loc[train, 'season'].unique()),
        calibration_season=int(target_season-1), prediction_season=int(target_season),
        training_rows=int(train.sum()), calibration_rows=int(cal.sum()))


def inputs(path):
    df = historical.load(path, last_season=2100)
    odds = historical_odds(path, df)
    X = ml.features(df).join(odds)
    valid = odds.notna().all(axis=1)
    return df.loc[valid].reset_index(drop=True), X.loc[valid].reset_index(drop=True)


def train(path, output, years=5, compare=True, progress=print):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    progress('Building past-only features and reading pre-closing odds')
    df, X = inputs(path)
    rows=[]
    if compare:
        for season in (2024, 2025):
            _, _, mask = split_masks(df, season, years)
            if not mask.any(): continue
            y = df.loc[mask, 'result'].to_numpy()
            benchmark = X.loc[mask, ['odds_home_probability','odds_draw_probability','odds_away_probability']].to_numpy()
            rows.append(dict(model='Bookmaker margin removed', target_season=season,
                             requested_years=0, actual_years=0, **evaluate.scores(benchmark,y)))
            for window in (3,5,8):
                progress(f'Evaluating {window}-season window on {season}/{str(season+1)[2:]} development data')
                model, metadata=fit(X,df,season,window)
                p=model.predict_proba(X.loc[mask])
                rows.append(dict(model='Existing LightGBM + odds', target_season=season,
                    requested_years=window, actual_years=len(metadata['training_seasons']),
                    **evaluate.scores(p,y)))
                evaluate.calibration(p,np.eye(3)[y]).to_csv(output/f'calibration_{season}_{window}.csv')
                frame=df.loc[mask,['match_id','date','div','home','away','hg','ag']].copy()
                frame[['home_probability','draw_probability','away_probability']]=p
                frame.to_csv(output/f'development_{season}_{window}.csv',index=False)
        pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    target=int(df.season.max())
    progress(f'Training selected {years}-season model for season {target}')
    model, metadata=fit(X,df,target,years)
    metadata.update(status='development_experiment_not_independent_validation',
        source_hash=historical.fingerprint(df),
        historical_odds_timestamps='unverified; not evidence of executable prices',
        market_scope='LightGBM 1X2/double chance; existing Dixon-Coles totals/BTTS',
        note='Window excludes the separate calibration season. Eight-year early folds can have fewer available seasons.')
    joblib.dump(dict(model=model,columns=list(X.columns),metadata=metadata),output/'lightgbm_with_odds.joblib')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return dict(output=str(output.resolve()),metadata=metadata)
