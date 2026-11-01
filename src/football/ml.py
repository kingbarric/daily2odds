"""
LightGBM home/draw/away model (brief step 4) on the leakage-safe pre-match features, calibrated on a separate
chronological validation season with scikit-learn.

Walk-forward per predicted season s: the booster learns from seasons first_train … s−2, the calibrator from season
s−1, and season s is predicted. No random splits; nothing the model or its calibrator sees is from season s or
later. One booster per season serves every calibration method, so methods are compared on identical models.

  features(df)   pre-match form / Elo / rest / promotion features + head-to-head record + league (categorical)
  walk(...)      {calibration method: (n, 3) probabilities, NaN outside the predicted seasons}, gain importance
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.football.data import LEAGUES
from src.football.features import pre_match_features
from src.football.models import h2h_features

# heavily regularised: match outcomes are mostly noise, so trees must stay shallow and leaves well populated
BASE = dict(objective="multiclass", learning_rate=0.02, min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=4)
CONFIGS = {"4 leaves × 400 trees": dict(num_leaves=4, n_estimators=400),
           "15 leaves × 250 trees": dict(num_leaves=15, n_estimators=250)}
CALIBRATIONS = ("none", "sigmoid", "isotonic", "temperature")


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs aligned to df (date-sorted, with decision dates): every column is known before the match's
    pre-closing odds collection."""
    X = pre_match_features(df).join(h2h_features(df))
    X["league"] = pd.Categorical(df["div"].to_numpy(), categories=list(LEAGUES))
    return X


def walk(X: pd.DataFrame, y: np.ndarray, season: np.ndarray, predict_seasons, first_train: int, params: dict,
         seed: int = 0) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Probabilities for rows in predict_seasons under every calibration method, and the booster's gain importance
    per predicted season (columns)."""
    y, season = np.asarray(y), np.asarray(season)
    out = {m: np.full((len(X), 3), np.nan) for m in CALIBRATIONS}
    importance = {}
    for s in predict_seasons:
        tr = (season >= first_train) & (season <= s - 2)
        cal, te = season == s - 1, season == s
        if not te.any() or tr.sum() < 1000 or cal.sum() < 300:
            continue
        booster = lgb.LGBMClassifier(**BASE, **params, random_state=seed).fit(X[tr], y[tr])
        out["none"][te] = booster.predict_proba(X[te])
        for m in CALIBRATIONS[1:]:
            c = CalibratedClassifierCV(FrozenEstimator(booster), method=m, ensemble=False).fit(X[cal], y[cal])
            out[m][te] = c.predict_proba(X[te])
        importance[s] = pd.Series(booster.booster_.feature_importance("gain"), index=X.columns)
    return out, pd.DataFrame(importance)
