"""
Leakage-safe pre-match features (brief step 2). For each match, a club's form uses only its matches played before
the match's decision date (pre-closing odds collection day — see data.capture_dates), so a Monday game never sees
the same club's Friday result. Without archived schedule-publication timestamps, rest days conservatively use the last match
played before the decision cutoff. Published league sizes replace future observed membership. All counts are league matches in the dataset only (cup games are not in football-data's files).

Columns, for side in (home, away):
  elo_home / elo_away / elo_diff     pre-match Elo (elo_diff includes 60 points home advantage)
  {side}_ppg_{n}, _gf_{n}, _ga_{n}   points per game, goals for / against per game over the last n = 5, 10 matches
  {side}_venue_ppg_5, _venue_gf_5, _venue_ga_5
                                     same over the last 5 matches at this venue (home team at home, away team away)
  {side}_gf_ewm, _ga_ewm             exponentially weighted goals for / against (half-life 10 matches) — older
                                     results count less
  {side}_rest_days                   days since the previous league match (capped at 28)
  {side}_n_prior                     league matches available before the decision date (data depth)
  {side}_games_season                matches already played this season before the decision date
  {side}_promoted, _relegated        club's league tier this season vs last season (1/0)
  season_progress                    mean games played this season / games in a full season
"""
import numpy as np
import pandas as pd

from src.football.data import LEAGUES
from src.football.models import elo_ratings

WINDOWS = (5, 10)
EWM_HALF_LIFE = 10
REST_CAP = 28


def _window_mean(cum: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    """Mean of the n values before position k, given cum = [0, x0, x0+x1, ...]; NaN when there are none."""
    lo = np.maximum(k - n, 0)
    cnt = k - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cum[k] - cum[lo]) / cnt, np.nan)


def _long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, club): venue 0 home / 1 away, goals for/against, points."""
    pts_h = np.select([df.hg > df.ag, df.hg == df.ag], [3, 1], 0)
    home = pd.DataFrame({"row": df.index, "team": df.home, "venue": 0, "gf": df.hg, "ga": df.ag, "pts": pts_h})
    away = pd.DataFrame({"row": df.index, "team": df.away, "venue": 1, "gf": df.ag, "ga": df.hg,
                         "pts": np.select([df.ag > df.hg, df.ag == df.hg], [3, 1], 0)})
    long = pd.concat([home, away], ignore_index=True)
    long["date"], long["decision"] = df.date.to_numpy()[long.row], df.decision.to_numpy()[long.row]
    long["season"], long["league"] = df.season.to_numpy()[long.row], df["div"].to_numpy()[long.row]
    return long.sort_values(["team", "date", "venue"], kind="stable").reset_index(drop=True)


def _team_features(g: pd.DataFrame) -> pd.DataFrame:
    dates, dec = g.date.to_numpy(), g.decision.to_numpy()
    k = np.searchsorted(dates, dec, side="left")              # matches finished before the decision date
    out = {"row": g.row.to_numpy(), "venue": g.venue.to_numpy()}
    cums = {c: np.concatenate([[0.0], np.cumsum(g[c].to_numpy(float))]) for c in ("gf", "ga", "pts")}
    for n in WINDOWS:
        out[f"ppg_{n}"] = _window_mean(cums["pts"], k, n)
        out[f"gf_{n}"] = _window_mean(cums["gf"], k, n)
        out[f"ga_{n}"] = _window_mean(cums["ga"], k, n)
    for c in ("gf", "ga"):
        ewm = g[c].astype(float).ewm(halflife=EWM_HALF_LIFE).mean().to_numpy()
        out[f"{c}_ewm"] = np.where(k > 0, ewm[np.maximum(k - 1, 0)], np.nan)
    # venue-specific form: the club's matches at the same venue as this one
    for c in ("ppg", "gf", "ga"):
        out[f"venue_{c}_5"] = np.full(len(g), np.nan)
    for v in (0, 1):
        at = g.venue.to_numpy() == v
        dv = dates[at]
        kv = np.searchsorted(dv, dec, side="left")
        for c, src in (("ppg", "pts"), ("gf", "gf"), ("ga", "ga")):
            cum = np.concatenate([[0.0], np.cumsum(g[src].to_numpy(float)[at])])
            out[f"venue_{c}_5"] = np.where(at, _window_mean(cum, kv, 5), out[f"venue_{c}_5"])
    prev = np.where(k > 0, dates[np.maximum(k - 1, 0)], np.datetime64("NaT"))
    rest = (dates - prev).astype("timedelta64[D]").astype(float)
    out["rest_days"] = np.minimum(rest, REST_CAP)
    out["n_prior"] = k
    first_of_season = pd.Series(np.arange(len(g))).groupby(g.season.to_numpy()).transform("min").to_numpy()
    out["games_season"] = np.maximum(k - first_of_season, 0)
    return pd.DataFrame(out)


def pre_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame aligned to df (needs date, decision, season, div, home, away, hg, ag)."""
    df = df.reset_index(drop=True).copy()
    df["decision"] = pd.to_datetime(df.decision).dt.normalize()
    if (df.decision > df.date).any():
        raise ValueError("decision date must not follow match date")
    long = _long(df)
    per_team = pd.concat([_team_features(g) for _, g in long.groupby("team", sort=False)], ignore_index=True)
    feats = pd.DataFrame(index=df.index)
    for v, side in ((0, "home"), (1, "away")):
        part = per_team[per_team.venue == v].set_index("row").drop(columns="venue").sort_index()
        feats = feats.join(part.add_prefix(f"{side}_"))
    feats["elo_home"], feats["elo_away"] = elo_ratings(df, by="decision")
    feats["elo_diff"] = feats.elo_home + 60.0 - feats.elo_away
    # Promotion metadata comes from the current fixture's known league and only
    # earlier results in the previous season. Never inspect future season membership.
    tier = df["div"].map(lambda d: LEAGUES.get(d, ("", "", np.nan))[2])
    for side in ("home", "away"):
        previous = []
        membership = {}
        for r in long.itertuples():
            membership.setdefault((r.team, r.season), []).append((r.date, LEAGUES[r.league][2]))
        season_first = df.groupby("season").date.min().to_dict()
        prior_season_available = []
        for row in df.itertuples():
            entries = membership.get((getattr(row, side), row.season-1), [])
            tiers = [t for date, t in entries if date < row.decision]
            previous.append(min(tiers) if tiers else np.nan)
            prior_season_available.append(season_first.get(row.season-1, pd.Timestamp.max) < row.decision)
        before = np.asarray(previous)
        prior_season_available = np.asarray(prior_season_available)
        new_club = np.isnan(before) & prior_season_available
        feats[f"{side}_promoted"] = ((before > tier.to_numpy()) | new_club).astype(int)
        feats[f"{side}_relegated"] = (before < tier.to_numpy()).astype(int)
    # Published league sizes, not clubs discovered in future result rows.
    sizes = {"E0": 20, "E1": 24, "E2": 24, "E3": 24, "SP1": 20,
             "D1": 18, "I1": 20, "F1": 20, "N1": 18, "P1": 18}
    def published_games(div, season):
        # Existing divisions keep their original values. New competitions use
        # published regular-season formats, never future observed club counts.
        if div in sizes:return 2 * ((18 if div == "F1" and season >= 2023 else sizes[div])-1)
        fixed={"D2":34,"SP2":42,"SC0":38,"SC1":36,"SC2":36}
        if div in fixed:return fixed[div]
        if div=="I2":return 42 if season<=2017 else 36 if season==2018 else 38
        if div=="F2":return 34 if season>=2024 else 38
        if div=="B1":return 34 if 2020<=season<=2022 else 30
        if div=="G1":return 26 if season>=2019 else 30
        if div=="T1":return {2020:40,2021:38,2022:36,2023:38,2024:36}.get(season,34)
        raise ValueError(f"No published season format configured for {div}")
    full = np.array([published_games(r.div,r.season) for r in df.itertuples()])
    feats["season_progress"] = (feats.home_games_season + feats.away_games_season) / 2 / full
    return feats
