"""
Pre-match prediction models built only from past results. Matches are processed in decision groups (`by`): every
prediction in a group uses only results from earlier groups. With by="date" that means strictly earlier days; the
bot uses by="decision" — the pre-closing odds collection day — so a Saturday prediction knows results up to
Thursday, exactly what was known when its price was quoted.

  Elo           one rating per club across its country's divisions (promoted/relegated clubs keep their rating);
                rating difference → home/draw/away via a per-division multinomial logistic fit on past seasons.
                Ratings are comparable only within a country (no cross-border matches in the data).
  Dixon-Coles   time-weighted Poisson model of goals (attack, defence, home advantage, low-score correction),
                refit for every decision group — one model per country (its divisions are linked by promotion);
                its score grid gives every market coherently.
  Head-to-head  record of the last meetings between the two clubs (either venue) in the previous 10 years.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

from src.football.data import LEAGUES
from src.football.markets import from_grid

MAX_GOALS = 10


def div_start(div: str) -> float:
    """Starting Elo for a club first seen in a division: 1700 − 100 × tier (Premier League 1600 … League Two 1300)."""
    return 1700.0 - 100.0 * LEAGUES.get(div, ("", "", 4))[2]


# ---------------------------------------------------------------- Elo

def elo_ratings(df: pd.DataFrame, k: float = 20.0, by: str = "date") -> tuple[np.ndarray, np.ndarray]:
    """Pre-match (home rating, away rating) for every row, from results in earlier `by` groups only. Goal-difference
    multiplier as in the World Football Elo ratings; home advantage of 60 points in the expected score. A club
    seen for the first time starts at the 25th percentile of its division."""
    rating, last_div = {}, {}
    home, away, div = df.home.to_numpy(), df.away.to_numpy(), df["div"].to_numpy()
    gd = (df.hg - df.ag).to_numpy()
    rh, ra = np.empty(len(df)), np.empty(len(df))
    # Apply results by actual match date, never by the decision group of their forecast.
    # This matters for postponed fixtures whose original decision precedes other fixtures.
    order = np.argsort(df.date.to_numpy(), kind="stable")
    cursor = 0

    def ensure(team, division):
        if team not in rating:
            peers = [rating[t] for t, d in last_div.items() if d == division]
            rating[team] = float(np.percentile(peers, 25)) if len(peers) >= 10 else div_start(division)

    groups = pd.to_datetime(df[by]).dt.normalize()
    for key, idx in df.groupby(groups, sort=True).indices.items():
        while cursor < len(order) and df.date.iloc[order[cursor]] < key:
            i = order[cursor]
            cursor += 1
            if not np.isfinite(gd[i]):
                continue
            ensure(home[i], div[i])
            ensure(away[i], div[i])
            expected = 1.0 / (1.0 + 10 ** (-(rating[home[i]] + 60.0 - rating[away[i]]) / 400.0))
            g = abs(gd[i])
            mult = 1.0 if g <= 1 else 1.5 if g == 2 else (11.0 + g) / 8.0
            delta = k * mult * ((1.0 + np.sign(gd[i])) / 2.0 - expected)
            rating[home[i]] += delta
            rating[away[i]] -= delta
            last_div[home[i]] = last_div[away[i]] = div[i]
        for i in idx:
            ensure(home[i], div[i])
            ensure(away[i], div[i])
            rh[i], ra[i] = rating[home[i]], rating[away[i]]
    return rh, ra


def elo_diff(df: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0, by: str = "date") -> np.ndarray:
    """Pre-match home rating + home advantage − away rating."""
    rh, ra = elo_ratings(df, k, by)
    return rh + home_adv - ra


def elo_probs(df: pd.DataFrame, diff: np.ndarray, test_seasons, train_years: int = 5,
              burn_in: int = 1) -> np.ndarray:
    """Home/draw/away probabilities for rows in test_seasons; each season's logistic map is fit on the previous
    train_years seasons only (skipping the first burn_in seasons while ratings settle)."""
    p = np.full((len(df), 3), np.nan)
    first = df.season.min() + burn_in
    x = (diff / 400.0).reshape(-1, 1)
    for s in test_seasons:
        for d in df["div"].unique():
            tr = ((df.season >= max(first, s - train_years)) & (df.season < s) & (df["div"] == d)).to_numpy()
            te = ((df.season == s) & (df["div"] == d)).to_numpy()
            if te.any() and tr.sum() > 300:
                m = LogisticRegression(C=100.0, max_iter=1000).fit(x[tr], df.result.to_numpy()[tr])
                p[te] = m.predict_proba(x[te])
    return p


# ---------------------------------------------------------------- Dixon-Coles

def _neg_loglik(theta, hi, ai, x, y, w, n, ridge, prior):
    att, dfn = theta[:n], theta[n:2 * n]
    mu, home, rho = theta[2 * n:]
    ll = mu + home + att[hi] + dfn[ai]                  # log expected home goals
    lm = mu + att[ai] + dfn[hi]                         # log expected away goals
    lam, mm = np.exp(ll), np.exp(lm)
    f = x * ll - lam + y * lm - mm
    g_ll, g_lm, g_rho = x - lam, y - mm, np.zeros_like(lam)
    # low-score correction tau on 0-0, 0-1, 1-0, 1-1 (derivative wrt log-rate = rate · derivative wrt rate)
    m = (x == 0) & (y == 0)
    lmm = lam[m] * mm[m]
    t = np.maximum(1 - lmm * rho, 1e-9)
    f[m] += np.log(t)
    g_ll[m] -= lmm * rho / t
    g_lm[m] -= lmm * rho / t
    g_rho[m] -= lmm / t
    m = (x == 0) & (y == 1)
    t = np.maximum(1 + lam[m] * rho, 1e-9)
    f[m] += np.log(t)
    g_ll[m] += lam[m] * rho / t
    g_rho[m] += lam[m] / t
    m = (x == 1) & (y == 0)
    t = np.maximum(1 + mm[m] * rho, 1e-9)
    f[m] += np.log(t)
    g_lm[m] += mm[m] * rho / t
    g_rho[m] += mm[m] / t
    m = (x == 1) & (y == 1)
    f[m] += np.log(1 - rho)
    g_rho[m] -= 1 / (1 - rho)
    wl, wm = w * g_ll, w * g_lm
    grad = np.empty_like(theta)
    grad[:n] = np.bincount(hi, wl, n) + np.bincount(ai, wm, n) - 2 * ridge * (att - prior[0])
    grad[n:2 * n] = np.bincount(ai, wl, n) + np.bincount(hi, wm, n) - 2 * ridge * (dfn - prior[1])
    grad[2 * n:] = (wl.sum() + wm.sum(), wl.sum(), (w * g_rho).sum())
    obj = (w * f).sum() - ridge * (((att - prior[0]) ** 2).sum() + ((dfn - prior[1]) ** 2).sum())
    return -obj, -grad


class DixonColes:
    """Goals model: log E[home goals] = mu + home + att[h] + def[a], log E[away goals] = mu + att[a] + def[h].
    Matches are weighted exp(-ln2 · age / half_life); clubs are shrunk toward their division's average (from the
    previous fit), which also gives a starting point for clubs with no history yet."""

    def __init__(self, half_life: float | None = 365.0, window_days: int = 4 * 365, ridge: float = 1.0):
        self.half_life, self.window_days, self.ridge = half_life, window_days, ridge
        self.params: dict[str, tuple[float, float]] = {}
        self.div_mean: dict[str, tuple[float, float]] = {}
        self.mu = self.home = self.rho = 0.0

    def fit(self, hist: pd.DataFrame, as_of: pd.Timestamp) -> "DixonColes":
        as_of = pd.Timestamp(as_of).normalize()
        hist = hist[(hist.date < as_of) & (hist.date >= as_of - pd.Timedelta(days=self.window_days))]
        if hist.empty:
            raise ValueError("no completed history before model cutoff")
        teams = pd.unique(np.concatenate([hist.home.to_numpy(), hist.away.to_numpy()]))
        n, index = len(teams), {t: i for i, t in enumerate(teams)}
        hi, ai = hist.home.map(index).to_numpy(), hist.away.map(index).to_numpy()
        age = (as_of - hist.date).dt.days.to_numpy(float)
        w = np.exp(-np.log(2) * age / self.half_life) if self.half_life else np.ones(len(hist))
        latest_div = pd.concat([hist[["date", "home", "div"]].rename(columns={"home": "team"}),
                                hist[["date", "away", "div"]].rename(columns={"away": "team"})]
                               ).sort_values("date").groupby("team")["div"].last()
        prior = np.array([self.div_mean.get(latest_div[t], (0.0, 0.0)) for t in teams]).reshape(-1, 2).T
        start = np.array([self.params.get(t, prior[:, i]) for i, t in enumerate(teams)]).reshape(-1, 2)
        x0 = np.concatenate([start[:, 0], start[:, 1], [self.mu, self.home, self.rho]])
        bounds = [(None, None)] * (2 * n + 2) + [(-0.2, 0.1)]
        res = minimize(_neg_loglik, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       args=(hi, ai, hist.hg.to_numpy(float), hist.ag.to_numpy(float), w, n, self.ridge, prior))
        if not res.success or not np.isfinite(res.x).all():
            raise ValueError("Dixon-Coles optimisation did not converge; forecast withheld")
        th = res.x
        self.params = {t: (th[i], th[n + i]) for i, t in enumerate(teams)}
        self.mu, self.home, self.rho = th[2 * n:]
        by_div = defaultdict(list)
        for t, d in latest_div.items():
            by_div[d].append(self.params[t])
        self.div_mean = {d: tuple(np.mean(v, axis=0)) for d, v in by_div.items()}
        return self

    def _team(self, team: str, div: str) -> tuple[float, float]:
        return self.params.get(team) or self.div_mean.get(div, (0.0, 0.0))

    def expected_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        h = np.array([self._team(t, d) for t, d in zip(df.home, df["div"])]).reshape(-1, 2)
        a = np.array([self._team(t, d) for t, d in zip(df.away, df["div"])]).reshape(-1, 2)
        return (np.exp(self.mu + self.home + h[:, 0] + a[:, 1]), np.exp(self.mu + a[:, 0] + h[:, 1]))

    def score_grid(self, df: pd.DataFrame) -> np.ndarray:
        """(n, MAX_GOALS+1, MAX_GOALS+1) probability of each exact score (home goals on axis 1)."""
        lam, mm = self.expected_goals(df)
        g = np.arange(MAX_GOALS + 1)
        grid = poisson.pmf(g[None, :, None], lam[:, None, None]) * poisson.pmf(g[None, None, :], mm[:, None, None])
        r = self.rho
        grid[:, 0, 0] *= 1 - lam * mm * r
        grid[:, 0, 1] *= 1 + lam * r
        grid[:, 1, 0] *= 1 + mm * r
        grid[:, 1, 1] *= 1 - r
        return grid / grid.sum(axis=(1, 2), keepdims=True)

    def markets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Probability of every selection in markets.MARKETS, from the score grid."""
        return from_grid(self.score_grid(df))

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(n, 3) home/draw/away and (n,) P(over 2.5 goals)."""
        m = self.markets(df)
        return m[["1x2:home", "1x2:draw", "1x2:away"]].to_numpy(), m["ou_2.5:over"].to_numpy()


def dixon_coles_walk(df: pd.DataFrame, start, refit_days: int = 1, by: str = "date", group=None,
                     **kw) -> pd.DataFrame:
    """Walk forward through every `by` group from start: fit on all matches played before the group's key date
    (refit at most every refit_days), predict the group. Returns every market's probabilities; NaN before start.
    group (labels aligned to df, e.g. country): each label gets its own model, so leagues that never meet do not
    share home advantage, scoring rate or rho."""
    if group is not None:
        labels = pd.Series(np.asarray(group), index=df.index)
        return pd.concat([dixon_coles_walk(g, start, refit_days, by, **kw)
                          for _, g in df.groupby(labels, sort=False)]).reindex(df.index)
    out = pd.DataFrame(np.nan, index=df.index, columns=from_grid(np.full((1, 2, 2), 0.25)).columns)
    dates = df.date.to_numpy()                                         # df is date-sorted
    model, last_fit = DixonColes(**kw), None
    for key, idx in df.groupby(by, sort=True).indices.items():
        if key < pd.Timestamp(start):
            continue
        if last_fit is None or (key - last_fit).days >= refit_days:
            model.fit(df.iloc[: np.searchsorted(dates, np.datetime64(key), side="left")], key)
            last_fit = key
        out.iloc[idx] = model.markets(df.iloc[idx]).to_numpy()
    return out


# ---------------------------------------------------------------- head-to-head

def h2h_features(df: pd.DataFrame, years: int = 10, last: int = 10) -> pd.DataFrame:
    """From the home club's point of view, over the last `last` meetings in the previous `years` years:
    h2h_n meetings, h2h_gd mean goal difference, h2h_ppg points per game (either venue), h2h_same_gd mean goal
    difference in meetings at this same venue (0 when there were none)."""
    past = defaultdict(list)
    for row in df.itertuples():
        if np.isfinite(row.hg) and np.isfinite(row.ag):
            past[frozenset((row.home, row.away))].append((row.date, row.home, row.hg, row.ag))
    for meetings in past.values():
        meetings.sort(key=lambda r: r[0])
    out = np.zeros((len(df), 4))
    span = pd.Timedelta(days=round(365.25 * years))
    cutoff = pd.to_datetime(df["decision"] if "decision" in df else df.date).dt.normalize()
    for i, (decision, home, away) in enumerate(zip(cutoff, df.home, df.away)):
        prior = [r for r in past[frozenset((home, away))]
                 if decision - span <= r[0] < decision][-last:]
        if prior:
            gd = np.array([(hg - ag) if h == home else (ag - hg) for _, h, hg, ag in prior])
            same = [hg - ag for _, h, hg, ag in prior if h == home]
            out[i] = (len(prior), gd.mean(), np.mean(np.where(gd > 0, 3, np.where(gd == 0, 1, 0))),
                      np.mean(same) if same else 0.0)
    return pd.DataFrame(out, columns=["h2h_n", "h2h_gd", "h2h_ppg", "h2h_same_gd"], index=df.index)
