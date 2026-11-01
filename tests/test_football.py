"""
Tests for the football research code (src/football): margin removal, no look-ahead in Elo / Dixon-Coles /
head-to-head, and the value-betting arithmetic.
"""
import numpy as np
import pandas as pd

from src.football import data as fd
from src.football import evaluate as ev
from src.football import models as fm


def matches(rows, div="E0", season=2020):
    """rows: (date, home, away, hg, ag)."""
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
    df["date"] = pd.to_datetime(df.date)
    df["season"], df["div"] = season, div
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], [0, 1], 2)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def league(seed=0, days=60, strength=None):
    """Round-robin-ish synthetic league; team T0 is much stronger than T5."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    strength = strength or {t: 0.5 - 0.2 * i for i, t in enumerate(teams)}
    rows, day = [], pd.Timestamp("2020-08-01")
    for d in range(days):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            rows.append((day + pd.Timedelta(days=3 * d), h, a,
                         rng.poisson(np.exp(0.3 + strength[h] - strength[a] + 0.2)),
                         rng.poisson(np.exp(0.3 + strength[a] - strength[h]))))
    return matches(rows)


def test_implied_removes_margin_proportionally():
    o = np.array([[2.0, 3.5, 4.0], [np.nan, 3.0, 3.0]])
    p = fd.implied(o)
    assert np.isclose(p[0].sum(), 1.0)
    assert np.isclose(p[0, 0] / p[0, 2], 4.0 / 2.0)
    assert np.isnan(p[1]).all()


def test_market_probs_falls_back_to_closing_average():
    df = pd.DataFrame({f"psc_{o}": [2.0, np.nan] for o in "hda"} | {f"avgc_{o}": [2.1, 2.9] for o in "hda"})
    p, src = fd.market_probs(df)
    assert list(src) == ["psc", "avgc"]
    assert np.allclose(p.sum(1), 1.0)


def test_elo_same_day_results_do_not_leak():
    rows = [("2020-08-01", "A", "B", 3, 0), ("2020-08-08", "A", "C", 0, 0), ("2020-08-08", "B", "D", 0, 0)]
    base = fm.elo_diff(matches(rows))
    changed = fm.elo_diff(matches(rows[:1] + [("2020-08-08", "A", "C", 0, 5), ("2020-08-08", "B", "D", 4, 0)]))
    assert np.allclose(base, changed)                      # day-2 predictions ignore day-2 results
    assert base[1] > base[2]                               # A beat B on day 1 → A rated above B afterwards


def test_dixon_coles_learns_strength_and_probabilities_sum_to_one():
    df = league()
    m = fm.DixonColes(half_life=None, window_days=3650).fit(df, df.date.max() + pd.Timedelta(days=1))
    p, over = m.predict(matches([("2021-01-01", "T0", "T5", 0, 0), ("2021-01-01", "T5", "T0", 0, 0)]))
    assert np.allclose(p.sum(1), 1.0) and ((over > 0) & (over < 1)).all()
    assert p[0, 0] > 0.6 and p[1, 2] > 0.4                 # strong T0 favoured home and away
    assert m.home > 0


def test_dixon_coles_walk_never_sees_the_day_it_predicts():
    df = league(days=30)
    cut = df.date.unique()[20]
    later = df.date >= cut
    scrambled = df.copy()
    scrambled.loc[later, ["hg", "ag"]] = scrambled.loc[later, ["ag", "hg"]].to_numpy() + 3
    a = fm.dixon_coles_walk(df, df.date.min() + pd.Timedelta(days=10)).to_numpy()
    b = fm.dixon_coles_walk(scrambled, df.date.min() + pd.Timedelta(days=10)).to_numpy()
    first_day = (df.date == cut).to_numpy()
    assert np.allclose(a[first_day], b[first_day])
    assert not np.allclose(a[(df.date > cut).to_numpy()], b[(df.date > cut).to_numpy()])


def test_head_to_head_uses_prior_meetings_from_home_side():
    df = matches([("2010-09-01", "A", "B", 1, 1), ("2019-01-01", "A", "B", 2, 0), ("2020-06-01", "B", "A", 0, 0),
                  ("2020-10-01", "A", "B", 9, 0)])
    h = fm.h2h_features(df, years=10)
    assert (h.iloc[0] == 0).all()                          # no earlier meeting
    assert h.iloc[2].h2h_n == 2 and h.iloc[2].h2h_gd == -1.0    # B at home: −2 (2019) and 0 (2010) → mean −1
    assert h.iloc[2].h2h_same_gd == 0.0                    # B has never hosted A
    assert h.iloc[3].h2h_n == 2                            # 2010 meeting is now older than 10 years


def test_value_bets_pick_best_expected_return_after_commission():
    p = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, np.nan]])
    o = np.array([[2.2, 3.0, 5.0], [1.9, 3.0, 4.0], [3.0, 3.0, 3.0]])
    y = np.array([0, 1, 0])
    b = ev.value_bets(p, o, y, edge=0.05)
    assert list(b.row) == [0]                              # row 1 has no 5% edge, row 2 has a missing probability
    assert b.pick[0] == 0 and np.isclose(b.profit[0], 1.2)
    with_fee = ev.value_bets(p, o, y, edge=0.0, commission=0.05)
    assert np.isclose(with_fee.profit[0], 1.2 * 0.95)


def test_rps_is_zero_for_certain_correct_forecasts():
    y = np.array([0, 2])
    assert ev.rps(np.eye(3)[y], y) == 0.0
    assert ev.rps(np.array([[0.0, 0.0, 1.0]]), np.array([0])) == 1.0
