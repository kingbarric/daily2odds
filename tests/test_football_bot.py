"""
Tests for the football bot pipeline (brief: src/football/football_prediction_bot_plan.md): CSV parsing and odds
timing, idempotent SQLite ingestion, coherent market probabilities and settlement, and no future-data leakage in
features / Elo / Dixon-Coles at the decision time.
"""
import numpy as np
import pandas as pd
import pytest

from src.football import data as fd
from src.football import db
from src.football import evaluate as ev
from src.football import features as ff
from src.football import markets as mk
from src.football import ml
from src.football import models as fm

CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,HS,AS,B365H,B365D,B365A,BFH,BFD,BFA,BFDH,BFDD,BFDA,PSCH,PSCD,PSCA,B365>2.5,B365<2.5
E1,15/08/2026,15:00,Leeds ,Hull,2,1,1,0,12,8,1.8,3.6,4.5,1.85,3.7,4.6,1.75,3.5,4.4,1.9,3.5,4.2,1.9,1.9
E1,15/08/2026,15:00,Leeds,Hull,2,1,1,0,12,8,1.8,3.6,4.5,1.85,3.7,4.6,1.75,3.5,4.4,1.9,3.5,4.2,1.9,1.9
E1,18/08/2026,19:45,Hull,Stoke,0,0,0,0,5,6,1.0,3.4,3.9,2.1,3.3,3.6,,3.2,3.5,2.0,3.3,3.9,2.1,1.7
E1,26/12/2025,15:00,Stoke,Leeds,1,3,0,1,9,14,3.5,3.4,2.1,3.6,3.5,2.1,3.4,3.3,2.0,3.7,3.4,2.1,1.8,2.0
"""


def parsed(tmp_path, text=CSV, season=2026):
    f = tmp_path / "x.csv"
    f.write_text(text)
    return fd.parse_file(f, "E1", season)


def test_parse_maps_bookmakers_snapshots_and_drops_bad_rows(tmp_path):
    m, o, iss = parsed(tmp_path)
    assert iss["duplicates"] == 1 and len(m) == 3            # second Leeds–Hull row is a duplicate (after name strip)
    leeds_hull = o[o.match_id == m.match_id[0]]
    books = leeds_hull.groupby(["bookmaker", "snapshot", "market"]).size().to_dict()
    assert books[("betfred", "pre_closing", "1x2")] == 3     # BFDH/BFDD/BFDA = Betfred
    assert books[("betfair", "pre_closing", "1x2")] == 3     # BFH/BFD/BFA = Betfair (BFD is its draw)
    assert books[("pinnacle", "closing", "1x2")] == 3
    assert books[("bet365", "pre_closing", "ou_2.5")] == 2
    hull_stoke = o[o.match_id == m.match_id[1]]
    assert not ((hull_stoke.bookmaker == "bet365") & (hull_stoke.market == "1x2")).any()   # odds 1.0 → set dropped
    assert not (hull_stoke.bookmaker == "betfred").any()      # incomplete set dropped
    assert iss["odds_not_above_1"] == 1 and iss["odds_incomplete_sets"] >= 2


def test_kickoff_and_capture_times_are_utc_and_before_kickoff(tmp_path):
    m, o, _ = parsed(tmp_path)
    sat = m.iloc[0]                                           # Sat 15 Aug 2026 15:00 BST
    assert sat.kickoff_utc == pd.Timestamp("2026-08-15 14:00")
    assert sat.pre_closing_captured_utc == pd.Timestamp("2026-08-14 16:00")     # Friday 17:00 BST
    tue = m.iloc[1]                                           # Tue 18 Aug 19:45 → Tuesday 13:00 BST
    assert tue.pre_closing_captured_utc == pd.Timestamp("2026-08-18 12:00")
    closing = o[(o.match_id == sat.match_id) & (o.snapshot == "closing")]
    assert closing.captured_at_utc.isna().all()
    assert o.captured_at_utc.isna().all()
    assert o.captured_at_estimated.all()


def test_holiday_afternoon_moves_capture_to_previous_slot(tmp_path):
    m, _, iss = parsed(tmp_path)
    boxing = m[m.match_date == "2025-12-26"].iloc[0]          # Friday 15:00 kick-off, before Friday 17:00
    assert pd.isna(boxing.pre_closing_captured_utc)  # availability unknown, never backdated
    assert iss["capture_moved_earlier"] == 0 and iss["capture_unproven"] == 1


def test_capture_dates_rule_for_every_weekday():
    days = pd.Series(pd.date_range("2026-08-10", "2026-08-16"))   # Mon … Sun
    got = fd.capture_dates(days).dt.strftime("%a %d").tolist()
    assert got == ["Fri 07", "Tue 11", "Tue 11", "Tue 11", "Fri 14", "Fri 14", "Fri 14"]


def test_ingest_is_idempotent_and_loads_decision_dates(tmp_path):
    m, o, _ = parsed(tmp_path)
    eng = db.engine(tmp_path / "t.db")
    for _ in range(2):
        with eng.begin() as conn:
            db.ingest(conn, m, o, fd.SOURCE)
    with eng.connect() as conn:
        assert conn.exec_driver_sql("select count(*) from matches").scalar() == 3
        assert conn.exec_driver_sql("select count(*) from odds_snapshots").scalar() == len(o)
        assert conn.exec_driver_sql("select count(*) from teams").scalar() == 3
        df = db.load_matches(conn)
        x = db.odds_matrix(conn, df.match_id, "pinnacle", "1x2", "closing", ("home", "draw", "away"), allow_estimated=True)
    assert list(df.home) == ["Stoke", "Leeds", "Hull"]        # date-sorted
    assert df.decision.tolist() == [pd.Timestamp("2025-12-26"), pd.Timestamp("2026-08-14"), pd.Timestamp("2026-08-18")]
    assert np.allclose(x[1], [1.9, 3.5, 4.2]) and np.isfinite(x).all()


def test_market_probabilities_are_coherent():
    rng = np.random.default_rng(1)
    grid = rng.random((5, 11, 11))
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    p = mk.from_grid(grid)
    for market in mk.EXCLUSIVE:
        assert np.allclose(p[mk.columns(market)].sum(axis=1), 1.0)
    assert (p["ou_1.5:over"] >= p["ou_2.5:over"]).all() and (p["ou_2.5:over"] >= p["ou_3.5:over"]).all()
    assert np.allclose(p["double_chance:home_draw"], p["1x2:home"] + p["1x2:draw"])
    exact = np.zeros((1, 11, 11))
    exact[0, 1, 1] = 1.0                                       # certain 1-1
    q = mk.from_grid(exact).iloc[0]
    assert q["1x2:draw"] == 1 and q["btts:yes"] == 1 and q["ou_1.5:over"] == 1 and q["ou_2.5:under"] == 1


def test_settlement_edge_cases():
    hg, ag = np.array([0, 1, 2, 3]), np.array([0, 1, 1, 0])
    assert mk.won("1x2", "draw", hg, ag).tolist() == [True, True, False, False]
    assert mk.won("double_chance", "home_away", hg, ag).tolist() == [False, False, True, True]
    assert mk.won("ou_1.5", "under", hg, ag).tolist() == [True, False, False, False]
    assert mk.won("ou_2.5", "over", hg, ag).tolist() == [False, False, True, True]
    assert mk.won("btts", "yes", hg, ag).tolist() == [False, True, True, False]
    assert mk.outcome("1x2", hg, ag).tolist() == [1, 1, 0, 0]


def fixtures(rows):
    """rows: (date, home, away, hg, ag) → frame with decision dates and league E1."""
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
    df["date"] = pd.to_datetime(df.date)
    df["season"], df["div"] = 2025, "E1"
    df["decision"] = fd.capture_dates(df.date)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], [0, 1], 2)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


EASTER = [("2026-03-28", "A", "B", 1, 0), ("2026-03-28", "C", "D", 0, 0),
          ("2026-04-03", "A", "C", 2, 0), ("2026-04-03", "B", "D", 1, 1),      # Good Friday
          ("2026-04-06", "A", "D", 0, 1), ("2026-04-06", "C", "B", 2, 2)]      # Easter Monday, same decision day


def test_features_ignore_results_from_the_decision_day_on():
    base = ff.pre_match_features(fixtures(EASTER))
    changed = fixtures(EASTER[:2] + [("2026-04-03", "A", "C", 0, 6), ("2026-04-03", "B", "D", 5, 0)] + EASTER[4:])
    after = ff.pre_match_features(changed)
    monday = [4, 5]
    pd.testing.assert_frame_equal(base.loc[monday], after.loc[monday])   # Friday results unknown at Friday 17:00
    assert base.loc[4, "home_ppg_5"] == 3.0 and base.loc[4, "home_n_prior"] == 1
    assert base.loc[4, "home_rest_days"] == 9.0                          # fixture dates are public in advance


def test_elo_and_dixon_coles_respect_decision_groups():
    df = fixtures(EASTER)
    changed = fixtures(EASTER[:2] + [("2026-04-03", "A", "C", 0, 6), ("2026-04-03", "B", "D", 5, 0)] + EASTER[4:])
    assert np.allclose(fm.elo_diff(df, by="decision")[4:], fm.elo_diff(changed, by="decision")[4:])
    a = fm.dixon_coles_walk(df, "2026-04-01", by="decision").to_numpy()[4:]
    b = fm.dixon_coles_walk(changed, "2026-04-01", by="decision").to_numpy()[4:]
    assert np.allclose(a, b)
    assert not np.allclose(fm.elo_diff(df, by="date")[4:], fm.elo_diff(changed, by="date")[4:])


def test_season_progress_uses_the_league_size():
    f = ff.pre_match_features(fixtures(EASTER))            # 4 clubs → 6 games in a full season
    assert np.isclose(f.loc[4, "season_progress"], (1 + 1) / 2 / 46)


def test_team_names_never_merge_across_countries(tmp_path):
    with db.engine(tmp_path / "t.db").begin() as conn:
        db.team_ids(conn, ["Arsenal"], "England")
        with pytest.raises(ValueError, match="another country"):
            db.team_ids(conn, ["Arsenal"], "Spain")


def test_dixon_coles_group_fits_each_country_alone():
    rows = [("2025-08-02", "A", "B", 2, 0), ("2025-08-02", "C", "D", 1, 1), ("2025-08-09", "A", "C", 3, 1),
            ("2025-08-09", "B", "D", 0, 2), ("2025-08-16", "A", "D", 1, 0), ("2025-08-16", "C", "B", 2, 2)]
    eng = fixtures(rows).assign(div="E0")
    esp = fixtures([(d, h + "x", a + "x", ag, hg) for d, h, a, hg, ag in rows]).assign(div="SP1")
    both = pd.concat([eng, esp]).sort_values("date", kind="stable").reset_index(drop=True)
    country = both["div"].map(lambda d: fd.LEAGUES[d][0])
    got = fm.dixon_coles_walk(both, "2025-08-09", group=country)
    alone = fm.dixon_coles_walk(esp, "2025-08-09").to_numpy()
    assert np.allclose(got[both["div"] == "SP1"].to_numpy(), alone, equal_nan=True)
    assert got.index.equals(both.index)


def synthetic_seasons(seed=0, per_season=500, seasons=range(2017, 2022)):
    rng = np.random.default_rng(seed)
    season = np.repeat(list(seasons), per_season)
    X = pd.DataFrame(rng.normal(size=(len(season), 3)), columns=["x0", "x1", "x2"])
    X["league"] = pd.Categorical(rng.choice(["E0", "SP1"], len(season)), categories=list(fd.LEAGUES))
    logits = np.stack([1.2 * X.x0, np.zeros(len(X)), -1.2 * X.x0], axis=1)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=r) for r in p])
    return X, y, season


def test_lightgbm_walk_never_learns_from_the_predicted_season():
    X, y, season = synthetic_seasons()
    params = dict(num_leaves=4, n_estimators=50)
    base, imp = ml.walk(X, y, season, [2021], 2017, params)
    scrambled = np.where(season >= 2021, (y + 1) % 3, y)          # results of the predicted season change
    after, _ = ml.walk(X, scrambled, season, [2021], 2017, params)
    te = season == 2021
    for m in ml.CALIBRATIONS:
        assert np.allclose(base[m][te], after[m][te])
        assert np.allclose(base[m][te].sum(axis=1), 1.0) and np.isnan(base[m][~te]).all()
    cal_changed = np.where(season == 2020, (y + 1) % 3, y)        # calibration season: booster must not move
    moved, _ = ml.walk(X, cal_changed, season, [2021], 2017, params)
    assert np.allclose(base["none"][te], moved["none"][te])
    assert not np.allclose(base["sigmoid"][te], moved["sigmoid"][te])
    assert imp[2021]["x0"] == imp[2021].max()                     # the informative feature dominates


def test_brier_and_calibration():
    p = np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]])
    y = np.array([0, 1, 1])
    assert np.isclose(ev.brier(p, y), (0.04 + 0.64 + 0.09) / 3)
    cal = ev.calibration(p[:, 0], y == 0)
    assert cal.loc["80%–90%", "n"] == 2 and cal.loc["80%–90%", "observed"] == 0.5


def test_historical_unknown_timestamps_cannot_be_used_as_verified_quotes(tmp_path):
    m, o, _ = parsed(tmp_path)
    with db.engine(tmp_path / "strict.db").begin() as conn:
        db.ingest(conn, m, o, fd.SOURCE)
        assert np.isnan(db.odds_matrix(conn, m.match_id, "bet365", "1x2", "pre_closing",
                                      ("home", "draw", "away"))).all()
        # A legacy row with a guessed non-null timestamp must also remain excluded.
        conn.exec_driver_sql("UPDATE odds_snapshots SET captured_at_utc='2026-01-01 00:00:00'")
        assert np.isnan(db.odds_matrix(conn, m.match_id, "bet365", "1x2", "pre_closing",
                                      ("home", "draw", "away"))).all()


def test_head_to_head_and_ml_features_ignore_results_after_decision():
    df = fixtures([("2026-03-21", "A", "B", 1, 0), ("2026-04-03", "A", "B", 2, 0),
                   ("2026-04-06", "B", "A", 0, 0)])
    changed = df.copy()
    changed.loc[1:, ["hg", "ag"]] = [9, 8]
    pd.testing.assert_series_equal(fm.h2h_features(df).iloc[2], fm.h2h_features(changed).iloc[2])
    pd.testing.assert_series_equal(ml.features(df).iloc[2], ml.features(changed).iloc[2])
    assert fm.h2h_features(df).iloc[2].h2h_n == 1


def test_elo_delayed_result_is_not_applied_at_its_forecast_date():
    df = fixtures([("2026-03-28", "A", "B", 1, 0), ("2026-04-03", "A", "B", 8, 0),
                   ("2026-04-04", "A", "C", 0, 0)])
    df.loc[1, "decision"] = pd.Timestamp("2026-03-24")
    df.loc[2, "decision"] = pd.Timestamp("2026-04-02")
    changed = df.copy()
    changed.loc[1, ["hg", "ag"]] = [0, 8]
    assert fm.elo_diff(df, by="decision")[2] == fm.elo_diff(changed, by="decision")[2]


def test_dixon_coles_fit_filters_future_results_itself():
    df = fixtures(EASTER)
    changed = df.copy()
    changed.loc[2:, ["hg", "ag"]] = [9, 8]
    cutoff = pd.Timestamp("2026-04-01")
    a = fm.DixonColes().fit(df, cutoff).markets(df.iloc[4:])
    b = fm.DixonColes().fit(changed, cutoff).markets(df.iloc[4:])
    pd.testing.assert_frame_equal(a, b)


def test_verified_quotes_require_prekickoff_time_and_known_by(tmp_path):
    m, o, _ = parsed(tmp_path)
    with db.engine(tmp_path / 'verified.db').begin() as conn:
        db.ingest(conn, m, o, fd.SOURCE)
        # Mark synthetic quotes as verified to exercise strict time comparisons.
        conn.exec_driver_sql("UPDATE odds_snapshots SET captured_at_estimated=0, captured_at_utc='2026-08-15 13:00:00'")
        ids = [m.match_id.iloc[0]]
        args = (conn, ids, 'bet365', '1x2', 'pre_closing', ('home', 'draw', 'away'))
        assert np.isfinite(db.odds_matrix(*args, known_by=pd.Timestamp('2026-08-15 13:30'))).all()
        assert np.isnan(db.odds_matrix(*args, known_by=pd.Timestamp('2026-08-15 12:30'))).all()
        conn.exec_driver_sql("UPDATE odds_snapshots SET captured_at_utc='2026-08-15 14:00:00'")
        assert np.isnan(db.odds_matrix(*args)).all()  # at kickoff is already too late


def test_future_fixture_membership_cannot_change_earlier_features():
    df = fixtures(EASTER)
    base = ml.features(df)
    future = fixtures([('2026-05-30','NEW TEAM','OTHER TEAM',9,8)])
    after = ml.features(pd.concat([df,future], ignore_index=True))
    pd.testing.assert_frame_equal(base, after.iloc[:len(df)].reset_index(drop=True))


def test_unknown_intervening_schedule_does_not_change_rest_feature():
    df = fixtures([('2026-03-21','A','B',1,0),('2026-04-06','A','C',2,0)])
    df.loc[1,'decision']=pd.Timestamp('2026-04-01')
    base=ml.features(df).iloc[1]
    extra=fixtures([('2026-04-03','A','D',9,0)])
    combined=pd.concat([df,extra],ignore_index=True).sort_values('date').reset_index(drop=True)
    after=ml.features(combined).iloc[2]
    pd.testing.assert_series_equal(base,after,check_names=False)


def test_intraday_decision_never_sees_evening_result_from_same_day():
    df=fixtures([('2026-03-28','A','B',1,0),('2026-04-03','A','B',2,0),('2026-04-06','B','A',0,0)])
    df.loc[2,'decision']=pd.Timestamp('2026-04-03 12:00')
    altered=df.copy(); altered.loc[1,['hg','ag']]=[0,9]
    pd.testing.assert_series_equal(ml.features(df).iloc[2],ml.features(altered).iloc[2])
    assert fm.h2h_features(df).iloc[2].h2h_n==1
