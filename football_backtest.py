"""
Football bot steps 3–5 — baselines, LightGBM and walk-forward evaluation
(brief: src/football/football_prediction_bot_plan.md).

Procedure (fixed before looking at results):
  Data       data/football/football.db (run football_ingest.py first): every league in data.LEAGUES (English top
             four + the top division of Spain, Germany, Italy, France, Netherlands, Portugal), 2016/17 → today.
  Timing     each match is predicted as of its pre-closing odds collection (Friday 17:00 / Tuesday 13:00 UK, one
             slot earlier for holiday afternoons): only results from before that day are used, and hypothetical bets
             use only those pre-closing prices. Closing prices are a benchmark — never a bet price or a model input.
  Splits     2016/17 warm-up · TRAIN 2017/18–2020/21 · TUNE 2021/22–2024/25 (choose Elo K and the Dixon-Coles
             half-life per market by log loss) · TEST 2025/26 held out, scored once with the tuned settings ·
             FORWARD 2026/27 to date, reported separately.
  Models     Frequency: each league's outcome frequencies over the previous ≤ 5 seasons (every market).
             Elo: 1X2 (+ double chance), logistic map per league fit on earlier seasons only.
             Dixon-Coles: one model per country; one score grid per match → 1X2, double chance, over/under
             1.5/2.5/3.5, BTTS.
             LightGBM (step 4): 1X2 from pre-match features (+ optionally the Dixon-Coles probabilities as inputs);
             per season s the booster trains on 2017/18 … s−2 and is calibrated on s−1 (scikit-learn sigmoid /
             isotonic / temperature, or none). Variant, size and calibration chosen on TUNE log loss; compared
             with Dixon-Coles (the simple baseline) per match, and only adopted where it measurably improves.
  Benchmark  bookmaker probabilities, margin removed proportionally: market-average pre-closing (available at
             decision time) and closing (Pinnacle, else market average). The files only price 1X2 and O/U 2.5;
             O/U 1.5/3.5, BTTS and double chance are compared with the frequency baseline only.
  Metrics    log loss, Brier, accuracy, sample size; Δ log loss vs the benchmark ± SE (paired per match);
             calibration of tuned Dixon-Coles on TUNE + TEST; hypothetical flat 1-unit singles at bet365 and
             Betfair Exchange (5% commission) pre-closing prices when model probability × price − 1 > 0 / 5%.

    .venv/bin/python football_backtest.py            # → reports/football_walkforward.md (~10 min first run, cached)
    .venv/bin/python football_backtest.py --fresh    # recompute the Dixon-Coles walks
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import DATA_DIR, PROJECT_ROOT
from src.football import data as fd
from src.football import db
from src.football import evaluate as ev
from src.football import markets as mk
from src.football import ml
from src.football import models as fm

REPORT = PROJECT_ROOT / "reports" / "football_walkforward_corrected.md"
ALLOW_ESTIMATED = False
CACHE = DATA_DIR / "football" / "cache"
SPLITS = {"TRAIN": (2017, 2020), "TUNE": (2021, 2024), "TEST": (2025, 2025), "FORWARD": (2026, 2100)}
ELO_K = (10, 20, 30)
DC_HALF_LIVES = (180, 365, 730)
WALK_START = "2017-07-01"
EDGES = (0.0, 0.05)
BFE_COMMISSION = 0.05
PRICED = {"1x2": ("home", "draw", "away"), "ou_2.5": ("over", "under")}
ML_FIRST_TRAIN = 2017
DC_INPUTS = ["1x2:home", "1x2:draw", "1x2:away", "ou_2.5:over", "btts:yes"]


def in_split(df, name) -> np.ndarray:
    lo, hi = SPLITS[name]
    return ((df.season >= lo) & (df.season <= hi)).to_numpy()


def frequency_baseline(df, years=5) -> pd.DataFrame:
    """Every market from each league's outcome frequencies over the previous `years` seasons."""
    cols = [c for m in mk.MARKETS for c in mk.columns(m)]
    won = pd.DataFrame({c: mk.won(*c.split(":"), df.hg, df.ag) for c in cols}, index=df.index).astype(float)
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    for (s, lg), idx in df.groupby(["season", "div"]).indices.items():
        tr = ((df.season >= s - years) & (df.season < s) & (df["div"] == lg)).to_numpy()
        if tr.any():
            out.iloc[idx] = won[tr].mean().to_numpy()
    return out


def elo_markets(df, k) -> pd.DataFrame:
    diff = fm.elo_diff(df, k=k, by="decision")
    p = fm.elo_probs(df, diff, range(SPLITS["TRAIN"][0], df.season.max() + 1), train_years=5, burn_in=0)
    return mk.from_1x2(p).set_index(df.index)


def dc_markets(df, half_life, fresh) -> pd.DataFrame:
    f = CACHE / f"wf_dc_timing_v2_country_{half_life}_{len(df)}_{df.date.max():%Y%m%d}_{df.decision.astype('int64').sum() % 10**9}.pkl"
    if f.exists() and not fresh:
        return pd.read_pickle(f)
    country = df["div"].map(lambda d: fd.LEAGUES[d][0])
    out = fm.dixon_coles_walk(df, WALK_START, by="decision", group=country, half_life=half_life,
                              window_days=10 * 365)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(f)
    return out


def lightgbm_markets(df, dc_t, tune) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Walk every variant × size, choose (variant, size, calibration) by TUNE 1X2 log loss.
    Returns (chosen markets, tuning table, gain importance of the chosen booster, chosen label)."""
    X = ml.features(df)
    dc_in = dc_t[DC_INPUTS].rename(columns=lambda c: "dc_" + c.replace(":", "_").replace(".", ""))  # LightGBM-safe
    variants = {"features": X, "features + Dixon-Coles": X.join(dc_in)}
    y = df.result.to_numpy()
    runs, imps, rows = {}, {}, {}
    for v, Xv in variants.items():
        for c, params in ml.CONFIGS.items():
            runs[v, c], imps[v, c] = ml.walk(Xv, y, df.season.to_numpy(), range(ML_FIRST_TRAIN + 2,
                                                                                 df.season.max() + 1),
                                             ML_FIRST_TRAIN, params)
            print(f"LightGBM {v}, {c} done", flush=True)
    ok = tune & np.logical_and.reduce([np.isfinite(p).all(1) for r in runs.values() for p in r.values()])
    for (v, c), r in runs.items():
        rows[f"{v} · {c}"] = {m: ev.match_log_loss(p[ok], y[ok]).mean() for m, p in r.items()}
    table = pd.DataFrame(rows).T
    label, method = table.stack().idxmin()
    key = next(k for k in runs if f"{k[0]} · {k[1]}" == label)
    chosen = mk.from_1x2(runs[key][method]).set_index(df.index)
    return chosen, table.round(4), imps[key], f"{label} · calibration {method}"


def bookmaker(conn, df, book, market, snapshot) -> pd.DataFrame:
    o = db.odds_matrix(conn, df.match_id, book, market, snapshot, PRICED[market], allow_estimated=ALLOW_ESTIMATED)
    return pd.DataFrame(fd.implied(o), index=df.index, columns=mk.columns(market))


def closing(conn, df, market) -> pd.DataFrame:
    p = bookmaker(conn, df, "pinnacle", market, "closing")
    alt = bookmaker(conn, df, "market_avg", market, "closing")
    return p.where(p.notna().all(axis=1), alt, axis=0)


def score_rows(models: dict, df, market, mask, ref: str) -> pd.DataFrame:
    """Scores per model on rows where every model has a probability; Δ log loss vs `ref` ± SE."""
    cols = mk.columns(market)
    have = {n: m for n, m in models.items() if set(cols) <= set(m.columns)}
    ok = mask & np.logical_and.reduce([np.isfinite(m[cols].to_numpy()).all(1) for m in have.values()])
    y = mk.outcome(market, df.hg, df.ag)[ok]
    ll_ref = ev.match_log_loss(have[ref][cols].to_numpy()[ok], y)
    rows = {}
    for name, m in have.items():
        p = m[cols].to_numpy()[ok]
        s = ev.scores(p, y)
        d, se = ev.mean_se(ev.match_log_loss(p, y) - ll_ref)
        rows[name] = {"matches": s["matches"], "log loss": round(s["log_loss"], 4), "Brier": round(s["brier"], 4),
                      "accuracy": round(s["accuracy"], 3),
                      f"Δ log loss vs {ref}": "" if name == ref else f"{d:+.4f} ± {se:.4f}"}
    return pd.DataFrame(rows).T


def singles(models: dict, prices: dict, df, mask, market) -> pd.DataFrame:
    cols = mk.columns(market)
    y = mk.outcome(market, df.hg, df.ag)
    rows = []
    for name, m in models.items():
        if not set(cols) <= set(m.columns):
            continue
        for label, (o, commission) in prices.items():
            o = np.where(mask[:, None], o, np.nan)
            for edge in EDGES:
                b = ev.value_bets(m[cols].to_numpy(), o, y, edge, commission)
                s = ev.bet_summary(b)
                rows.append({"model": name, "prices (pre-closing)": label, "min edge": f"{edge:.0%}",
                             "bets": s["bets"], "hit rate": round(s["hit_rate"], 3),
                             "avg odds": round(s["avg_odds"], 2),
                             "ROI": f"{s['roi']:+.1%}" if s["bets"] > 1 else "-",
                             "± SE": f"{s['se']:.1%}" if s["bets"] > 1 else "-",
                             "t": round(s["t"], 1) if s["bets"] > 1 else "-"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    global ALLOW_ESTIMATED
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--allow-estimated-odds", action="store_true",
                    help="Sensitivity analysis only: historical quote times are unverified")
    args = ap.parse_args()
    ALLOW_ESTIMATED = args.allow_estimated_odds
    eng = db.engine()
    with eng.connect() as conn:
        df = db.load_matches(conn)
        if df.empty:
            sys.exit("database is empty — run football_ingest.py first")
        if not ALLOW_ESTIMATED:
            verified = conn.exec_driver_sql("SELECT count(*) FROM odds_snapshots WHERE captured_at_estimated=0 "
                                            "AND captured_at_utc IS NOT NULL").scalar()
            if not verified:
                sys.exit("No verified historical quote timestamps. Strict odds backtest unavailable. "
                         "Existing reports preserved. --allow-estimated-odds is sensitivity analysis only.")
        pre = {m: bookmaker(conn, df, "market_avg", m, "pre_closing") for m in PRICED}
        close = {m: closing(conn, df, m) for m in PRICED}
        bet365 = {m: db.odds_matrix(conn, df.match_id, "bet365", m, "pre_closing", PRICED[m], allow_estimated=ALLOW_ESTIMATED) for m in PRICED}
        bfe = {m: db.odds_matrix(conn, df.match_id, "betfair_exchange", m, "pre_closing", PRICED[m], allow_estimated=ALLOW_ESTIMATED) for m in PRICED}
    print(f"{len(df)} matches {df.date.min():%Y-%m-%d} → {df.date.max():%Y-%m-%d}", flush=True)

    freq = frequency_baseline(df)
    elo = {k: elo_markets(df, k) for k in ELO_K}
    dc = {}
    for hl in DC_HALF_LIVES:
        dc[hl] = dc_markets(df, hl, args.fresh)
        print(f"Dixon-Coles half-life {hl} done", flush=True)

    # tuning on TUNE seasons only
    tune = in_split(df, "TUNE")
    y1 = mk.outcome("1x2", df.hg, df.ag)
    tune_elo = {}
    for k, m in elo.items():
        ok = tune & np.isfinite(m[mk.columns("1x2")].to_numpy()).all(1)
        tune_elo[f"K = {k}"] = ev.match_log_loss(m[mk.columns("1x2")].to_numpy()[ok], y1[ok]).mean()
    best_k = ELO_K[int(np.argmin(list(tune_elo.values())))]
    tune_dc, best_hl = {}, {}
    for market in mk.EXCLUSIVE:
        cols, y = mk.columns(market), mk.outcome(market, df.hg, df.ag)
        ok = tune & np.logical_and.reduce([np.isfinite(dc[hl][cols].to_numpy()).all(1) for hl in DC_HALF_LIVES])
        tune_dc[market] = {f"{hl} days": ev.match_log_loss(dc[hl][cols].to_numpy()[ok], y[ok]).mean()
                           for hl in DC_HALF_LIVES}
        best_hl[market] = DC_HALF_LIVES[int(np.argmin(list(tune_dc[market].values())))]
    # tuned Dixon-Coles: each market from its own best half-life; double chance follows 1X2
    dc_t = pd.concat([dc[best_hl[m]][mk.columns(m)] for m in mk.EXCLUSIVE]
                     + [dc[best_hl["1x2"]][mk.columns("double_chance")]], axis=1)
    elo_t = elo[best_k]
    tuning = pd.DataFrame(tune_dc).T.round(4)
    tuning["chosen"] = [f"{best_hl[m]} days" for m in tuning.index]
    gbm, gbm_tuning, gbm_imp, gbm_label = lightgbm_markets(df, dc_t, tune)

    def models_for(market):
        out = {"Frequency": freq, "Elo": elo_t, "Dixon-Coles": dc_t, "LightGBM": gbm}
        if market in PRICED:
            out |= {"Market pre-closing": pre[market], "Closing (benchmark)": close[market]}
        return out

    def split_tables(name):
        mask = in_split(df, name)
        parts = []
        for market in mk.EXCLUSIVE:
            ref = "Market pre-closing" if market in PRICED else "Frequency"
            t = score_rows(models_for(market), df, market, mask, ref)
            parts += [f"**{market}**", "", t.to_markdown(), ""]
        return parts

    # calibration of tuned Dixon-Coles on TUNE + TEST
    cal_mask = in_split(df, "TUNE") | in_split(df, "TEST")
    cal = {}
    for market in mk.MARKETS:
        cols = mk.columns(market)
        won = np.stack([mk.won(*c.split(":"), df.hg, df.ag) for c in cols], axis=1)
        t = ev.calibration(dc_t[cols].to_numpy()[cal_mask], won[cal_mask])
        cal[market] = [f"{r.predicted:.0%} / {r.observed:.0%} ({r.n})" for r in t.itertuples()]
        cal[market] = pd.Series(cal[market], index=t.index)
    won1 = np.stack([mk.won("1x2", s, df.hg, df.ag) for s in PRICED["1x2"]], axis=1)
    t = ev.calibration(gbm[mk.columns("1x2")].to_numpy()[cal_mask], won1[cal_mask])
    cal["1x2 (LightGBM)"] = pd.Series([f"{r.predicted:.0%} / {r.observed:.0%} ({r.n})" for r in t.itertuples()],
                                      index=t.index)
    cal_table = pd.DataFrame(cal).fillna("")

    # step 4 question: does LightGBM beat the simple baseline? + what it relies on + where each model stands
    ml_vs_base = {name: score_rows({"Dixon-Coles": dc_t, "Elo": elo_t, "LightGBM": gbm}, df, "1x2",
                                   in_split(df, name), "Dixon-Coles") for name in ("TUNE", "TEST", "FORWARD")}
    c1x2, y1x2 = mk.columns("1x2"), mk.outcome("1x2", df.hg, df.ag)
    ok = tune & np.isfinite(gbm[c1x2].to_numpy()).all(1) & np.isfinite(dc_t[c1x2].to_numpy()).all(1)
    d_tune, se_tune = ev.mean_se(ev.match_log_loss(gbm[c1x2].to_numpy()[ok], y1x2[ok])
                                 - ev.match_log_loss(dc_t[c1x2].to_numpy()[ok], y1x2[ok]))
    adopt_gbm = d_tune < -2 * se_tune                        # rule fixed in advance, decided on TUNE only
    imp_seasons = [s for s in gbm_imp.columns if SPLITS["TUNE"][0] <= s <= SPLITS["TEST"][1]]
    share = (gbm_imp[imp_seasons] / gbm_imp[imp_seasons].sum()).mean(axis=1).sort_values(ascending=False)
    h2h_share = share[share.index.str.startswith("h2h_")].sum()
    imp_table = share.head(15).map("{:.1%}".format).rename("share of gain").to_frame()
    tune_test = in_split(df, "TUNE") | in_split(df, "TEST")
    per_league = {}
    for lg, (country, name, _) in fd.LEAGUES.items():
        mask = tune_test & (df["div"] == lg).to_numpy()
        if not mask.any():
            continue
        t = score_rows({"Market pre-closing": pre["1x2"], "Dixon-Coles": dc_t, "LightGBM": gbm}, df, "1x2", mask,
                       "Market pre-closing")
        d = "Δ log loss vs Market pre-closing"
        per_league[f"{lg} {name}"] = {"matches": int(t.loc["Market pre-closing", "matches"]),
                                      "market accuracy": t.loc["Market pre-closing", "accuracy"],
                                      "market log loss": t.loc["Market pre-closing", "log loss"],
                                      "Dixon-Coles Δ": t.loc["Dixon-Coles", d], "LightGBM Δ": t.loc["LightGBM", d]}
    per_league = pd.DataFrame(per_league).T
    high = {}
    for market in mk.MARKETS:
        cols = mk.columns(market)
        p = dc_t[cols].to_numpy()[cal_mask].ravel()
        won = np.stack([mk.won(*c.split(":"), df.hg, df.ag) for c in cols], axis=1)[cal_mask].ravel()
        sel = np.isfinite(p) & (p >= 0.7) & (p <= 0.9)
        high[market] = (int(sel.sum()), float(p[sel].mean()) if sel.any() else np.nan,
                        float(won[sel].mean()) if sel.any() else np.nan)

    # hypothetical singles at pre-closing prices
    bets = {}
    for name in ("TUNE", "TEST", "FORWARD"):
        mask = in_split(df, name)
        bets[name] = pd.concat([singles({"Elo": elo_t, "Dixon-Coles": dc_t, "LightGBM": gbm},
                                        {"bet365": (bet365[m], 0.0), "Betfair Exchange": (bfe[m], BFE_COMMISSION)},
                                        df, mask, m).assign(market=m) for m in PRICED], ignore_index=True)
        bets[name] = bets[name][bets[name].bets > 0]

    test = in_split(df, "TEST")
    t1 = score_rows(models_for("1x2"), df, "1x2", test, "Market pre-closing")
    leagues = ", ".join(f"{fd.LEAGUES[lg][1]} ({fd.LEAGUES[lg][0]})" for lg in fd.LEAGUES if (df["div"] == lg).any())
    vs = ml_vs_base["TEST"].loc["LightGBM", "Δ log loss vs Dixon-Coles"]
    lines = [
        "# Football bot — baselines, LightGBM and walk-forward evaluation",
        "",
        f"Generated {date.today()} by `football_backtest.py` from `data/football/football.db` "
        f"({len(df):,} matches, {df.date.min():%Y-%m-%d} → {df.date.max():%Y-%m-%d}; {leagues}). "
        "Brief: `src/football/football_prediction_bot_plan.md`. Paper research only.",
        "",
        "Every prediction is made as of the pre-closing odds collection for that match (Friday 17:00 / Tuesday "
        "13:00 UK; earlier for holiday afternoons) using only results known by then. Bets use only those "
        "pre-closing prices; closing prices are a benchmark. Splits: 2016/17 warm-up · TRAIN 2017/18–2020/21 · "
        "TUNE 2021/22–2024/25 · **TEST 2025/26 (held out)** · FORWARD 2026/27 to date.",
        "",
        "## Summary (TEST 2025/26)",
        "",
        f"- 1X2 accuracy: Dixon-Coles {t1.loc['Dixon-Coles', 'accuracy']:.1%}, Elo {t1.loc['Elo', 'accuracy']:.1%}, "
        f"market pre-closing {t1.loc['Market pre-closing', 'accuracy']:.1%}, closing "
        f"{t1.loc['Closing (benchmark)', 'accuracy']:.1%}, frequency {t1.loc['Frequency', 'accuracy']:.1%} "
        f"({int(t1.loc['Dixon-Coles', 'matches'])} matches). Log loss Dixon-Coles "
        f"{t1.loc['Dixon-Coles', 'log loss']:.4f} vs market pre-closing {t1.loc['Market pre-closing', 'log loss']:.4f}.",
        f"- LightGBM ({gbm_label}, chosen on TUNE): 1X2 accuracy {t1.loc['LightGBM', 'accuracy']:.1%}, log loss "
        f"{t1.loc['LightGBM', 'log loss']:.4f}; per match vs Dixon-Coles {vs} (negative = LightGBM better). "
        "Decision rule, fixed in advance: adopt LightGBM only if it beats Dixon-Coles by more than 2 SE on TUNE. "
        f"TUNE Δ {d_tune:+.4f} ± {se_tune:.4f} → **"
        + ("adopt LightGBM for 1X2" if adopt_gbm else "keep Dixon-Coles as the 1X2 model; no blend") + "**. "
        f"Head-to-head features carry {h2h_share:.1%} of the booster's gain.",
        "- Calibration of tuned Dixon-Coles for selections it rates 70–90% (the typical accumulator leg), "
        "TUNE + TEST: " + "; ".join(f"{m} predicted {v[1]:.0%} → won {v[2]:.0%} (n={v[0]})"
                                     for m, v in high.items() if v[0]) + ".",
        "- The historical files only price 1X2 and over/under 2.5, so expected value for O/U 1.5/3.5, BTTS and "
        "double chance — the usual short-priced accumulator legs — cannot be backtested from this source; the "
        "optimiser needs a timestamped odds feed for those markets (brief step 7).",
        "",
        "## 1. Tuning (TUNE seasons, log loss — lower is better)",
        "",
        f"Elo: {', '.join(f'{k} {v:.4f}' for k, v in tune_elo.items())} → K = {best_k}.",
        "",
        "Dixon-Coles half-life (how fast old results lose weight), chosen per market:",
        "",
        tuning.to_markdown(),
        "",
        "LightGBM 1X2 (TUNE log loss; booster trained on 2017/18 … s−2, calibrated on s−1, predicting s; "
        "columns = calibration method):",
        "",
        gbm_tuning.to_markdown(),
        "",
        f"Chosen: {gbm_label}. Mean TUNE log loss by calibration method: "
        + ", ".join(f"{m} {v:.4f}" for m, v in gbm_tuning.mean().items())
        + " (each calibrator learns from one season, ~4,000 matches).",
        "",
        "## 2. Held-out TEST 2025/26",
        "",
        "Δ log loss is per match against the pre-closing market (1X2, O/U 2.5) or the frequency baseline; "
        "negative = better than the reference; ± is one standard error.",
        "",
        *split_tables("TEST"),
        "## 3. TUNE seasons 2021/22–2024/25 (larger sample; settings were chosen here)",
        "",
        *split_tables("TUNE"),
        "## 4. FORWARD 2026/27 to date",
        "",
        *split_tables("FORWARD"),
        "## 5. Does LightGBM beat the simple baseline? (1X2, per match vs Dixon-Coles)",
        "",
        "Negative Δ = better than Dixon-Coles; ± is one standard error. The brief's rule: no blend or switch "
        "without a measured improvement.",
        "",
        *[part for name, t in ml_vs_base.items() for part in (f"**{name}**", "", t.to_markdown(), "")],
        "What the chosen booster relies on (share of split gain, mean over its 2021/22–2025/26 models; "
        f"head-to-head features together: {h2h_share:.1%}):",
        "",
        imp_table.to_markdown(),
        "",
        "## 6. Per league — 1X2, TUNE + TEST (2021/22–2025/26)",
        "",
        "Δ log loss vs the pre-closing market average for that league (negative = model better).",
        "",
        per_league.to_markdown(),
        "",
        "## 7. Calibration — tuned Dixon-Coles (all markets) and LightGBM (1X2), TUNE + TEST",
        "",
        "Each cell: mean predicted probability / observed win rate (number of selections), pooling every selection "
        "of the market.",
        "",
        cal_table.to_markdown(),
        "",
        "## 8. Hypothetical singles at pre-closing prices (flat 1 unit)",
        "",
        "Bet the selection with the highest expected return when model probability × price − 1 exceeds the "
        "minimum edge. bet365 pre-closing O/U prices start in 2019/20; Betfair Exchange in 2024/25 (5% commission "
        "on winnings). |t| < 2 is indistinguishable from luck.",
        "",
        "TEST 2025/26:",
        "",
        bets["TEST"].to_markdown(index=False),
        "",
        "TUNE 2021/22–2024/25:",
        "",
        bets["TUNE"].to_markdown(index=False),
        "",
        "FORWARD 2026/27 to date:",
        "",
        bets["FORWARD"].to_markdown(index=False),
        "",
        "## Limitations",
        "",
        "- Odds collection times are football-data's stated rule, not per-price timestamps "
        "(`captured_at_estimated = 1`).",
        "- League matches only: cup games are missing, so rest days and form ignore them.",
        "- xG exists only for 2026/27 so far; not used.",
        "- Elo ratings and Dixon-Coles strengths are comparable only within a country (no cross-border matches); "
        "European cup results are not in the data.",
        "- 2020/21 was played mostly behind closed doors (home advantage collapsed); it is the calibration season "
        "for 2021/22 LightGBM predictions.",
        "- Accumulator returns need prices for every leg's market; not backtestable from these files beyond "
        "1X2 + O/U 2.5.",
        "",
    ]
    lines.insert(2, "TIMING CORRECTION: historical odds timestamps are unverified. "
                 "Any estimated-odds results are sensitivity analysis, not executable returns or proof of edge. "
                 "Pre-existing evaluation reports are retained as legacy results.")
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
