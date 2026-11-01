"""
Football research: how well do models built only from past results predict English league matches, and does any
of that edge survive the bookmakers' prices? Baselines for the football betting-model plan (ROADMAP: Football).

Procedure (fixed before looking at results):
  1. Data: Premier League, Championship, League One, League Two, 2006/07 → today (football-data.co.uk).
     2006–2015 is history/warm-up only. TEST = every match from 2016/17 on (10 full seasons + the current one).
  2. Models, all strictly pre-match: Elo (+ per-division logistic map fit on the previous 5 seasons), Dixon-Coles
     (half-life 1 year, 4-year window, refit every match day). Base rates = each division's home/draw/away (or
     over/under) frequencies over the previous 5 seasons. Market = Pinnacle closing odds with the margin removed,
     or the closing average where Pinnacle is missing (late 2025/26 on).
  3. Prediction quality on TEST: log-loss, ranked probability score, accuracy; by division and season.
  4. Betting: flat 1-unit bets whenever model probability × price − 1 > edge (0, 5%, 10%), on each price source;
     Betfair Exchange after 5% commission on winnings. Bets at Pinnacle's early price also get closing-line value
     (CLV = price taken × Pinnacle closing fair probability − 1: the expected return if the closing line is right).
  5. Head-to-head: walk-forward multinomial logistic regression (train on TEST seasons before s, test on s),
     base inputs vs base inputs + the four h2h features; per-match log-loss difference ± SE.
     Bases: market, Dixon-Coles, Elo, and nothing (h2h alone vs base rates).
  6. History length: Dixon-Coles with half-lives 90 days → flat 10 years (weekly refits, same TEST matches).

    .venv/bin/python research_football.py            # → reports/football_baselines.md (~5 min, cached after)
    .venv/bin/python research_football.py --fresh    # recompute the Dixon-Coles walks
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import DATA_DIR, PROJECT_ROOT
from src.football import data as fd
from src.football import evaluate as ev
from src.football import models as fm

REPORT = PROJECT_ROOT / "reports" / "football_baselines.md"
CACHE = DATA_DIR / "football" / "cache"
FIRST_SEASON, TEST_START = 2006, 2016
EDGES = (0.0, 0.05, 0.10)
BFE_COMMISSION = 0.05
H2H_COLS = ["h2h_n", "h2h_gd", "h2h_ppg", "h2h_same_gd"]
HALF_LIVES = {"90 days": (90, 365), "180 days": (180, 2 * 365), "1 year": (365, 4 * 365),
              "2 years": (730, 8 * 365), "flat 10 years": (None, 3653)}
ENGLAND = {c: n for c, n in fd.DIVS.items() if fd.LEAGUES[c][0] == "England"}   # this study's scope
PRICES = {"ps": "Pinnacle early", "psc": "Pinnacle closing", "avgc": "Average closing",
          "maxc": "Best closing (any bookmaker)", "bfec": "Betfair Exchange closing"}


def dc_walk(df, fresh, refit_days=1, half_life=365, window_days=4 * 365):
    f = CACHE / f"dc_{half_life}_{window_days}_{refit_days}_{len(df)}_{df.date.max():%Y%m%d}.npz"
    if f.exists() and not fresh:
        z = np.load(f)
        return z["p"], z["over"]
    m = fm.dixon_coles_walk(df, f"{TEST_START}-07-01", refit_days=refit_days, half_life=half_life,
                            window_days=window_days)
    p, over = m[["1x2:home", "1x2:draw", "1x2:away"]].to_numpy(), m["ou_2.5:over"].to_numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, p=p, over=over)
    return p, over


def base_rates(df, y, k, years=5):
    p = np.full((len(df), k), np.nan)
    for (s, d), idx in df.groupby(["season", "div"]).indices.items():
        tr = ((df.season >= s - years) & (df.season < s) & (df["div"] == d)).to_numpy()
        if tr.any():
            p[idx] = np.bincount(y[tr], minlength=k) / tr.sum()
    return p


def finite(*arrays):
    return np.logical_and.reduce([np.isfinite(a).all(1) if a.ndim == 2 else np.isfinite(a) for a in arrays])


def score_table(models: dict, y, mask) -> pd.DataFrame:
    t = pd.DataFrame({name: ev.scores(p[mask], y[mask]) for name, p in models.items()}).T
    t["matches"] = t.matches.astype(int)
    return t.round({"log_loss": 4, "brier": 4, "rps": 4, "accuracy": 3})


def breakdown(df, models: dict, y, mask, by: str) -> pd.DataFrame:
    rows = {}
    for key, idx in df[mask].groupby(by).indices.items():
        sel = np.flatnonzero(mask)[idx]
        r = {"matches": len(sel)}
        for name, p in models.items():
            s = ev.scores(p[sel], y[sel])
            r[f"{name} acc"], r[f"{name} log-loss"] = round(s["accuracy"], 3), round(s["log_loss"], 4)
        rows[fd.season_name(key) if by == "season" else fd.DIVS[key]] = r
    return pd.DataFrame(rows).T


def season_span(df, rows) -> str:
    s = df.season.to_numpy()[rows]
    return f"{fd.season_name(s.min())}–{fd.season_name(s.max())}" if len(s) else "-"


def betting_table(df, models: dict, y, test, market, close_fair) -> pd.DataFrame:
    rows = []
    for name, p in models.items():
        for src, label in PRICES.items():
            o = fd.odds(df, src, market)
            o[~test] = np.nan
            if not np.isfinite(o).all(1).any():
                continue
            for edge in EDGES:
                b = ev.value_bets(p, o, y, edge, BFE_COMMISSION if src == "bfec" else 0.0)
                r = {"model": name, "src": src, "prices": label, "edge": edge,
                     "seasons": season_span(df, np.flatnonzero(np.isfinite(o).all(1))), **ev.bet_summary(b),
                     "clv": np.nan, "clv_se": np.nan, "picks": np.bincount(b.pick, minlength=p.shape[1]) / len(b)}
                if src == "ps":
                    clv = b.odds.to_numpy() * close_fair[b.row.to_numpy(), b.pick.to_numpy()] - 1
                    r["clv"], r["clv_se"] = ev.mean_se(clv[np.isfinite(clv)])
                    r["drift"], r["drift_se"] = clv_drift(o, close_fair, b.row.to_numpy(), b.pick.to_numpy())
                rows.append(r)
    return pd.DataFrame(rows)


def clv_drift(o, close_fair, rows, picks) -> tuple[float, float]:
    """CLV of the picks minus the average CLV of every bet on the same outcome at similar early odds (20 bins) —
    separates information the model has from the outcome/price mix it happens to bet."""
    clv = o * close_fair - 1
    ok = np.isfinite(clv).all(1)
    d = []
    for k in range(o.shape[1]):
        r = rows[(picks == k) & ok[rows]]
        if len(r):
            b = np.digitize(o[:, k], np.quantile(o[ok, k], np.linspace(0, 1, 21)[1:-1]))
            d.append(clv[r, k] - pd.Series(clv[ok, k]).groupby(b[ok]).mean().reindex(b[r]).to_numpy())
    return ev.mean_se(np.concatenate(d))


def blind_clv(df, test, close_fair, market) -> float:
    """CLV of backing every outcome at Pinnacle's early price — what no information at all earns."""
    c = fd.odds(df, "ps", market)[test] * close_fair[test] - 1
    return float(np.nanmean(c))


def show_bets(t: pd.DataFrame) -> str:
    out = pd.DataFrame({"model": t.model, "prices": t.prices, "seasons": t.seasons,
                        "min edge": t.edge.map("{:.0%}".format), "bets": t.bets, "hit rate": t.hit_rate.round(3),
                        "avg odds": t.avg_odds.round(2), "ROI": t.roi.map("{:+.1%}".format),
                        "± SE": t.se.map("{:.1%}".format), "t": t.t.round(1),
                        "CLV": [f"{m:+.1%} ± {s:.1%}" if np.isfinite(m) else "" for m, s in zip(t.clv, t.clv_se)],
                        "CLV vs similar odds": [f"{m:+.2%} ± {s:.2%}" if np.isfinite(m) else ""
                                                for m, s in zip(t.drift, t.drift_se)]})
    return out.to_markdown(index=False)


def row(t: pd.DataFrame, model: str, src: str, edge: float = 0.0) -> pd.Series:
    return t[(t.model == model) & (t.src == src) & (t.edge == edge)].iloc[0]


def h2h_test(df, bases: dict, h2h: pd.DataFrame, y) -> pd.DataFrame:
    """Per-match log-loss with h2h minus without (negative = h2h helps), walk-forward by season."""
    season, n_meet = df.season.to_numpy(), h2h.h2h_n.to_numpy()
    H = h2h[H2H_COLS].to_numpy()
    rows = {}
    for name, X in bases.items():
        X = np.zeros((len(df), 1)) if X is None else X
        ok = finite(X)
        diffs, meets = [], []
        for s in range(TEST_START + 1, season.max() + 1):
            tr, te = ok & (season >= TEST_START) & (season < s), ok & (season == s)
            if not te.any():
                continue
            ll = []
            for Z in (X, np.hstack([X, H])):
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Z[tr], y[tr])
                ll.append(ev.match_log_loss(m.predict_proba(Z[te]), y[te]))
            diffs.append(ll[1] - ll[0])
            meets.append(n_meet[te])
        d, k = np.concatenate(diffs), np.concatenate(meets)
        m, se = ev.mean_se(d)
        m5, se5 = ev.mean_se(d[k >= 5])
        rows[name] = {"matches": len(d), "Δ log-loss": f"{m:+.4f}", "± SE": f"{se:.4f}", "t": round(m / se, 1),
                      "≥5 meetings: matches": int((k >= 5).sum()), "≥5 Δ log-loss": f"{m5:+.4f}",
                      "≥5 ± SE": f"{se5:.4f}"}
    return pd.DataFrame(rows).T


def history_table(df, y, y_ou, test_mask, market, market_ou, fresh) -> pd.DataFrame:
    rows = {}
    for label, (hl, window) in HALF_LIVES.items():
        p, over = dc_walk(df, fresh, refit_days=7, half_life=hl, window_days=window)
        p_ou = np.column_stack([over, 1 - over])
        m = test_mask & finite(p, market)
        mo = test_mask & finite(p_ou, market_ou)
        s, so = ev.scores(p[m], y[m]), ev.scores(p_ou[mo], y_ou[mo])
        rows[f"Dixon-Coles, half-life {label}" if hl else f"Dixon-Coles, {label}"] = {
            "1X2 log-loss": round(s["log_loss"], 4), "1X2 acc": round(s["accuracy"], 3),
            "O/U log-loss": round(so["log_loss"], 4), "O/U acc": round(so["accuracy"], 3)}
    s, so = ev.scores(market[m], y[m]), ev.scores(market_ou[mo], y_ou[mo])
    rows["Market (closing)"] = {"1X2 log-loss": round(s["log_loss"], 4), "1X2 acc": round(s["accuracy"], 3),
                                "O/U log-loss": round(so["log_loss"], 4), "O/U acc": round(so["accuracy"], 3)}
    return pd.DataFrame(rows).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="recompute Dixon-Coles walks instead of using the cache")
    args = ap.parse_args()

    seasons = range(FIRST_SEASON, date.today().year if date.today().month < 7 else date.today().year + 1)
    fd.download(seasons, ENGLAND)
    df = fd.load(seasons, ENGLAND)
    y, y_ou = df.result.to_numpy(), 1 - df.over.to_numpy()          # y_ou: 0 over, 1 under
    test = (df.season >= TEST_START).to_numpy()
    print(f"{len(df)} matches, TEST {test.sum()} from {fd.season_name(TEST_START)}", flush=True)

    elo = fm.elo_probs(df, fm.elo_diff(df), range(TEST_START, df.season.max() + 1))
    dc, dc_over = dc_walk(df, args.fresh)
    dc_ou = np.column_stack([dc_over, 1 - dc_over])
    market, market_src = fd.market_probs(df)
    market_ou, market_ou_src = fd.market_probs(df, "ou")
    ps_early, psc_fair = fd.implied(fd.odds(df, "ps")), fd.implied(fd.odds(df, "psc"))
    psc_ou_fair = fd.implied(fd.odds(df, "psc", "ou"))
    base, base_ou = base_rates(df, y, 3), base_rates(df, y_ou, 2)
    h2h = fm.h2h_features(df)
    print("models done", flush=True)

    # 3. prediction quality
    m_all = test & finite(base, elo, dc, market)
    m_pin = m_all & finite(ps_early)
    q_pin = score_table({"Base rates": base, "Elo": elo, "Dixon-Coles": dc, "Pinnacle early": ps_early,
                         "Market (closing)": market}, y, m_pin)
    q_all = score_table({"Base rates": base, "Elo": elo, "Dixon-Coles": dc, "Market (closing)": market}, y, m_all)
    by_div = breakdown(df, {"Elo": elo, "DC": dc, "Market": market}, y, m_all, "div")
    by_season = breakdown(df, {"Elo": elo, "DC": dc, "Market": market}, y, m_all, "season")
    m_ou = test & finite(base_ou, dc_ou, market_ou)
    q_ou = score_table({"Base rates": base_ou, "Dixon-Coles": dc_ou, "Market (closing)": market_ou}, y_ou, m_ou)

    # 4. betting
    bets_1x2 = betting_table(df, {"Elo": elo, "Dixon-Coles": dc}, y, test, "1x2", psc_fair)
    bets_ou = betting_table(df, {"Dixon-Coles": dc_ou}, y_ou, test, "ou", psc_ou_fair)
    div_rows = []
    for d, name in ENGLAND.items():
        in_div = test & (df["div"] == d).to_numpy()
        for src in ("psc", "avgc", "maxc"):
            o = fd.odds(df, src)
            o[~in_div] = np.nan
            s = ev.bet_summary(ev.value_bets(dc, o, y, 0.05))
            div_rows.append({"division": name, "prices": PRICES[src], "bets": s["bets"], "ROI": f"{s['roi']:+.1%}",
                             "± SE": f"{s['se']:.1%}", "t": round(s["t"], 1)})
    bets_div = pd.DataFrame(div_rows)

    # 5. head-to-head
    h2h_res = h2h_test(df, {"Market (closing)": np.log(market), "Dixon-Coles": np.log(dc),
                            "Elo": (fm.elo_diff(df) / 400).reshape(-1, 1), "Nothing (base rates)": None}, h2h, y)
    print("h2h done", flush=True)

    # 6. history length
    hist = history_table(df, y, y_ou, test, market, market_ou, args.fresh)
    print("history done", flush=True)

    dc_s, mk_s, elo_s = q_all.loc["Dixon-Coles"], q_all.loc["Market (closing)"], q_all.loc["Elo"]
    pin = q_pin.loc["Pinnacle early"]
    dc_early, elo_early = row(bets_1x2, "Dixon-Coles", "ps"), row(bets_1x2, "Elo", "ps")
    dc_close, dc_best = row(bets_1x2, "Dixon-Coles", "psc"), row(bets_1x2, "Dixon-Coles", "maxc")
    ou_early = row(bets_ou, "Dixon-Coles", "ps")
    n_rows = len(bets_1x2) + len(bets_ou)
    n_pos = int((bets_1x2.roi > 0).sum() + (bets_ou.roi > 0).sum())
    max_t = float(pd.concat([bets_1x2, bets_ou]).query("roi > 0").t.max()) if n_pos else np.nan
    dc_hist = hist.drop(index="Market (closing)")
    best_hl, best_hl_ou = (dc_hist[c].idxmin().replace("Dixon-Coles, ", "") for c in ("1X2 log-loss", "O/U log-loss"))
    h2h_d = h2h_res.loc[["Market (closing)", "Dixon-Coles", "Elo"], "Δ log-loss"].astype(float)
    h2h_useless = bool((h2h_res.loc[["Market (closing)", "Dixon-Coles", "Elo"], "t"].astype(float) > -2).all())
    behind_everywhere = bool((by_div["DC log-loss"] > by_div["Market log-loss"]).all()
                             and (by_season["DC log-loss"] > by_season["Market log-loss"]).all())
    blind_1x2, blind_ou = blind_clv(df, test, psc_fair, "1x2"), blind_clv(df, test, psc_ou_fair, "ou")
    lines = [
        "# Football: predicting English league results from past data — baselines",
        "",
        f"Generated {date.today()} by `research_football.py`. Premier League, Championship, League One, League Two. "
        f"History from {fd.season_name(FIRST_SEASON)}; every number below is out-of-sample on "
        f"{test.sum():,} TEST matches from {fd.season_name(TEST_START)} to {df.date.max():%Y-%m-%d} "
        "(each prediction uses only matches played before that day).",
        "",
        "## Summary",
        "",
        f"- **Prediction:** match-result accuracy Elo {elo_s.accuracy:.1%}, Dixon-Coles {dc_s.accuracy:.1%}, "
        f"closing market {mk_s.accuracy:.1%}, always-home-win {np.mean(y[m_all] == 0):.1%} "
        f"(Premier League alone: DC {by_div.loc['Premier League', 'DC acc']:.1%}, market "
        f"{by_div.loc['Premier League', 'Market acc']:.1%}). Log-loss DC {dc_s.log_loss:.4f} vs market "
        f"{mk_s.log_loss:.4f}; Pinnacle's early price {pin.log_loss:.4f}"
        + (" — the models learn real signal but are behind the market in every division and season." if
           behind_everywhere else " (see the division and season tables for where the gap narrows)."),
        f"- **Closing-line value** (the low-noise test of an edge): DC picks at Pinnacle's early price "
        f"{dc_early.clv:+.1%} ± {dc_early.clv_se:.1%}, Elo {elo_early.clv:+.1%}; backing every outcome blindly "
        f"{blind_1x2:+.1%}. Over/under: DC {ou_early.clv:+.1%} vs blind {blind_ou:+.1%}. Against bets on the same "
        f"outcome at similar early odds, DC picks gain {dc_early.drift:+.2%} ± {dc_early.drift_se:.2%} (1X2) and "
        f"{ou_early.drift:+.2%} ± {ou_early.drift_se:.2%} (over/under)"
        + (": the line does drift slightly toward the model between Friday and kick-off — a little real "
           "information — but far less than the bookmaker's margin, so CLV stays negative."
           if min(dc_early.drift - 2 * dc_early.drift_se, ou_early.drift - 2 * ou_early.drift_se) > 0
           and max(dc_early.clv, ou_early.clv) < 0
           else ": no reliable movement toward the models' picks." if max(dc_early.drift - 2 * dc_early.drift_se,
                                                                       ou_early.drift - 2 * ou_early.drift_se) <= 0
           else " — mixed; see section 3."),
        f"- **Betting:** DC into Pinnacle closing {dc_close.roi:+.1%} ± {dc_close.se:.1%} over {dc_close.bets:,} "
        f"bets; at the best price across all bookmakers {dc_best.roi:+.1%} ± {dc_best.se:.1%} (not achievable — "
        f"winning accounts get restricted). The models' \"value\" bets are mostly underdogs (average odds "
        f"{dc_close.avg_odds:.2f}; home/draw/away {'/'.join(f'{x:.0%}' for x in dc_close.picks)}). {n_pos} of {n_rows} betting rows are positive (largest t {max_t:.1f})"
        + ("; across that many rows, a few positives at |t| < 2 are what chance alone produces." if max_t < 2
           else " — see section 3." if n_pos else "."),
        f"- **Head-to-head:** alone vs base rates Δ log-loss {float(h2h_res.loc['Nothing (base rates)', 'Δ log-loss']):+.4f}"
        f" (it proxies team strength); added to the market, Dixon-Coles or Elo {h2h_d.min():+.4f} to "
        f"{h2h_d.max():+.4f}"
        + (" — no information beyond current team strength." if h2h_useless else " — see section 4."),
        f"- **History length:** best Dixon-Coles weighting for match results: {best_hl}; for over/under: "
        f"{best_hl_ou}. Weighting all 10 years equally: 1X2 log-loss "
        f"{hist.loc['Dixon-Coles, flat 10 years', '1X2 log-loss']:.4f} vs {dc_hist['1X2 log-loss'].min():.4f} at best.",
        "",
        "## 1. Prediction quality — match result (home / draw / away)",
        "",
        f"Matches where Pinnacle's early price is also known ({season_span(df, np.flatnonzero(m_pin))}):",
        "",
        q_pin.to_markdown(),
        "",
        f"All TEST matches ({season_span(df, np.flatnonzero(m_all))}; market = Pinnacle closing, or average closing "
        f"for {int((market_src[m_all] == 'avgc').sum())} late matches):",
        "",
        q_all.to_markdown(),
        "",
        "By division:",
        "",
        by_div.to_markdown(),
        "",
        "By season:",
        "",
        by_season.to_markdown(),
        "",
        "## 2. Prediction quality — over / under 2.5 goals",
        "",
        f"{season_span(df, np.flatnonzero(m_ou))} (closing over/under prices start in 2019/20):",
        "",
        q_ou.to_markdown(),
        "",
        "## 3. Betting simulation — flat 1-unit stakes",
        "",
        "Bet the outcome with the highest expected return when model probability × price − 1 exceeds the minimum "
        "edge. ROI = profit per unit staked; t = ROI / SE (|t| < 2 is indistinguishable from luck). "
        "\"Best closing\" needs an account with every bookmaker, and soft bookmakers restrict winning accounts, "
        "so treat it as an upper bound. Betfair Exchange is after 5% commission on winnings.",
        "",
        "Match result:",
        "",
        show_bets(bets_1x2),
        "",
        "Over / under 2.5 goals:",
        "",
        show_bets(bets_ou),
        "",
        "Dixon-Coles match-result bets by division (min edge 5%):",
        "",
        bets_div.to_markdown(index=False),
        "",
        "## 4. Head-to-head meetings — do they add anything?",
        "",
        "Walk-forward logistic regression, base inputs vs base + h2h (meetings in the last 10 years: count, mean goal "
        "difference, points per game, goal difference at this venue). Δ log-loss < 0 means h2h improved the "
        "predictions; |t| < 2 is noise.",
        "",
        h2h_res.to_markdown(),
        "",
        "## 5. How much history helps — Dixon-Coles time decay",
        "",
        "Same TEST matches, weekly refits. Half-life = age at which a match counts half as much as yesterday's.",
        "",
        hist.to_markdown(),
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
