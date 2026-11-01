"""
Football bot step 1 — one-command historical ingestion (brief: src/football/football_prediction_bot_plan.md).

Downloads every league-season from 2016/17 to the current season from football-data.co.uk, validates it, upserts it
into data/football/football.db (teams, matches, odds_snapshots) and writes reports/football_data_quality.md
(rows dropped and why, missing values, odds coverage per bookmaker, team match counts). Safe to re-run: the
current season is re-downloaded and every league-season is replaced, never duplicated.

    .venv/bin/python football_ingest.py                 # all LEAGUES, 2016/17 → current season
    .venv/bin/python football_ingest.py --leagues E0 E1
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import PROJECT_ROOT
from src.football import data as fd
from src.football import db

REPORT = PROJECT_ROOT / "reports" / "football_data_quality_corrected.md"
FIRST_SEASON = 2016
MISSING_COLS = {"kickoff_utc": "kick-off time", "hthg": "half-time score", "home_shots": "shots",
                "home_shots_target": "shots on target", "home_corners": "corners", "home_yellow": "cards",
                "referee": "referee", "home_xg": "xG"}


def current_season(today: date) -> int:
    return today.year if today.month >= 7 else today.year - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=list(fd.LEAGUES))
    ap.add_argument("--first", type=int, default=FIRST_SEASON)
    args = ap.parse_args()
    unknown = set(args.leagues) - set(fd.LEAGUES)
    if unknown:
        sys.exit(f"add {sorted(unknown)} to data.LEAGUES (country, name, tier) first")

    latest = current_season(date.today())
    seasons = range(args.first, latest + 1)
    fd.download(seasons, args.leagues)
    eng = db.engine()
    issues, coverage, missing, counts = [], [], [], []
    for s in seasons:
        for lg in args.leagues:
            f = fd.raw_path(s, lg)
            if not f.exists():
                issues.append({"league": lg, "season": fd.season_name(s), "rows": 0, "note": "no file published"})
                continue
            m, o, iss = fd.parse_file(f, lg, s)
            with eng.begin() as conn:
                n_m, n_o = db.ingest(conn, m, o, fd.SOURCE)
            issues.append({"league": lg, "season": fd.season_name(s), **iss, "matches_stored": n_m, "odds_stored": n_o})
            missing.append({"league": lg, "season": fd.season_name(s),
                            **{label: f"{m[c].isna().mean():.0%}" for c, label in MISSING_COLS.items()}})
            full = o.groupby(["bookmaker", "market", "snapshot"]).match_id.nunique() / len(m)
            coverage.append(full.rename(f"{lg} {fd.season_name(s)}"))
            if s < latest:
                games = pd.concat([m.home, m.away]).value_counts()
                expected = 2 * (len(games) - 1)
                odd = games[games != expected]
                counts.append({"league": lg, "season": fd.season_name(s), "teams": len(games),
                               "expected games each": expected,
                               "teams with a different count": ", ".join(f"{t} {n}" for t, n in odd.items()) or "-"})
            print(f"{lg} {fd.season_name(s)}: {n_m} matches, {n_o} odds", flush=True)

    with eng.connect() as conn:
        total = pd.read_sql("select league, season, count(*) n, min(match_date) first, max(match_date) last "
                            "from matches group by league, season order by season, league", conn)
        n_odds = conn.exec_driver_sql("select count(*) from odds_snapshots").scalar()
        n_teams = conn.exec_driver_sql("select count(*) from teams").scalar()
    cov = pd.concat(coverage, axis=1).fillna(0)
    by_book = cov.T.groupby(lambda k: "2016/17–2018/19" if int(k.split()[1][:4]) < 2019 else
                            ("2019/20–2023/24" if int(k.split()[1][:4]) < 2024 else "2024/25 →")).mean().T
    by_book = by_book.map("{:.0%}".format).reset_index()

    lines = [
        "# Football data quality",
        "",
        f"Generated {date.today()} by `football_ingest.py`. Source: football-data.co.uk (no explicit licence on the "
        "site; personal research use, no redistribution). Database: `data/football/football.db` — "
        f"{int(total.n.sum()):,} matches, {n_teams} teams, {n_odds:,} odds snapshots.",
        "",
        "Odds timing correction: historical CSVs provide no independently verified per-price collection "
        "timestamps. Actual odds captured_at_utc is NULL and captured_at_estimated=1. The schedule estimate "
        "is retained on matches only where it precedes a known kickoff. Holiday prices are never backdated. "
        "Strict odds evaluation excludes all estimated quotes, including legacy guessed timestamps. "
        "Closing prices are descriptive benchmarks only; existing evaluation reports are preserved.",
        "",
        "## Validation — rows dropped or flagged per league-season",
        "",
        pd.DataFrame(issues).fillna("").to_markdown(index=False),
        "",
        "## Missing values (share of matches)",
        "",
        pd.DataFrame(missing).to_markdown(index=False),
        "",
        "## Odds coverage — share of matches with a complete price set",
        "",
        by_book.to_markdown(index=False),
        "",
        "## Team match counts (completed seasons)",
        "",
        pd.DataFrame(counts).to_markdown(index=False),
        "",
        "## Stored",
        "",
        total.assign(season=total.season.map(fd.season_name)).to_markdown(index=False),
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
