"""
League results, match statistics and bookmaker odds from football-data.co.uk (free CSVs, one per league per season).
Brief: src/football/football_prediction_bot_plan.md.

  LEAGUES  England E0 Premier League, E1 Championship, E2 League One, E3 League Two; the top division of the next
           six UEFA-ranked countries: SP1 La Liga, D1 Bundesliga, I1 Serie A, F1 Ligue 1, N1 Eredivisie,
           P1 Primeira Liga (any football-data code can be added).

Odds timing (football-data notes / matches.php, checked 2026-10-27): "pre-closing" odds are collected on Friday
afternoons (no later than 17:00 UK) for weekend fixtures and on Tuesdays (no later than 13:00 UK) for midweek
fixtures; "closing" odds (an extra C in the column name, e.g. B365CH) are taken just before kick-off. Neither schedule proves individual price availability. Actual captured_at_utc is NULL for historical CSVs;
strict evaluation excludes these quotes. Closing odds are a benchmark only.
The site states no data licence (disclaimer page covers gambling only): personal research use, no redistribution.

Older API kept for research_football.py: load() / odds() / implied() / market_probs() with short source codes
  ps = Pinnacle pre-closing, psc = Pinnacle closing, avgc / maxc = market average / best closing,
  bfec = Betfair Exchange closing.
"""
import re

import numpy as np
import pandas as pd
import requests

from config.settings import DATA_DIR

RAW = DATA_DIR / "football" / "raw"
URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
SOURCE = "football-data.co.uk"
UK = "Europe/London"
# code → (country, name, tier); tier is used for promoted/relegated flags
LEAGUES = {"E0": ("England", "Premier League", 1), "E1": ("England", "Championship", 2),
           "E2": ("England", "League One", 3), "E3": ("England", "League Two", 4),
           "SP1": ("Spain", "La Liga", 1), "D1": ("Germany", "Bundesliga", 1), "I1": ("Italy", "Serie A", 1),
           "F1": ("France", "Ligue 1", 1), "N1": ("Netherlands", "Eredivisie", 1),
           "P1": ("Portugal", "Primeira Liga", 1)}
DIVS = {code: name for code, (_, name, _) in LEAGUES.items()}
ODDS_1X2 = {"ps": ("PSH", "PSD", "PSA"), "psc": ("PSCH", "PSCD", "PSCA"), "avgc": ("AvgCH", "AvgCD", "AvgCA"),
            "maxc": ("MaxCH", "MaxCD", "MaxCA"), "bfec": ("BFECH", "BFECD", "BFECA")}
ODDS_OU = {"ps": ("P>2.5", "P<2.5"), "psc": ("PC>2.5", "PC<2.5"), "avgc": ("AvgC>2.5", "AvgC<2.5"),
           "maxc": ("MaxC>2.5", "MaxC<2.5"), "bfec": ("BFEC>2.5", "BFEC<2.5")}
OUTCOMES = ("h", "d", "a")

# column prefix → bookmaker. Betbrain (Bb*) averages/maxima predate the market Avg/Max columns (2019/20).
BOOKMAKERS = {"1XB": "1xbet", "B365": "bet365", "BFE": "betfair_exchange", "BFD": "betfred", "BF": "betfair",
              "BMGM": "betmgm", "BV": "betvictor", "VC": "betvictor", "BW": "bet_and_win", "CL": "coral",
              "IW": "interwetten", "LB": "ladbrokes", "PP": "paddy_power", "PS": "pinnacle", "P": "pinnacle",
              "SKB": "skybet", "WH": "william_hill", "Avg": "market_avg", "BbAv": "market_avg",
              "Max": "market_max", "BbMx": "market_max"}
_PREFIX = "|".join(sorted(map(re.escape, BOOKMAKERS), key=len, reverse=True))   # longest first: BFDH = Betfred home
_COL_1X2 = re.compile(rf"^({_PREFIX})(C?)([HDA])$")
_COL_OU = re.compile(rf"^({_PREFIX})(C?)([<>])2\.5$")
_SEL = {"H": "home", "D": "draw", "A": "away", ">": "over", "<": "under"}
STATS = {"HTHG": "hthg", "HTAG": "htag", "HS": "home_shots", "AS": "away_shots", "HST": "home_shots_target",
         "AST": "away_shots_target", "HC": "home_corners", "AC": "away_corners", "HF": "home_fouls",
         "AF": "away_fouls", "HY": "home_yellow", "AY": "away_yellow", "HR": "home_red", "AR": "away_red",
         "HxG": "home_xg", "AxG": "away_xg"}


def season_code(start: int) -> str:
    """2016 → '1617' (the URL and file naming used by football-data.co.uk)."""
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def season_name(start: int) -> str:
    return f"{start}/{(start + 1) % 100:02d}"


def raw_path(start: int, div: str):
    return RAW / f"{season_code(start)}_{div}.csv"


def download(starts, divs=DIVS, refresh_latest: bool = True) -> None:
    """Fetch any missing season files; the latest season is re-fetched because it is still being played."""
    RAW.mkdir(parents=True, exist_ok=True)
    latest = max(starts)
    for s in starts:
        for d in divs:
            f = raw_path(s, d)
            if f.exists() and f.stat().st_size and not (refresh_latest and s == latest):
                continue
            r = requests.get(URL.format(code=season_code(s), div=d), timeout=30)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            f.write_bytes(r.content)


def team_name(name: str) -> str:
    """Normalised club name (football-data names are consistent across seasons; fixture APIs get mapped onto these)."""
    return " ".join(str(name).split())


def capture_dates(match_date: pd.Series) -> pd.Series:
    """Day football-data collected the pre-closing odds: the Friday on/before a Fri–Mon match, the Tuesday on/before
    a Tue–Thu match. Results from that day on were not known at capture time."""
    wd = match_date.dt.weekday.to_numpy()                  # Mon 0 … Sun 6
    back = np.where(wd >= 4, wd - 4, np.where(wd == 0, 3, wd - 1))
    return match_date - pd.to_timedelta(back, unit="D")


def capture_times_utc(match_date: pd.Series, earlier: pd.Series | None = None) -> pd.Series:
    """Latest pre-closing collection time: Friday 17:00 UK or Tuesday 13:00 UK, as naive UTC. Rows flagged `earlier`
    (kick-off before that time, e.g. Boxing Day / Good Friday afternoons) move back one slot: Friday → Tuesday,
    Tuesday → previous Friday."""
    day = capture_dates(match_date)
    if earlier is not None:
        back = np.where(day.dt.weekday == 4, 3, 4) * np.asarray(earlier, bool)
        day = day - pd.to_timedelta(back, unit="D")
    hour = np.where(day.dt.weekday == 4, 17, 13)
    local = (day + pd.to_timedelta(hour, unit="h")).dt.tz_localize(UK, ambiguous=True, nonexistent="shift_forward")
    return local.dt.tz_convert("UTC").dt.tz_localize(None)


def parse_file(path, league: str, season: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One football-data CSV → (matches, odds snapshots in long form, validation counts). Invalid rows are dropped
    and counted, never silently kept."""
    raw = pd.read_csv(path, encoding="latin-1")
    raw.columns = [c.replace("ï»¿", "").strip() for c in raw.columns]
    issues = {"rows": len(raw)}
    raw = raw.dropna(how="all")
    bad = raw[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"]].isna().any(axis=1)
    issues["missing_teams_scores_or_date"] = int(bad.sum())
    raw = raw[~bad].copy()
    raw["home"], raw["away"] = raw.HomeTeam.map(team_name), raw.AwayTeam.map(team_name)
    raw["match_date"] = pd.to_datetime(raw.Date, dayfirst=True, format="mixed").dt.normalize()
    same = raw.home == raw.away
    neg = (pd.to_numeric(raw.FTHG, errors="coerce") < 0) | (pd.to_numeric(raw.FTAG, errors="coerce") < 0)
    issues["home_equals_away"], issues["negative_score"] = int(same.sum()), int(neg.sum())
    raw = raw[~same & ~neg]
    dup = raw.duplicated(["match_date", "home", "away"])
    issues["duplicates"] = int(dup.sum())
    raw = raw[~dup]
    lo, hi = pd.Timestamp(season, 7, 1), pd.Timestamp(season + 1, 8, 15)     # 2019/20 finished in July 2020
    issues["date_outside_season"] = int(((raw.match_date < lo) | (raw.match_date > hi)).sum())

    m = pd.DataFrame({"league": league, "season": season, "match_date": raw.match_date,
                      "home": raw.home, "away": raw.away,
                      "fthg": raw.FTHG.astype(int), "ftag": raw.FTAG.astype(int)})
    m["match_id"] = (league + "-" + m.match_date.dt.strftime("%Y%m%d") + "-"
                     + m.home.str.replace(r"\W+", "", regex=True) + "-" + m.away.str.replace(r"\W+", "", regex=True))
    kick = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if "Time" in raw:
        t = pd.to_datetime(raw.Date.astype(str) + " " + raw.Time.astype(str), dayfirst=True, format="mixed",
                           errors="coerce")
        kick = t.dt.tz_localize(UK, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC").dt.tz_localize(None)
    m["kickoff_utc"] = kick
    for col, name in STATS.items():
        m[name] = pd.to_numeric(raw[col], errors="coerce") if col in raw else np.nan
    m["referee"] = raw["Referee"].astype("string") if "Referee" in raw else pd.NA
    # Collection schedules are estimates, NOT proof of when an individual price existed.
    # Never backdate a holiday price to the previous collection slot to make it usable.
    rule = capture_times_utc(m.match_date)
    unproven = (rule >= m.kickoff_utc) | m.kickoff_utc.isna()
    m["pre_closing_captured_utc"] = rule.mask(unproven)
    issues["kickoff_unknown"] = int(m.kickoff_utc.isna().sum())
    issues["capture_moved_earlier"] = 0  # compatibility: prices are no longer moved
    issues["capture_unproven"] = int(unproven.sum())

    long = []
    for col in raw.columns:
        hit = _COL_1X2.match(col) or _COL_OU.match(col)
        if not hit:
            continue
        prefix, closing, sel = hit.groups()
        v = pd.to_numeric(raw[col], errors="coerce")
        long.append(pd.DataFrame({"match_id": m.match_id, "bookmaker": BOOKMAKERS[prefix],
                                  "market": "1x2" if sel in "HDA" else "ou_2.5", "selection": _SEL[sel],
                                  "snapshot": "closing" if closing else "pre_closing", "odds": v,
                                  "prefix": prefix}))
    odds = pd.concat(long, ignore_index=True) if long else pd.DataFrame(
        columns=["match_id", "bookmaker", "market", "selection", "snapshot", "odds", "prefix"])
    odds = odds.dropna(subset=["odds"])
    issues["odds_not_above_1"] = int((odds.odds <= 1.0).sum())
    odds = odds[odds.odds > 1.0]
    # market Avg/Max win over the older Betbrain columns if a file ever carries both; then keep complete sets only
    odds["rank"] = odds.prefix.str.startswith("Bb").astype(int)
    odds = odds.sort_values("rank").drop_duplicates(["match_id", "bookmaker", "market", "selection", "snapshot"])
    size = odds.groupby(["match_id", "bookmaker", "market", "snapshot"]).selection.transform("size")
    full = np.where(odds.market == "1x2", 3, 2)
    issues["odds_incomplete_sets"] = int((size != full).sum())
    odds = odds[size == full].drop(columns=["prefix", "rank"])
    # sum of implied probabilities: a bookmaker's is above 1; best-of-market can dip below; exchange back prices
    # sit just above 1 in a liquid market (wider = illiquid/stale, below 1 = not simultaneous)
    book = odds.groupby(["match_id", "bookmaker", "market", "snapshot"]).odds.transform(lambda o: (1 / o).sum())
    lo = odds.bookmaker.map({"market_max": 0.90, "betfair_exchange": 0.97}).fillna(1.0)
    hi = odds.bookmaker.map({"betfair_exchange": 1.15}).fillna(1.35)
    odd_margin = (book < lo) | (book > hi)
    issues["odds_implausible_margin"] = int(odd_margin.sum())
    odds = odds[~odd_margin]
    when = m.set_index("match_id")
    odds["captured_at_utc"] = pd.NaT
    odds["captured_at_utc"] = pd.to_datetime(odds.captured_at_utc)
    odds["captured_at_estimated"] = True  # actual capture time is unknown; schedule retained on match only
    odds["source"] = SOURCE
    return m.reset_index(drop=True), odds.reset_index(drop=True), issues


def _col(raw: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(raw[c], errors="coerce") if c in raw else pd.Series(np.nan, index=raw.index)


def load(starts, divs=DIVS) -> pd.DataFrame:
    """One row per played match straight from the CSVs, sorted by date. result: 0 home win, 1 draw, 2 away win."""
    frames = []
    for s in starts:
        for d in divs:
            f = raw_path(s, d)
            if not f.exists():
                continue
            raw = pd.read_csv(f, encoding="latin-1").dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            df = pd.DataFrame({"season": s, "div": d,
                               "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
                               "home": raw["HomeTeam"].str.strip(), "away": raw["AwayTeam"].str.strip(),
                               "hg": raw["FTHG"].astype(int), "ag": raw["FTAG"].astype(int)})
            for src, cols in ODDS_1X2.items():
                for o, c in zip(OUTCOMES, cols):
                    df[f"{src}_{o}"] = _col(raw, c)
            for src, (over, under) in ODDS_OU.items():
                df[f"{src}_o"], df[f"{src}_u"] = _col(raw, over), _col(raw, under)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], [0, 1], 2)
    df["over"] = (df.hg + df.ag > 2).astype(int)
    return df.sort_values(["date", "div", "home"], kind="stable").reset_index(drop=True)


def odds(df: pd.DataFrame, src: str, market: str = "1x2") -> np.ndarray:
    """(n, 3) home/draw/away or (n, 2) over/under decimal odds; NaN where the source has no price."""
    cols = [f"{src}_{o}" for o in (OUTCOMES if market == "1x2" else ("o", "u"))]
    return df[cols].to_numpy(float)


def implied(o: np.ndarray) -> np.ndarray:
    """Bookmaker probabilities with the margin removed proportionally; NaN rows stay NaN."""
    inv = 1.0 / o
    return inv / inv.sum(axis=1, keepdims=True)


def market_probs(df: pd.DataFrame, market: str = "1x2") -> tuple[np.ndarray, np.ndarray]:
    """Closing-market probabilities: Pinnacle closing where published, else the closing average.
    Returns (probs, source) with source 'psc' / 'avgc' / '' per row."""
    p = implied(odds(df, "psc", market))
    src = np.where(np.isfinite(p).all(axis=1), "psc", "").astype(object)
    alt = implied(odds(df, "avgc", market))
    use = (src == "") & np.isfinite(alt).all(axis=1)
    p[use], src[use] = alt[use], "avgc"
    return p, src
