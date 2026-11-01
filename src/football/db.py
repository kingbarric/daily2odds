"""
SQLite store for the football bot (SQLAlchemy Core). All datetimes are naive UTC.

  teams           team_id, name (normalised), country
  matches         one row per match: league, season, date, kick-off, team ids, scores, half-time, shots, corners,
                  fouls, cards, xG (2026/27 on), referee, pre-closing odds collection time, source
  odds_snapshots  one decimal price per (match, bookmaker, market, selection, snapshot) with captured_at_utc and
                  whether that time is estimated from the source's collection rule

Ingestion is idempotent: a league-season is upserted and its odds replaced, so re-running never duplicates.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import (Boolean, Column, Date, Float, ForeignKey, Index, Integer, MetaData, String, Table,
                        create_engine, delete, insert, select)
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.settings import DATA_DIR
from src.football.data import LEAGUES, STATS, capture_dates, capture_times_utc

DB_PATH = DATA_DIR / "football" / "football.db"
SECONDS = DATETIME(storage_format="%(year)04d-%(month)02d-%(day)02d %(hour)02d:%(minute)02d:%(second)02d",
                   regexp=r"(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)")
meta = MetaData()

teams = Table("teams", meta,
              Column("team_id", Integer, primary_key=True),
              Column("name", String, nullable=False, unique=True),
              Column("country", String))

matches = Table("matches", meta,
                Column("match_id", String, primary_key=True),
                Column("league", String, nullable=False),
                Column("season", Integer, nullable=False),
                Column("match_date", Date, nullable=False),
                Column("kickoff_utc", SECONDS),                        # NULL where the source has no time (< 2019/20)
                Column("home_team_id", ForeignKey("teams.team_id"), nullable=False),
                Column("away_team_id", ForeignKey("teams.team_id"), nullable=False),
                Column("fthg", Integer), Column("ftag", Integer),
                *[Column(name, Float) for name in STATS.values()],
                Column("referee", String),
                Column("pre_closing_captured_utc", SECONDS),
                Column("status", String, nullable=False, default="played"),
                Column("source", String, nullable=False),
                Column("ingested_at_utc", SECONDS, nullable=False),
                Index("ix_matches_league_season", "league", "season"),
                Index("ix_matches_date", "match_date"))

# primary key doubles as the lookup index (bookmaker, market, snapshot first); no rowid keeps the table ~half size
odds_snapshots = Table("odds_snapshots", meta,
                       Column("bookmaker", String, primary_key=True),
                       Column("market", String, primary_key=True),        # 1x2, ou_2.5
                       Column("snapshot", String, primary_key=True),      # pre_closing, closing
                       Column("match_id", ForeignKey("matches.match_id"), primary_key=True),
                       Column("selection", String, primary_key=True),     # home/draw/away, over/under
                       Column("odds", Float, nullable=False),
                       Column("captured_at_utc", SECONDS),
                       Column("captured_at_estimated", Boolean, nullable=False),
                       Column("source", String, nullable=False),
                       sqlite_with_rowid=False)


def engine(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    e = create_engine(f"sqlite:///{path}")
    meta.create_all(e)
    return e


def _none(v):
    """pandas missing values → None for the database."""
    return None if v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else v


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        out.append({k: (v.to_pydatetime() if isinstance(v, pd.Timestamp) else _none(v)) for k, v in r.items()})
    return out


def team_ids(conn, names, country: str | None = None) -> dict[str, int]:
    """Team ids by name, inserting new names. A name already stored for another country raises instead of merging
    two different clubs."""
    names = sorted(set(names))
    conn.execute(sqlite_insert(teams).on_conflict_do_nothing(index_elements=["name"]),
                 [{"name": n, "country": country} for n in names])
    rows = conn.execute(select(teams.c.name, teams.c.team_id, teams.c.country).where(teams.c.name.in_(names))).all()
    clash = [f"{n} ({c})" for n, _, c in rows if country and c and c != country]
    if clash:
        raise ValueError(f"team names already used in another country: {clash} — make them unique in data.team_name")
    return {n: i for n, i, _ in rows}


def ingest(conn, m: pd.DataFrame, odds: pd.DataFrame, source: str) -> tuple[int, int]:
    """Upsert one league-season of matches and replace its odds. Returns (matches, odds rows) written."""
    league = m.league.iloc[0]
    ids = team_ids(conn, pd.concat([m.home, m.away]), LEAGUES.get(league, (None,))[0])
    rows = m.assign(home_team_id=m.home.map(ids), away_team_id=m.away.map(ids), match_date=m.match_date.dt.date,
                    status="played", source=source,
                    ingested_at_utc=datetime.now(timezone.utc).replace(tzinfo=None)).drop(columns=["home", "away"])
    stmt = sqlite_insert(matches)
    stmt = stmt.on_conflict_do_update(index_elements=["match_id"],
                                      set_={c: stmt.excluded[c] for c in rows.columns if c != "match_id"})
    conn.execute(stmt, _records(rows))
    conn.execute(delete(odds_snapshots).where(odds_snapshots.c.match_id.in_(list(m.match_id)),
                                              odds_snapshots.c.source == source))
    if len(odds):
        conn.execute(insert(odds_snapshots), _records(odds))
    return len(rows), len(odds)


def load_matches(conn, leagues=None, seasons=None) -> pd.DataFrame:
    """Matches with team names, sorted by date, plus the columns the models use: div (league), date, hg/ag,
    result (0 home, 1 draw, 2 away) and decision (date of the pre-closing odds collection — only results before
    it may be used for that match)."""
    h, a = teams.alias("h"), teams.alias("a")
    q = (select(matches, h.c.name.label("home"), a.c.name.label("away"))
         .join(h, matches.c.home_team_id == h.c.team_id).join(a, matches.c.away_team_id == a.c.team_id))
    if leagues:
        q = q.where(matches.c.league.in_(list(leagues)))
    if seasons:
        q = q.where(matches.c.season.in_(list(seasons)))
    df = pd.DataFrame(conn.execute(q).mappings().all())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df.match_date)
    df["div"], df["hg"], df["ag"] = df.league, df.fthg, df.ftag
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], [0, 1], 2)
    # Recompute schedule for legacy databases too: older imports backdated holiday quotes.
    cap = capture_times_utc(df.date)
    kick = pd.to_datetime(df.kickoff_utc)
    valid = kick.notna() & (cap < kick)
    df["decision"] = cap.dt.normalize().where(valid, df.date)
    return df.sort_values(["date", "league", "home"], kind="stable").reset_index(drop=True)


def odds_matrix(conn, match_ids, bookmaker: str, market: str, snapshot: str, selections, *, allow_estimated: bool = False,
                known_by=None) -> np.ndarray:
    """(n, len(selections)) decimal odds aligned to match_ids; NaN where that bookmaker has no complete price."""
    q = (select(odds_snapshots.c.match_id, odds_snapshots.c.selection, odds_snapshots.c.odds)
         .where(odds_snapshots.c.bookmaker == bookmaker, odds_snapshots.c.market == market,
                odds_snapshots.c.snapshot == snapshot))
    if not allow_estimated:
        q = q.where(odds_snapshots.c.captured_at_estimated.is_(False),
                    odds_snapshots.c.captured_at_utc.is_not(None))
    if not allow_estimated and snapshot == "pre_closing":
        q = q.join(matches, matches.c.match_id == odds_snapshots.c.match_id).where(
            matches.c.kickoff_utc.is_not(None), odds_snapshots.c.captured_at_utc < matches.c.kickoff_utc)
    if known_by is not None:
        q = q.where(odds_snapshots.c.captured_at_utc <= known_by,
                    odds_snapshots.c.captured_at_estimated.is_(False))
    o = pd.DataFrame(conn.execute(q).all(), columns=["match_id", "selection", "odds"])
    wide = o.pivot(index="match_id", columns="selection", values="odds") if len(o) else pd.DataFrame()
    return wide.reindex(index=list(match_ids), columns=list(selections)).to_numpy(float)
