"""Run a single paper cycle, or a persistent scheduled worker with --watch."""
import argparse
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from sqlalchemy import create_engine

from src.football import db
from src.football.forward_store import ForwardStore
from src.football.providers import APIFootball, ProviderError
from src.football.public_provider import PublicCSV
from src.football.scoreboard_provider import ScoreboardCSV
from src.football.odds_api import OddsAPI, OddsAPIProvider
from src.football.workflow import run_cycle


class BudgetSession:
    """Persist an account-wide local daily request cap across worker restarts.

    Other applications using the same provider account are outside this local counter.
    Each attempted HTTP request consumes one local unit, including failures.
    """
    def __init__(self, path, limit=100, session=None, clock=None):
        self.path, self.limit = str(path), limit
        self.session = session or requests.Session()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        with sqlite3.connect(self.path) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS football_api_usage(day TEXT PRIMARY KEY, requests INTEGER NOT NULL)')

    def get(self, *args, **kwargs):
        day = self.clock().astimezone(timezone.utc).date().isoformat()
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('INSERT OR IGNORE INTO football_api_usage VALUES (?,0)', (day,))
            count = conn.execute('SELECT requests FROM football_api_usage WHERE day=?', (day,)).fetchone()[0]
            if count >= self.limit:
                raise ProviderError('Local daily API request budget exhausted; resumes next UTC day')
            conn.execute('UPDATE football_api_usage SET requests=requests+1 WHERE day=?', (day,))
        return self.session.get(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=db.DB_PATH, help='Read-only historical database')
    parser.add_argument('--paper-db', type=Path, default=db.DB_PATH.parent/'paper_forward.db', help='Separate additive paper records; historical database is preserved')
    parser.add_argument('--provider', choices=['auto','odds-api','public-csv','scoreboard-csv','api-football'], default='auto', help='auto combines public daily scoreboards with publisher prices; api-football requires current-season entitlement')
    parser.add_argument('--odds-model',type=Path,help='Optional locally trained experimental LightGBM odds head; use a separate paper DB')
    parser.add_argument('--rank-by',choices=['probability','ev'],default='probability')
    parser.add_argument('--odds-extra-events',type=int,default=3,help='Maximum event-specific totals/BTTS/double-chance calls')
    parser.add_argument('--odds-cycle-credits',type=int,default=30)
    parser.add_argument('--team-map', type=Path, help='JSON provider team name -> historical canonical name')
    parser.add_argument('--watch', action='store_true', help='Run scheduled cycles until stopped')
    parser.add_argument('--interval', type=int, default=28800, help='Seconds between cycles (default 8 hours)')
    parser.add_argument('--daily-request-limit', type=int, default=100)
    parser.add_argument('--max-odds-fixtures', type=int, default=20)
    parser.add_argument('--stake', type=float, default=10)
    parser.add_argument('--initial-balance', type=float, default=1000)
    parser.add_argument('--forecast-only', action='store_true', help='Collect and forecast without paper stakes')
    args = parser.parse_args()
    if args.interval < 60 or args.daily_request_limit < 1 or args.max_odds_fixtures < 0:
        parser.error('interval >=60, daily request limit >=1, and odds fixture cap >=0 required')
    key = os.getenv('API_FOOTBALL_KEY', '')  # config.settings has loaded .env through db import
    if not key and args.provider=='api-football':
        parser.error('Set API_FOOTBALL_KEY in .env; no bookmaker credentials are used')
    if not args.db.is_file():
        parser.error('Historical database missing; run football_ingest.py first')
    engine = create_engine(f'sqlite:///file:{args.db.resolve()}?mode=ro&uri=true')
    with engine.connect() as conn:
        names = {r[0] for r in conn.exec_driver_sql('SELECT name FROM teams')}
    engine.dispose()
    mapping = {n: n for n in names}
    if args.team_map:
        mapping.update(json.loads(args.team_map.read_text()))
    store = ForwardStore(args.paper_db)
    odds_key=os.getenv('ODDS_API_KEY','')
    use_odds=args.provider=='odds-api' or (args.provider=='auto' and bool(odds_key))
    if use_odds and not odds_key:parser.error('Set ODDS_API_KEY to use The Odds API')
    if use_odds:
        schedule=ScoreboardCSV(canonical_teams=names,known_fixtures=store.fixtures(),snapshot_dir=db.DB_PATH.parent/'forward_sources')
        api=OddsAPI(odds_key,canonical_teams=names,snapshot_dir=db.DB_PATH.parent/'odds_api'/'snapshots',budget_db=db.DB_PATH.parent/'odds_api'/'usage.db',cycle_budget=args.odds_cycle_credits)
        provider=OddsAPIProvider(api,schedule=schedule,extra_events=args.odds_extra_events)
    elif args.provider=='api-football':
        provider = APIFootball(key, team_mapping=mapping, canonical_teams=names,
                              session=BudgetSession(args.paper_db, args.daily_request_limit))
    else:
        cls=PublicCSV if args.provider=='public-csv' else ScoreboardCSV
        provider = cls(canonical_teams=names, known_fixtures=store.fixtures(),
                       snapshot_dir=db.DB_PATH.parent/'forward_sources')
    prediction_fn=None
    if args.odds_model:
        from src.football.odds_training import predictor
        prediction_fn=predictor(args.odds_model,provider)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        while True:
            try:
                result = run_cycle(store, provider, args.db, place_paper=not args.forecast_only,
                                   stake=args.stake, initial_balance=args.initial_balance,
                                   max_odds_fixtures=args.max_odds_fixtures,rank_by=args.rank_by,
                                   **({'predictor':prediction_fn} if prediction_fn else {}))
                result['provider'] = 'odds-api' if use_odds else 'api-football' if args.provider=='api-football' else 'public-csv' if args.provider=='public-csv' else 'scoreboard-csv'
                result['fixture_coverage'] = 'public five-day schedules plus The Odds API UK decimal odds on supported/matched events' if use_odds else 'provider current-season entitlement' if args.provider=='api-football' else 'published fixture batch only; not guaranteed complete five-day coverage' if args.provider=='public-csv' else 'five-day daily public scoreboards across 10 configured leagues; explicit team aliases; upstream completeness not guaranteed'
                result['rank_by']=args.rank_by
                result['training_mode']='experimental historical odds LightGBM 1X2/DC plus Dixon-Coles goals' if args.odds_model else 'existing Dixon-Coles'
                if hasattr(provider,'latest_metadata'):
                    result['source_metadata'] = provider.latest_metadata
                Path(args.paper_db).with_suffix('.status.json').write_text(json.dumps(result,indent=2))
                if Path(args.paper_db).name=='paper_forward.db':
                    (Path(args.paper_db).parent/'latest_forward_status.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(result, sort_keys=True), flush=True)
            except (ProviderError, ValueError, sqlite3.Error) as exc:
                logging.error('Cycle failed (%s): %s', type(exc).__name__, str(exc))
                if not args.watch:
                    raise SystemExit(1) from None
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__ == '__main__':
    main()
