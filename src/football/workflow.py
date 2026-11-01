"""Forward paper workflow using the existing Dixon–Coles model; no new model.

All I/O is injectable. Model fitting uses completed matches strictly before today's
UTC midnight, avoiding assumptions about same-day result availability.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from src.football import db, markets
from src.football.data import LEAGUES
from src.football.models import DixonColes
from src.football.optimiser import Pick, optimise
from src.football.uncertainty import lower_bounds


def instant(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if value.utcoffset() != timedelta(0):
        raise ValueError('aware UTC timestamp required')
    return value


def load_history(path, as_of):
    """Read existing history without creating/migrating/updating the historical DB."""
    if not Path(path).is_file():
        raise ValueError('Historical database missing: run football_ingest.py first')
    engine = create_engine(f'sqlite:///file:{Path(path).resolve()}?mode=ro&uri=true')
    try:
        with engine.connect() as conn:
            history = db.load_matches(conn)
    finally:
        engine.dispose()
    cutoff = pd.Timestamp(instant(as_of)).tz_localize(None).normalize()
    if history.empty:
        raise ValueError('Historical database has no matches')
    history = history[(history.date < cutoff) & history.hg.notna() & history.ag.notna()].copy()
    return history.reset_index(drop=True), cutoff.to_pydatetime().replace(tzinfo=timezone.utc)


def merge_forward_history(history, fixtures, as_of):
    """Use newly received regulation results without rewriting historical tables."""
    as_of = instant(as_of)
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
    rows = []
    for f in fixtures:
        if f['status'] != 'finished' or instant(f['updated_at']) >= as_of:
            continue
        date = pd.Timestamp(instant(f['kickoff_utc'])).tz_convert('Europe/London').tz_localize(None).normalize()
        if date >= cutoff or f['hg'] is None or f['ag'] is None:
            continue
        rows.append(dict(date=date, div=f['league'], season=int(f['season']), home=f['home'], away=f['away'],
                         hg=f['hg'], ag=f['ag'], result=0 if f['hg'] > f['ag'] else 1 if f['hg'] == f['ag'] else 2,
                         match_id='api-football:' + f['provider_id'], decision=date))
    if not rows:
        return history
    return (pd.concat([history, pd.DataFrame(rows)], ignore_index=True)
            .drop_duplicates(['date', 'div', 'home', 'away'], keep='first')
            .sort_values(['date', 'div', 'home'], kind='stable').reset_index(drop=True))


def predict_fixtures(history, fixtures, as_of, diagnostics=None):
    """Same existing tuned DC settings; country-specific fits, all markets per score grid.

    Existing evaluation chose 365d half-life for 1X2 and 730d for totals/BTTS.
    Provenance hashes exact training rows. This does not retrain LightGBM or tune.
    """
    as_of = instant(as_of)
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
    hist = history[(history.date < cutoff) & history.hg.notna() & history.ag.notna()].copy()
    if hist.empty:
        return {}, 'dc-timing-v2:no-history'
    fingerprint = hashlib.sha256(hist[['date', 'div', 'home', 'away', 'hg', 'ag']].to_csv(index=False).encode()).hexdigest()
    version = 'dc-timing-v2:1x2-365:goals-730:window3650:' + fingerprint
    out = {}
    countries = hist['div'].map(lambda d: LEAGUES[d][0])
    for country in sorted({LEAGUES[f['league']][0] for f in fixtures}):
        train = hist[countries == country]
        if len(train) < 100:
            continue
        group = [f for f in fixtures if LEAGUES[f['league']][0] == country]
        known = set(train.home) | set(train.away)
        group = [f for f in group if f['home'] in known and f['away'] in known]
        if not group:
            continue
        frame = pd.DataFrame(group).rename(columns={'league': 'div'})
        one = DixonColes(half_life=365, window_days=3650).fit(train, cutoff).markets(frame)
        goal_model = DixonColes(half_life=730, window_days=3650).fit(train, cutoff)
        goals = goal_model.markets(frame)
        if diagnostics is not None:
            home_goals,away_goals=goal_model.expected_goals(frame)
            for f,h,a in zip(group,home_goals,away_goals):
                diagnostics[f['provider_id']]=dict(expected_home_goals=float(h),expected_away_goals=float(a),expected_total_goals=float(h+a))
        for i, fixture in enumerate(group):
            p = {key: float(value) for key, value in goals.iloc[i].items()}
            for market in ('1x2', 'double_chance'):
                for key in markets.columns(market):
                    p[key] = float(one.iloc[i][key])
            out[fixture['provider_id']] = p
    return out, version


def candidates(store, now):
    """Latest forecast and latest quote only; stale older favourable quotes never win."""
    now = instant(now)
    fixtures = {f['provider_id']: f for f in store.fixtures()}
    forecasts = {}
    for row in store.forecasts():
        if instant(row['as_of']) <= now and (row['provider_id'] not in forecasts or
                instant(row['as_of']) > instant(forecasts[row['provider_id']]['as_of'])):
            forecasts[row['provider_id']] = row
    latest = {}
    for row in store.odds():
        key = tuple(row[k] for k in ('provider_id', 'bookmaker', 'market', 'selection'))
        if instant(row['captured_at_utc']) <= now and (key not in latest or
                instant(row['captured_at_utc']) >= instant(latest[key]['captured_at_utc'])):
            latest[key] = row
    picks, refs = [], {}
    for quote in latest.values():
        fixture = fixtures.get(quote['provider_id'])
        forecast = forecasts.get(quote['provider_id'])
        key = quote['market'] + ':' + quote['selection']
        if not fixture or not forecast or fixture['status'] != 'scheduled' or key not in forecast['probabilities']:
            continue
        if instant(forecast['kickoff_utc']) != instant(fixture['kickoff_utc']):
            continue  # revised schedule needs a new forecast
        if now - instant(forecast['as_of']) > timedelta(hours=24):
            continue
        if not timedelta(0) <= now - instant(quote['source_updated_at']) <= timedelta(minutes=60):
            continue
        pick = Pick(snapshot_id=quote['id'], match_id=fixture['provider_id'], bookmaker=quote['bookmaker'],
                    market=quote['market'], selection=quote['selection'], odds=quote['odds'],
                    probability=forecast['probabilities'][key], p_lower=forecast['lower'][key],
                    captured_at_utc=instant(quote['captured_at_utc']), kickoff_utc=instant(fixture['kickoff_utc']),
                    home=fixture['home'], away=fixture['away'], league=fixture['league'])
        picks.append(pick)
        refs[pick.snapshot_id] = forecast['id']
    return picks, refs


def run_cycle(store, provider, history_path, *, clock=None, place_paper=True, stake=1,
              initial_balance=1000, max_odds_fixtures=25, predictor=predict_fixtures, rank_by="ev"):
    clock = clock or (lambda: datetime.now(timezone.utc))
    start = instant(clock())
    messages = []
    store.initialise_bankroll(start, initial_balance)
    store.settle(start)  # settle persisted results even during a provider outage
    # Refresh every unresolved recorded fixture, including postponements outside the window.
    unresolved = [f['provider_id'] for f in store.fixtures() if f['status'] not in ('finished', 'cancelled', 'abandoned')]
    if unresolved:
        store.upsert_fixtures(provider.results(unresolved), instant(clock()))
        store.settle(instant(clock()))
    store.upsert_fixtures(provider.fixtures(start.date(), (start + timedelta(days=5)).date()), instant(clock()))
    store.settle(instant(clock()))
    upcoming = [f for f in store.fixtures() if f['status'] == 'scheduled'
                and start < instant(f['kickoff_utc']) <= start + timedelta(days=5)]
    upcoming.sort(key=lambda f: (f['kickoff_utc'], f['provider_id']))
    quote_limit=len(upcoming) if getattr(provider,"odds_preloaded",False) else max_odds_fixtures
    for fixture in upcoming[:quote_limit]:
        try:
            store.add_odds(provider.odds(fixture['provider_id']), instant(clock()))
        except (ValueError, RuntimeError) as exc:
            messages.append(f"Odds unavailable for {fixture['provider_id']}: {type(exc).__name__}")
    if len(upcoming) > quote_limit:
        messages.append(f'Odds collection capped at {max_odds_fixtures} fixtures per cycle')
    fit_time = instant(clock())
    history, cutoff = load_history(history_path, fit_time)
    history = merge_forward_history(history, store.fixtures(), fit_time)
    probabilities, version = predictor(history, upcoming, fit_time)
    evidence = store.forecasts()
    fixtures = store.fixtures()
    frozen = []
    for fixture in upcoming:
        p = probabilities.get(fixture['provider_id'])
        as_of = instant(clock())
        if p is None or as_of >= instant(fixture['kickoff_utc']) or cutoff >= as_of:
            continue
        lower, _ = lower_bounds(p, evidence, fixtures, as_of, league=fixture['league'])
        frozen.append(store.add_forecast(fixture['provider_id'], p, lower, as_of, cutoff,
                                        version + ':wilson95-league-bin10-min100-v2' + (':head='+provider.prediction_sources[fixture['provider_id']] if hasattr(provider,'prediction_sources') and fixture['provider_id'] in provider.prediction_sources else '')))
    now = instant(clock())
    picks, refs = candidates(store, now)
    tickets = optimise(picks, now=now, min_robust_ev=0.02, rank_by=rank_by)
    placed = []
    if place_paper:
        # Prevent repeated polling from multiplying exposure to the same match.
        used = {leg['provider_id'] for t in store.tickets() for leg in t['legs']}
        for ticket in tickets:
            if used.intersection(p.match_id for p in ticket.legs):
                continue
            payload = {'legs': [dict(forecast_id=refs[p.snapshot_id], odds_id=p.snapshot_id,
                                    provider_id=p.match_id, market=p.market, selection=p.selection, odds=p.odds)
                                for p in ticket.legs]}
            try:
                placed.append(store.place_ticket(payload, instant(clock()), stake=stake, initial_balance=initial_balance))
                used.update(p.match_id for p in ticket.legs)
            except ValueError as exc:
                messages.append(f'Paper placement skipped: {exc}')
    return dict(fixtures=len(upcoming), forecasts=len(frozen), candidates=len(picks),
                tickets=len(tickets), placed=len(placed), decision='TICKETS' if tickets else 'NO BET',
                messages=messages, bankroll=store.bankroll())
