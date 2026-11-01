"""Recorded-provider end-to-end paper cycles; no external service or real stakes."""
from datetime import datetime, timedelta, timezone
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.football import db, data
from src.football.forward_store import ForwardStore
from src.football.uncertainty import lower_bounds
from src.football.workflow import candidates, load_history, predict_fixtures, run_cycle
from football_forward import BudgetSession
from src.football.providers import ProviderError

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
PROBS = {'1x2:home': .9, '1x2:draw': .06, '1x2:away': .04}


def fixture(identifier, kickoff=NOW+timedelta(days=1), **kwargs):
    return dict(provider_id=str(identifier), league='E0', season=2026, home='A', away='B',
                kickoff_utc=kickoff, status='scheduled', **kwargs)


def historical(path):
    """Minimal historical schema/data for workflow reads; no production file."""
    eng = db.engine(path)
    with eng.begin() as conn:
        ids = db.team_ids(conn, ['A', 'B'], 'England')
        conn.execute(db.matches.insert(), [dict(match_id='old', league='E0', season=2025,
            match_date=NOW.date()-timedelta(days=2), kickoff_utc=NOW.replace(tzinfo=None)-timedelta(days=2),
            home_team_id=ids['A'], away_team_id=ids['B'], fthg=1, ftag=0,
            status='played', source='recorded', ingested_at_utc=NOW.replace(tzinfo=None))])
    eng.dispose()


class RecordedProvider:
    def __init__(self):
        self.rows = [fixture('1'), fixture('2')]

    def fixtures(self, start, end):
        return self.rows

    def results(self, ids):
        return [f for f in self.rows if f['provider_id'] in ids]

    def odds(self, identifier):
        return [dict(provider_id=identifier, bookmaker='Recorded Book', market='1x2', selection='home',
                     odds=2.2 if identifier == '1' else 2.25,
                     captured_at_utc=NOW, source_updated_at=NOW-timedelta(minutes=1))]


def predictor(history, fixtures, as_of):
    assert (history.date < pd.Timestamp(as_of).tz_localize(None).normalize()).all()
    return {f['provider_id']: PROBS for f in fixtures}, 'recorded-existing-model-test'


def seed_evidence(store, count=100):
    start = NOW-timedelta(days=20)
    rows = [fixture(f'e{i}', kickoff=start+timedelta(hours=3)) for i in range(count)]
    store.upsert_fixtures(rows, start)
    for row in rows:
        store.add_forecast(row['provider_id'], PROBS, {k: 0 for k in PROBS}, start,
                           start-timedelta(days=1), 'recorded-evidence')
    store.upsert_fixtures([dict(r, status='finished', hg=1, ag=0) for r in rows], start+timedelta(days=1))


def test_cold_start_freezes_predictions_and_returns_no_bet(tmp_path):
    path = tmp_path/'football.db'
    historical(path)
    store = ForwardStore(path)
    result = run_cycle(store, RecordedProvider(), path, clock=lambda: NOW, predictor=predictor)
    assert result['decision'] == 'NO BET'
    assert result['forecasts'] == 2 and result['candidates'] == 2
    assert store.bankroll()['cash'] == 1000 and not store.tickets()
    assert all(set(f['lower'].values()) == {0} for f in store.forecasts())
    # Same timestamp/content repeats idempotently; no duplicate frozen revisions.
    run_cycle(store, RecordedProvider(), path, clock=lambda: NOW, predictor=predictor)
    assert len(store.forecasts()) == 2
    assert store.conn.execute('SELECT count(*) FROM matches').fetchone()[0] == 1
    store.close()


def test_full_paper_cycle_placement_restart_and_automatic_settlement(tmp_path):
    path = tmp_path/'football.db'
    historical(path)
    store = ForwardStore(path)
    seed_evidence(store)
    provider = RecordedProvider()
    first = run_cycle(store, provider, path, clock=lambda: NOW, predictor=predictor)
    assert first['placed'] == 1 and first['tickets'] == 1
    assert store.tickets()[0]['combined_odds'] == pytest.approx(4.95)
    assert store.bankroll()['cash'] == 999
    run_cycle(store, provider, path, clock=lambda: NOW, predictor=predictor)
    assert len(store.tickets()) == 1
    store.close()
    store = ForwardStore(path)
    provider.rows = [dict(f, status='finished', hg=2, ag=0) for f in provider.rows]
    settled = run_cycle(store, provider, path, clock=lambda: NOW+timedelta(days=2), predictor=predictor)
    assert settled['bankroll']['cash'] == pytest.approx(1003.95)
    assert store.tickets()[0]['status'] == 'won'
    assert len(store.settle(NOW+timedelta(days=3))) == 0
    store.close()


def test_uncertainty_rejects_future_results_and_duplicate_forecasts(tmp_path):
    store = ForwardStore(tmp_path/'f.db')
    seed_evidence(store, 99)
    p, count = lower_bounds(PROBS, store.forecasts()*4, store.fixtures(), NOW)
    assert count['1x2:home'] == 99 and p['1x2:home'] == 0
    p, count = lower_bounds(PROBS, store.forecasts(), store.fixtures(), NOW-timedelta(days=21))
    assert count['1x2:home'] == 0 and p['1x2:home'] == 0
    # Even a result labelled finished cannot enter before its recorded receipt.
    future_received = [dict(f, updated_at=(NOW+timedelta(days=1)).isoformat()) for f in store.fixtures()]
    assert lower_bounds(PROBS, store.forecasts(), future_received, NOW)[1]['1x2:home'] == 0
    store.close()


def test_candidate_uses_latest_quote_and_requires_source_freshness(tmp_path):
    store = ForwardStore(tmp_path/'f.db')
    store.upsert_fixtures([fixture('1')], NOW)
    store.add_forecast('1', PROBS, PROBS, NOW-timedelta(minutes=5), NOW-timedelta(days=1), 'test')
    base = dict(provider_id='1', bookmaker='Book', market='1x2', selection='home', odds=2.5,
                captured_at_utc=NOW-timedelta(minutes=2), source_updated_at=NOW-timedelta(minutes=2))
    store.add_odds([base], NOW)
    store.add_odds([dict(base, odds=1.2, captured_at_utc=NOW)], NOW)
    picks, _ = candidates(store, NOW)
    assert len(picks) == 1 and picks[0].odds == 1.2
    store.add_odds([dict(base, odds=3, captured_at_utc=NOW, source_updated_at=NOW-timedelta(hours=2))], NOW)
    assert candidates(store, NOW)[0] == []
    store.close()


def test_history_read_is_nonmutating_and_excludes_future(tmp_path):
    path = tmp_path/'f.db'
    historical(path)
    before = path.read_bytes()
    history, cutoff = load_history(path, NOW)
    assert len(history) == 1 and cutoff.hour == 0 and cutoff < NOW
    assert path.read_bytes() == before
    future = history.copy()
    future['date'] = pd.Timestamp('2026-10-28')
    combined = pd.concat([history, future], ignore_index=True)
    # Below minimum history -> model must not invent predictions.
    predictions, _ = predict_fixtures(combined, [fixture('1')], NOW)
    assert predictions == {}


def test_provider_daily_budget_survives_restart(tmp_path):
    class Session:
        def get(self, *args, **kwargs):
            return 'response'
    path = tmp_path/'budget.db'
    first = BudgetSession(path, limit=1, session=Session(), clock=lambda: NOW)
    assert first.get('recorded') == 'response'
    second = BudgetSession(path, limit=1, session=Session(), clock=lambda: NOW)
    with pytest.raises(ProviderError, match='budget'):
        second.get('recorded')
    tomorrow = BudgetSession(path, limit=1, session=Session(), clock=lambda: NOW+timedelta(days=1))
    assert tomorrow.get('recorded') == 'response'


def test_calibration_chooses_earliest_revision_before_binning(tmp_path):
    store = ForwardStore(tmp_path/'f.db')
    seed_evidence(store, 1)
    original = store.forecasts()[0]
    later = dict(original, id='revision', as_of=(NOW-timedelta(days=19, hours=22)).isoformat(),
                 probabilities={'1x2:home': .65})
    original = dict(original, probabilities={'1x2:home': .55})
    result, counts = lower_bounds({'1x2:home': .65}, [original, later], store.fixtures(), NOW, minimum_samples=1)
    assert counts['1x2:home'] == 0 and result['1x2:home'] == 0
    store.close()


def test_stored_results_settle_even_if_fixture_discovery_fails(tmp_path):
    path = tmp_path/'f.db'
    historical(path)
    store = ForwardStore(path)
    seed_evidence(store)
    provider = RecordedProvider()
    run_cycle(store, provider, path, clock=lambda: NOW, predictor=predictor)
    later = NOW+timedelta(days=2)
    store.upsert_fixtures([dict(f, status='finished', hg=2, ag=0) for f in provider.rows], later)
    def unavailable(*args):
        raise ProviderError('outage')
    provider.fixtures = unavailable
    with pytest.raises(ProviderError):
        run_cycle(store, provider, path, clock=lambda: later, predictor=predictor)
    assert store.tickets()[0]['status'] == 'won'
    assert store.bankroll()['cash'] == pytest.approx(1003.95)
    store.close()


def test_actual_existing_model_predictions_do_not_change_with_future_scores():
    # Exercise the real model, not just the injected orchestration test double.
    dates = pd.date_range('2026-01-01', periods=120, freq='2D')
    hist = pd.DataFrame(dict(date=dates, div='E0', home=['A', 'B']*60,
                             away=['B', 'A']*60, hg=[2, 1]*60, ag=[0, 1]*60))
    future = pd.DataFrame(dict(date=[pd.Timestamp('2026-10-28')], div=['E0'],
                               home=['A'], away=['B'], hg=[99], ag=[0]))
    before, version = predict_fixtures(pd.concat([hist, future]), [fixture('1')], NOW)
    future.loc[0, ['hg', 'ag']] = [0, 99]
    after, after_version = predict_fixtures(pd.concat([hist, future]), [fixture('1')], NOW)
    assert before == after and version == after_version
    for market in ('1x2', 'ou_1.5', 'ou_2.5', 'ou_3.5', 'btts'):
        values = [v for k, v in before['1'].items() if k.startswith(market+':')]
        assert sum(values) == pytest.approx(1)
        assert all(0 <= x <= 1 for x in values)


def test_new_forward_results_feed_training_only_after_receipt_and_not_twice(tmp_path):
    from src.football.workflow import merge_forward_history
    path = tmp_path/'f.db'
    historical(path)
    history, _ = load_history(path, NOW)
    old = fixture('101', kickoff=NOW-timedelta(days=3), hg=3, ag=0)
    old.update(status='finished', updated_at=(NOW-timedelta(days=2)).isoformat())
    future_received = dict(old, provider_id='102', home='B', away='A', updated_at=(NOW+timedelta(days=1)).isoformat())
    same_day = dict(old, provider_id='103', kickoff_utc=NOW-timedelta(hours=3))
    merged = merge_forward_history(history, [old, future_received, same_day], NOW)
    assert len(merged) == 2 and 'api-football:101' in set(merged.match_id)
    assert 'api-football:102' not in set(merged.match_id)
    assert len(merge_forward_history(merged, [old], NOW)) == 2


def test_forward_calibration_does_not_borrow_other_leagues(tmp_path):
    store=ForwardStore(tmp_path/'f.db');seed_evidence(store,100)
    _,counts=lower_bounds(PROBS,store.forecasts(),store.fixtures(),NOW,league='E0')
    assert counts['1x2:home']==100
    bounds,counts=lower_bounds(PROBS,store.forecasts(),store.fixtures(),NOW,league='SP1')
    assert counts['1x2:home']==0 and bounds['1x2:home']==0
    store.close()
