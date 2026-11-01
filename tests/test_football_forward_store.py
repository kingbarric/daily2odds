from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from src.football.forward_store import ForwardStore, utc

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
KO = NOW + timedelta(hours=2)


def fixture(i, **changes):
    return dict(provider_id=str(i), league='E0', season='2026', kickoff_utc=KO,
                home='Home', away='Away', status='scheduled', **changes)


def setup_ticket(store, ids=('1', '2'), **kwargs):
    store.upsert_fixtures([fixture(i) for i in ids], NOW)
    legs = []
    for i, price in zip(ids, (2.0, 2.5)):
        f = store.add_forecast(i, {'1x2:home': .6}, {'1x2:home': .5}, NOW,
                               NOW-timedelta(days=1), 'v1')
        o = store.add_odds([dict(provider_id=i, bookmaker='Book', market='1x2',
                                selection='home', odds=price, source_updated_at=NOW)], NOW)[0]
        legs.append(dict(forecast_id=f['id'], odds_id=o['id'], provider_id=i,
                         market='1x2', selection='home', odds=price))
    return dict(legs=legs, **kwargs)


def finish(store, i, hg=2, ag=0):
    row = fixture(i)
    row.update(status='finished', hg=hg, ag=ag)
    store.upsert_fixtures([row], KO+timedelta(hours=2))


def test_historical_coexistence_and_persistence(tmp_path):
    path = tmp_path/'football.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE matches (id INTEGER, payload TEXT)')
        conn.execute("INSERT INTO matches VALUES (1,'historical')")
    s = ForwardStore(path)
    t = s.place_ticket(setup_ticket(s), NOW, stake=10, initial_balance=100)
    assert s.bankroll()['cash'] == 90
    assert s.bankroll()['reserved'] == 10
    assert s.bankroll()['equity'] == 100
    s.close()
    s = ForwardStore(path)
    assert s.tickets()[0]['id'] == t['id']
    assert tuple(s.conn.execute('SELECT * FROM matches').fetchone()) == (1, 'historical')
    assert s.bankroll()['cash'] == 90


def test_idempotency_and_immutable_sql(tmp_path):
    s = ForwardStore(tmp_path/'db')
    ticket = setup_ticket(s)
    a = s.place_ticket(ticket, NOW)
    assert s.place_ticket(ticket, KO+timedelta(days=1)) == a
    f = s.forecasts()[0]
    assert s.add_forecast('1', f['probabilities'], f['lower'], NOW, NOW-timedelta(days=1), 'v1') == f
    with pytest.raises(ValueError):
        s.add_forecast('1', {'1x2:home': .7}, f['lower'], NOW, NOW-timedelta(days=1), 'v1')
    o = s.odds()[0]
    assert s.add_odds([o], NOW)[0] == o
    for table in ('forecasts', 'live_odds', 'ledger'):
        for sql in (f'DELETE FROM {table}', f'UPDATE {table} SET id=id'):
            with pytest.raises(sqlite3.IntegrityError):
                s.conn.execute(sql)
    assert len(s._rows('ledger')) == 2
    assert s.forecasts()[0]['probabilities'] == {'1x2:home': .6}


@pytest.mark.parametrize('kind', ['postko', 'future', 'future_source', 'stale', 'mismatch', 'bookmaker', 'asof'])
def test_invalid_placement(tmp_path, kind):
    s = ForwardStore(tmp_path/'db')
    ticket = setup_ticket(s)
    when = NOW
    if kind == 'postko':
        when = KO
    elif kind == 'stale':
        when = NOW+timedelta(minutes=61)
    elif kind == 'mismatch':
        ticket['legs'][0]['odds'] = 2.01
    elif kind in ('future', 'future_source'):
        row = dict(provider_id='1', bookmaker='Book', market='1x2', selection='home', odds=2,
                   captured_at_utc=NOW+timedelta(minutes=1) if kind == 'future' else NOW,
                   source_updated_at=NOW+timedelta(minutes=1))
        with pytest.raises(ValueError):
            s.add_odds([row], NOW)
        return
    elif kind == 'bookmaker':
        o = s.add_odds([dict(provider_id='1', bookmaker='Other', market='1x2', selection='home', odds=2,
                            source_updated_at=NOW)], NOW)[0]
        ticket['legs'][0]['odds_id'] = o['id']
    elif kind == 'asof':
        when = NOW-timedelta(seconds=1)
    with pytest.raises(ValueError):
        s.place_ticket(ticket, when)
    assert s.tickets() == []
    assert s._rows('ledger') == []


def test_win_void_reduced_payout_and_corrections(tmp_path):
    s = ForwardStore(tmp_path/'db')
    s.place_ticket(setup_ticket(s), NOW, stake=10, initial_balance=100)
    finish(s, '1')
    row = fixture('2')
    row['status'] = 'abandoned'
    s.upsert_fixtures([row], KO+timedelta(hours=2))
    result = s.settle(KO+timedelta(hours=2))[0]
    assert result['status'] == 'won' and result['payout'] == 20
    assert s.bankroll()['cash'] == 110
    assert s.bankroll()['reserved'] == 0
    assert s.bankroll()['roi'] == 1
    finish(s, '1', 0, 3)
    assert s.settle(KO+timedelta(days=1)) == []
    assert s.tickets()[0] == result


def test_loss_settles_while_other_pending(tmp_path):
    s = ForwardStore(tmp_path/'db')
    s.place_ticket(setup_ticket(s), NOW, stake=10, initial_balance=100)
    finish(s, '1', 0, 2)
    assert s.settle(KO+timedelta(hours=2))[0]['status'] == 'lost'
    assert s.bankroll()['cash'] == 90
    assert s.bankroll()['max_drawdown'] == 10
    assert s.bankroll()['reserved'] == 0


@pytest.mark.parametrize('reschedule', [False, True])
def test_postponement_and_all_void(tmp_path, reschedule):
    s = ForwardStore(tmp_path/'db')
    s.place_ticket(setup_ticket(s), NOW, stake=10, initial_balance=100)
    rows = [fixture(i) for i in ('1', '2')]
    for row in rows:
        row['status'] = 'postponed'
        if reschedule:
            row['kickoff_utc'] = KO+timedelta(hours=49)
    s.upsert_fixtures(rows, KO)
    if not reschedule:
        assert s.settle(KO+timedelta(hours=47)) == []
    result = s.settle(KO+timedelta(hours=48))[0]
    assert result['status'] == 'void' and result['payout'] == 10
    assert s.bankroll()['equity'] == 100


def test_insufficient_funds_includes_reservations(tmp_path):
    path = tmp_path/'db'
    s = ForwardStore(path)
    s.place_ticket(setup_ticket(s), NOW, stake=60, initial_balance=100)
    other = ForwardStore(path)
    ticket = setup_ticket(other, ('3', '4'))
    with pytest.raises(ValueError, match='insufficient'):
        other.place_ticket(ticket, NOW, stake=60, initial_balance=10000)
    assert len(other.tickets()) == 1
    assert other.bankroll()['cash'] == 40
    assert other.bankroll()['initial_balance'] == 100


def test_snapshot_kickoff_and_temporal_validation(tmp_path):
    s = ForwardStore(tmp_path/'db')
    ticket = setup_ticket(s)
    rows = [fixture(i) for i in ('1', '2')]
    for row in rows:
        row['kickoff_utc'] = KO+timedelta(hours=12)
    s.upsert_fixtures(rows, NOW)
    assert utc(s.forecasts()[0]['kickoff_utc']) == KO
    with pytest.raises(ValueError):
        s.place_ticket(ticket, KO)
    with pytest.raises(ValueError):
        s.add_forecast('1', {'1x2:home': .6}, {'1x2:home': .5}, NOW, NOW, 'v2')
    with pytest.raises(ValueError):
        s.add_forecast('1', {'1x2:home': float('nan')}, {'1x2:home': .5}, NOW, NOW-timedelta(days=1), 'v2')
    assert utc(NOW.isoformat()) == NOW
    with pytest.raises(ValueError):
        utc(datetime(2026, 1, 1))


def test_full_win_and_ticket_conflicts(tmp_path):
    s = ForwardStore(tmp_path/'db')
    ticket = setup_ticket(s, id='test')
    s.place_ticket(ticket, NOW)
    with pytest.raises(ValueError, match='conflict'):
        s.place_ticket(ticket, NOW, stake=2)
    for i in ('1', '2'):
        finish(s, i)
    assert s.settle(KO+timedelta(hours=2))[0]['payout'] == 5
    assert s.bankroll()['cash'] == 1004


def test_batch_rollback_and_postkickoff_odds(tmp_path):
    s = ForwardStore(tmp_path/'db')
    s.upsert_fixtures([fixture('1')], NOW)
    good = dict(provider_id='1', bookmaker='Book', market='1x2', selection='home', odds=2,
                source_updated_at=NOW)
    with pytest.raises(ValueError):
        s.add_odds([good, dict(good, odds=float('inf'))], NOW)
    assert s.odds() == []
    with pytest.raises(ValueError):
        s.add_odds([good], KO)
    with pytest.raises(ValueError):
        s.upsert_fixtures([fixture('2'), dict(fixture('3'), hg=-1)], NOW)
    assert len(s.fixtures()) == 1


@pytest.mark.parametrize('alter', ['duplicate', 'one', 'six', 'range', 'probability', 'cancelled'])
def test_leg_constraints(tmp_path, alter):
    s = ForwardStore(tmp_path/'db')
    ticket = setup_ticket(s)
    if alter == 'duplicate':
        ticket['legs'][1] = dict(ticket['legs'][0])
    elif alter == 'one':
        ticket['legs'] = ticket['legs'][:1]
    elif alter == 'six':
        ticket['legs'] *= 3
    elif alter in ('range', 'probability'):
        leg = ticket['legs'][0]
        if alter == 'range':
            o = s.add_odds([dict(provider_id='1', bookmaker='Book', market='1x2', selection='home',
                                odds=3, source_updated_at=NOW)], NOW)[0]
            leg.update(odds_id=o['id'], odds=3)
        else:
            f = s.add_forecast('1', {'btts:yes': .6}, {'btts:yes': .5}, NOW,
                               NOW-timedelta(days=1), 'other')
            leg['forecast_id'] = f['id']
    else:
        row = fixture('1')
        row['status'] = 'cancelled'
        s.upsert_fixtures([row], NOW)
    with pytest.raises(ValueError):
        s.place_ticket(ticket, NOW)
    assert s.bankroll()['equity'] == 0


def test_configurable_postponement_and_source_age(tmp_path):
    s = ForwardStore(tmp_path/'db', postponement_hours=2)
    ticket = setup_ticket(s)
    s.place_ticket(ticket, NOW+timedelta(minutes=60))
    rows = [fixture(i) for i in ('1', '2')]
    for row in rows:
        row['status'] = 'postponed'
    s.upsert_fixtures(rows, KO)
    assert s.settle(KO+timedelta(hours=1)) == []
    assert s.settle(KO+timedelta(hours=2))[0]['status'] == 'void'


@pytest.mark.parametrize('market,selection,hg,ag,expected', [
    ('1x2', 'draw', 1, 1, True), ('double_chance', 'draw_away', 0, 1, True),
    ('btts', 'no', 0, 2, True), ('btts', 'yes', 1, 1, True),
    ('ou_2.5', 'over', 2, 1, True), ('ou_1.5', 'under', 1, 1, False),
])
def test_regulation_market_results(market, selection, hg, ag, expected):
    assert ForwardStore._won(market, selection, hg, ag) is expected


def test_fresh_capture_does_not_hide_stale_source(tmp_path):
    s = ForwardStore(tmp_path/'db')
    ticket = setup_ticket(s)
    o = s.add_odds([dict(provider_id='1', bookmaker='Book', market='1x2', selection='home', odds=2,
                        captured_at_utc=NOW, source_updated_at=NOW-timedelta(minutes=61))], NOW)[0]
    ticket['legs'][0]['odds_id'] = o['id']
    with pytest.raises(ValueError, match='stale'):
        s.place_ticket(ticket, NOW)
    assert s.tickets() == []


def test_distinct_snapshots_cannot_duplicate_match_exposure(tmp_path):
    path = tmp_path/'db'
    first = ForwardStore(path)
    ticket = setup_ticket(first)
    first.place_ticket(ticket, NOW)
    other = ForwardStore(path)
    # Same real fixtures, new provider quote IDs and forecast revision.
    legs = []
    for leg in ticket['legs']:
        q = next(q for q in first.odds() if q['id'] == leg['odds_id'])
        q['id'] += '-revision'
        quote = other.add_odds([q], NOW)[0]
        legs.append(dict(leg, odds_id=quote['id']))
    with pytest.raises(ValueError, match='exposure'):
        other.place_ticket({'legs': legs}, NOW)
    assert len(other.tickets()) == 1 and other.bankroll()['cash'] == 999
    other.close()
    first.close()


def test_later_legs_settle_after_ticket_already_lost_without_second_credit(tmp_path):
    s=ForwardStore(tmp_path/'db')
    s.place_ticket(setup_ticket(s),NOW,stake=10,initial_balance=100)
    finish(s,'1',hg=0,ag=2)
    assert s.settle(KO+timedelta(hours=2))[0]['status']=='lost'
    assert s.tickets()[0]['legs'][1]['status']=='pending'
    credits=s.conn.execute("SELECT count(*) FROM ledger WHERE kind='credit'").fetchone()[0]
    finish(s,'2',hg=2,ag=0)
    assert s.settle(KO+timedelta(hours=3))==[]
    assert s.tickets()[0]['legs'][1]['status']=='won'
    assert s.bankroll()['cash']==90
    assert s.conn.execute("SELECT count(*) FROM ledger WHERE kind='credit'").fetchone()[0]==credits
    s.close()
