"""Isolated, append-only paper trading records in an explicitly selected SQLite file.

All public timestamps must be timezone-aware UTC datetimes or ISO UTC strings.
Mutation methods return stored dictionaries (batch methods return lists). Ticket
identity defaults to a hash of its immutable leg references and stake; an explicit
``id`` may also be supplied. Money and statistics use decimal currency units.
"""
import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


MARKETS = {
    '1x2': ('home', 'draw', 'away'),
    'double_chance': ('home_draw', 'home_away', 'draw_away'),
    'btts': ('yes', 'no'),
    **{f'ou_{line}': ('over', 'under') for line in (1.5, 2.5, 3.5)},
}


def _iso(value):
    """Normalize an aware UTC timestamp to a fixed-width ISO string."""
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
        if not isinstance(dt, datetime) or dt.utcoffset() != timedelta(0):
            raise ValueError('timestamp must be aware UTC')
        return dt.astimezone(timezone.utc).isoformat(timespec='microseconds')
    except (TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('invalid UTC timestamp') from exc


def utc(value):
    """Parse an ISO string or aware UTC datetime into an aware UTC datetime."""
    return datetime.fromisoformat(_iso(value))


def _number(value, low=0, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected finite number')
    if value < low or (high is not None and value > high):
        raise ValueError('number outside allowed range')
    return float(value)


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('expected nonempty text')
    return value


def _selection(market, selection):
    if market not in MARKETS or selection not in MARKETS[market]:
        raise ValueError('unsupported market or selection')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class ForwardStore:
    def __init__(self, path, *, postponement_hours=48):
        self.path = str(path)
        self.postponement_hours = _number(postponement_hours)
        if self.path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS fixtures (
          provider_id TEXT PRIMARY KEY, league TEXT NOT NULL, season TEXT NOT NULL,
          kickoff_utc TEXT NOT NULL, home TEXT NOT NULL, away TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('scheduled','live','finished','cancelled','abandoned','postponed')),
          hg INTEGER CHECK(hg>=0), ag INTEGER CHECK(ag>=0), updated_at TEXT NOT NULL,
          CHECK(status!='finished' OR (hg IS NOT NULL AND ag IS NOT NULL)));
        CREATE TABLE IF NOT EXISTS live_odds (
          id TEXT PRIMARY KEY, provider_id TEXT NOT NULL REFERENCES fixtures(provider_id),
          bookmaker TEXT NOT NULL, market TEXT NOT NULL, selection TEXT NOT NULL,
          odds REAL NOT NULL CHECK(odds>1 AND odds<1e100), captured_at_utc TEXT NOT NULL,
          source_updated_at TEXT NOT NULL, CHECK(source_updated_at<=captured_at_utc));
        CREATE TABLE IF NOT EXISTS forecasts (
          id TEXT PRIMARY KEY, provider_id TEXT NOT NULL REFERENCES fixtures(provider_id),
          as_of TEXT NOT NULL, kickoff_utc TEXT NOT NULL, model_version TEXT NOT NULL,
          probabilities TEXT NOT NULL, lower TEXT NOT NULL, training_cutoff_utc TEXT NOT NULL,
          UNIQUE(provider_id,as_of,model_version),
          CHECK(training_cutoff_utc<as_of AND as_of<kickoff_utc));
        CREATE TABLE IF NOT EXISTS paper_tickets (
          id TEXT PRIMARY KEY, content_hash TEXT NOT NULL UNIQUE, placed_at TEXT NOT NULL,
          stake REAL NOT NULL CHECK(stake>0), combined_odds REAL NOT NULL CHECK(combined_odds BETWEEN 4.75 AND 5.25),
          status TEXT NOT NULL CHECK(status IN ('pending','won','lost','void')),
          payout REAL NOT NULL DEFAULT 0 CHECK(payout>=0), settled_at TEXT);
        CREATE TABLE IF NOT EXISTS ticket_legs (
          ticket_id TEXT NOT NULL REFERENCES paper_tickets(id),
          forecast_id TEXT NOT NULL REFERENCES forecasts(id), odds_id TEXT NOT NULL REFERENCES live_odds(id),
          provider_id TEXT NOT NULL REFERENCES fixtures(provider_id), market TEXT NOT NULL,
          selection TEXT NOT NULL, odds REAL NOT NULL CHECK(odds>1), kickoff_utc TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','won','lost','void')),
          PRIMARY KEY(ticket_id,provider_id));
        CREATE TABLE IF NOT EXISTS ledger (
          id TEXT PRIMARY KEY, ticket_id TEXT REFERENCES paper_tickets(id),
          kind TEXT NOT NULL CHECK(kind IN ('deposit','debit','credit')), amount REAL NOT NULL,
          created_at TEXT NOT NULL,
          CHECK((kind='debit' AND amount<0) OR (kind!='debit' AND amount>=0)));
        ''')
        for table in ('forecasts', 'live_odds', 'ledger'):
            for operation in ('UPDATE', 'DELETE'):
                self.conn.execute(f'''CREATE TRIGGER IF NOT EXISTS forward_{table}_{operation.lower()}
                    BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable record'); END''')

    def close(self):
        self.conn.close()

    @contextmanager
    def _transaction(self):
        self.conn.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.conn.execute('COMMIT')
        except BaseException:
            self.conn.execute('ROLLBACK')
            raise

    def _rows(self, table):
        return [dict(r) for r in self.conn.execute(f'SELECT * FROM {table} ORDER BY rowid')]

    def fixtures(self):
        return self._rows('fixtures')

    def odds(self):
        return self._rows('live_odds')

    def forecasts(self):
        rows = self._rows('forecasts')
        for row in rows:
            for key in ('probabilities', 'lower'):
                row[key] = json.loads(row[key])
        return rows

    def _fixture(self, provider_id):
        row = self.conn.execute('SELECT * FROM fixtures WHERE provider_id=?', (provider_id,)).fetchone()
        if row is None:
            raise ValueError('unknown fixture')
        return dict(row)

    def upsert_fixtures(self, rows, now):
        now = _iso(now)
        result = []
        with self._transaction():
            for row in rows:
                values = {k: _text(row[k]) for k in ('provider_id', 'league', 'home', 'away', 'status')}
                values.update(season=_text(str(row['season'])), kickoff_utc=_iso(row['kickoff_utc']),
                              hg=row.get('hg'), ag=row.get('ag'), updated_at=now)
                for key in ('hg', 'ag'):
                    if values[key] is not None and (type(values[key]) is not int or values[key] < 0):
                        raise ValueError('goals must be nonnegative integers')
                if values['status'] not in ('scheduled', 'live', 'finished', 'cancelled', 'abandoned', 'postponed'):
                    raise ValueError('invalid fixture status')
                if values['status'] == 'finished' and (values['hg'] is None or values['ag'] is None):
                    raise ValueError('finished fixture needs regulation goals')
                old = self.conn.execute('SELECT updated_at FROM fixtures WHERE provider_id=?', (values['provider_id'],)).fetchone()
                if old and old[0] > now:
                    raise ValueError('stale fixture update')
                keys = list(values)
                self.conn.execute(f"INSERT INTO fixtures ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)}) "
                                  f"ON CONFLICT(provider_id) DO UPDATE SET {','.join(k+'=excluded.'+k for k in keys if k!='provider_id')}", list(values.values()))
                result.append(values)
        return result

    def add_odds(self, rows, now):
        now = _iso(now)
        result = []
        with self._transaction():
            for row in rows:
                fixture = self._fixture(row['provider_id'])
                captured = _iso(row.get('captured_at_utc', now))
                source = _iso(row['source_updated_at'])
                if fixture['status'] != 'scheduled' or not source <= captured <= now < fixture['kickoff_utc']:
                    raise ValueError('odds must be captured before scheduled kickoff and not in the future')
                _selection(row['market'], row['selection'])
                price = _number(row['odds'], 1)
                if price <= 1 or price >= 1e100:
                    raise ValueError('decimal odds must exceed one')
                item = dict(id=_text(row.get('id', str(uuid.uuid4()))), provider_id=fixture['provider_id'],
                            bookmaker=_text(row['bookmaker']), market=row['market'], selection=row['selection'],
                            odds=price, captured_at_utc=captured, source_updated_at=source)
                old = self.conn.execute('SELECT * FROM live_odds WHERE id=?', (item['id'],)).fetchone()
                if old:
                    if dict(old) != item:
                        raise ValueError('odds id content conflict')
                else:
                    self.conn.execute('INSERT INTO live_odds VALUES (?,?,?,?,?,?,?,?)', list(item.values()))
                result.append(item)
        return result

    def add_forecast(self, provider_id, probabilities, lower, as_of, training_cutoff_utc, model_version):
        as_of, cutoff = _iso(as_of), _iso(training_cutoff_utc)
        _text(model_version)
        if not isinstance(probabilities, dict) or not probabilities or not isinstance(lower, dict) or set(lower) != set(probabilities):
            raise ValueError('probability and lower keys must match')
        probabilities, lower = dict(probabilities), dict(lower)
        for key in probabilities:
            if not isinstance(key, str) or key.count(':') != 1:
                raise ValueError('expected market:selection key')
            _selection(*key.split(':'))
            probabilities[key] = _number(probabilities[key], 0, 1)
            lower[key] = _number(lower[key], 0, probabilities[key])
        with self._transaction():
            old = self.conn.execute('SELECT * FROM forecasts WHERE provider_id=? AND as_of=? AND model_version=?',
                                    (provider_id, as_of, model_version)).fetchone()
            if old:
                if (old['probabilities'], old['lower'], old['training_cutoff_utc']) != (_json(probabilities), _json(lower), cutoff):
                    raise ValueError('forecast content conflict')
                return next(r for r in self.forecasts() if r['id'] == old['id'])
            fixture = self._fixture(provider_id)
            if fixture['status'] != 'scheduled' or not cutoff < as_of < fixture['kickoff_utc']:
                raise ValueError('forecast requires training cutoff < as_of < scheduled kickoff')
            identifier = str(uuid.uuid4())
            self.conn.execute('INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?)',
                              (identifier, provider_id, as_of, fixture['kickoff_utc'], model_version,
                               _json(probabilities), _json(lower), cutoff))
            return next(r for r in self.forecasts() if r['id'] == identifier)

    def initialise_bankroll(self, now, initial_balance=1000):
        initial_balance = _number(initial_balance)
        with self._transaction():
            self.conn.execute("INSERT OR IGNORE INTO ledger VALUES ('initial_deposit',NULL,'deposit',?,?)",
                              (initial_balance, _iso(now)))
        return self.bankroll()

    def place_ticket(self, ticket, now, stake=1, initial_balance=1000):
        now, stake = _iso(now), _number(stake)
        initial_balance = _number(initial_balance)
        if stake <= 0:
            raise ValueError('stake must be positive')
        legs = ticket['legs']
        if not 2 <= len(legs) <= 5 or len({x['provider_id'] for x in legs}) != len(legs):
            raise ValueError('need 2–5 distinct fixtures')
        keys = ('forecast_id', 'odds_id', 'provider_id', 'market', 'selection', 'odds')
        content = sorted(({k: leg[k] for k in keys} for leg in legs), key=lambda r: r['provider_id'])
        for leg in content:
            leg['odds'] = _number(leg['odds'], 1)
        digest = hashlib.sha256(_json(dict(legs=content, stake=stake)).encode()).hexdigest()
        identifier = _text(ticket.get('id', digest))
        with self._transaction():
            old = self.conn.execute('SELECT * FROM paper_tickets WHERE id=? OR content_hash=?', (identifier, digest)).fetchone()
            if old:
                if old['content_hash'] != digest:
                    raise ValueError('ticket content conflict')
                return next(t for t in self.tickets() if t['id'] == old['id'])
            # Exposure is checked under BEGIN IMMEDIATE, not just in the runner.
            # Distinct snapshots from concurrent workers must not duplicate a match stake.
            for leg in content:
                if self.conn.execute('SELECT 1 FROM ticket_legs WHERE provider_id=? LIMIT 1',
                                     (leg['provider_id'],)).fetchone():
                    raise ValueError('fixture already has paper exposure')
            bookmaker, combined, stored = None, 1.0, []
            for leg in content:
                forecast = self.conn.execute('SELECT * FROM forecasts WHERE id=?', (leg['forecast_id'],)).fetchone()
                odds = self.conn.execute('SELECT * FROM live_odds WHERE id=?', (leg['odds_id'],)).fetchone()
                fixture = self._fixture(leg['provider_id'])
                if forecast is None or odds is None or forecast['provider_id'] != leg['provider_id']:
                    raise ValueError('unknown or mismatched forecast/odds')
                if any(odds[k] != leg[k] for k in ('provider_id', 'market', 'selection', 'odds')):
                    raise ValueError('leg differs from immutable odds')
                if f"{leg['market']}:{leg['selection']}" not in json.loads(forecast['probabilities']):
                    raise ValueError('selection missing from forecast')
                if fixture['status'] != 'scheduled' or not forecast['as_of'] <= now < min(forecast['kickoff_utc'], fixture['kickoff_utc']):
                    raise ValueError('forecast or fixture not eligible at placement')
                if not odds['source_updated_at'] <= odds['captured_at_utc'] <= now:
                    raise ValueError('future odds')
                if datetime.fromisoformat(now) - datetime.fromisoformat(odds['source_updated_at']) > timedelta(minutes=60):
                    raise ValueError('stale odds source')
                if bookmaker is not None and bookmaker != odds['bookmaker']:
                    raise ValueError('legs must use same bookmaker')
                bookmaker = odds['bookmaker']
                combined *= odds['odds']
                stored.append((*leg.values(), forecast['kickoff_utc']))
            if not 4.75 <= combined <= 5.25:
                raise ValueError('combined odds outside 4.75–5.25')
            self.conn.execute("INSERT OR IGNORE INTO ledger VALUES ('initial_deposit',NULL,'deposit',?,?)", (initial_balance, now))
            if self.bankroll()['cash'] < stake:
                raise ValueError('insufficient funds')
            self.conn.execute("INSERT INTO paper_tickets VALUES (?,?,?,?,?,'pending',0,NULL)", (identifier, digest, now, stake, combined))
            for leg in stored:
                self.conn.execute('INSERT INTO ticket_legs (ticket_id,forecast_id,odds_id,provider_id,market,selection,odds,kickoff_utc) VALUES (?,?,?,?,?,?,?,?)', (identifier, *leg))
            self.conn.execute("INSERT INTO ledger VALUES (?,?,'debit',?,?)", ('debit:'+identifier, identifier, -stake, now))
            return next(t for t in self.tickets() if t['id'] == identifier)

    @staticmethod
    def _won(market, selection, hg, ag):
        if market == '1x2':
            return {'home': hg > ag, 'draw': hg == ag, 'away': hg < ag}[selection]
        if market == 'double_chance':
            return {'home_draw': hg >= ag, 'home_away': hg != ag, 'draw_away': hg <= ag}[selection]
        if market == 'btts':
            return ((hg > 0 and ag > 0) == (selection == 'yes'))
        return ((hg + ag > float(market[3:])) == (selection == 'over'))

    def settle(self, now):
        now = _iso(now)
        settled = []
        with self._transaction():
            for ticket in self.tickets():
                if ticket['placed_at'] > now:
                    continue
                for leg in ticket['legs']:
                    if leg['status'] != 'pending':
                        continue
                    fixture = self._fixture(leg['provider_id'])
                    if fixture['updated_at'] > now:
                        continue
                    deadline = datetime.fromisoformat(leg['kickoff_utc']) + timedelta(hours=self.postponement_hours)
                    state = 'pending'
                    if fixture['status'] in ('cancelled', 'abandoned') or datetime.fromisoformat(fixture['kickoff_utc']) > deadline:
                        state = 'void'
                    elif fixture['status'] == 'postponed' and datetime.fromisoformat(now) >= deadline:
                        state = 'void'
                    elif fixture['status'] == 'finished' and now >= leg['kickoff_utc'] and now >= fixture['kickoff_utc']:
                        state = 'won' if self._won(leg['market'], leg['selection'], fixture['hg'], fixture['ag']) else 'lost'
                    leg['status'] = state
                    self.conn.execute('UPDATE ticket_legs SET status=? WHERE ticket_id=? AND provider_id=?', (state, ticket['id'], leg['provider_id']))
                # A losing leg may have settled the ticket before other matches
                # finished. Resolve their later pick outcomes without a second
                # payout or rewriting the already-final financial result.
                if ticket['status'] != 'pending':
                    continue
                states = [leg['status'] for leg in ticket['legs']]
                if 'lost' in states:
                    status, payout = 'lost', 0.0
                elif 'pending' in states:
                    continue
                else:
                    status = 'void' if all(s == 'void' for s in states) else 'won'
                    payout = ticket['stake'] * math.prod(leg['odds'] if leg['status'] == 'won' else 1 for leg in ticket['legs'])
                self.conn.execute('UPDATE paper_tickets SET status=?, payout=?, settled_at=? WHERE id=?', (status, payout, now, ticket['id']))
                self.conn.execute("INSERT INTO ledger VALUES (?,?,'credit',?,?)", ('credit:'+ticket['id'], ticket['id'], payout, now))
                settled.append(ticket['id'])
        return [t for t in self.tickets() if t['id'] in settled]

    def tickets(self):
        rows = self._rows('paper_tickets')
        for row in rows:
            row['legs'] = [dict(r) for r in self.conn.execute('SELECT * FROM ticket_legs WHERE ticket_id=? ORDER BY provider_id', (row['id'],))]
        return rows

    def bankroll(self):
        ledger, tickets = self._rows('ledger'), self.tickets()
        cash = math.fsum(r['amount'] for r in ledger)
        reserved = math.fsum(t['stake'] for t in tickets if t['status'] == 'pending')
        deposits = math.fsum(r['amount'] for r in ledger if r['kind'] == 'deposit')
        settled = [t for t in tickets if t['status'] != 'pending']
        risked = math.fsum(t['stake'] for t in settled if t['status'] != 'void')
        profit = math.fsum(t['payout'] - t['stake'] for t in settled)
        decided = [t for t in settled if t['status'] in ('won', 'lost')]
        equity, peak, drawdown = deposits, deposits, 0.0
        for t in sorted(settled, key=lambda t: t['settled_at']):
            equity += t['payout'] - t['stake']
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        return dict(cash=cash, reserved=reserved, equity=cash+reserved, initial_balance=deposits,
                    profit=profit, roi=profit/risked if risked else 0.0,
                    hit_rate=sum(t['status'] == 'won' for t in decided)/len(decided) if decided else 0.0,
                    drawdown=peak-(cash+reserved), max_drawdown=drawdown,
                    pending=sum(t['status'] == 'pending' for t in tickets), settled=len(settled))
