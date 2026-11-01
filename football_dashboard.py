"""Read-only Streamlit dashboard: streamlit run football_dashboard.py."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.football import db, evaluate
from src.football.forward_store import ForwardStore
from src.football.workflow import candidates, instant
from src.football.optimiser import optimise


def main():
    st.set_page_config(page_title='Football paper predictions', layout='wide')
    st.title('Football predictions · paper only')
    st.caption('Probabilities are estimates. No demonstrated profitable edge. All stakes and returns are hypothetical.')
    path = Path(os.getenv('FOOTBALL_DB_PATH', str(db.DB_PATH.parent/'paper_forward.db')))
    if not path.is_file():
        st.info('No database for paper records. Run football_forward.py; existing historical data is preserved. No sample data is displayed.')
        return
    # Read-only URI: dashboard must never migrate or create the user's database.
    import sqlite3
    store = object.__new__(ForwardStore)
    store.conn = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    store.conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'forecasts' not in tables:
            st.info('Historical database found. Run football_forward.py to initialise the forward paper records.')
            return
        fixtures, forecasts, odds, tickets = store.fixtures(), store.forecasts(), store.odds(), store.tickets()
        now = datetime.now(timezone.utc)
        st.button('Refresh')
        latest = max((q['captured_at_utc'] for q in odds), default=None)
        st.caption(f'UTC now: {now.isoformat()} · Last collected odds: {latest or "none"} · Forecasts: {len(forecasts)}')
        tabs = st.tabs(['Upcoming fixtures', 'Ranked singles', '~5.00 tickets', 'Prediction history', 'Diagnostics', 'Paper bankroll'])
        picks, _ = candidates(store, now)
        with tabs[0]:
            upcoming = [f for f in fixtures if instant(f['kickoff_utc']) > now and f['status'] in ('scheduled', 'postponed')]
            st.dataframe(pd.DataFrame(upcoming), use_container_width=True)
        with tabs[1]:
            rows = [dict(match=f'{p.home} – {p.away}', market=p.market, selection=p.selection,
                         bookmaker=p.bookmaker, odds=p.odds, probability=p.probability,
                         conservative_probability=p.p_lower, nominal_ev=p.probability*p.odds-1,
                         conservative_ev=p.p_lower*p.odds-1, captured_at=p.captured_at_utc,
                         kickoff=p.kickoff_utc) for p in picks]
            st.caption('Rankings are estimates, not recommendations. Conservative probabilities require at least 100 settled matches per probability bin.')
            frame = pd.DataFrame(rows)
            if not frame.empty:
                frame = frame.sort_values('conservative_ev', ascending=False)
            st.dataframe(frame, use_container_width=True)
        with tabs[2]:
            proposed = optimise(picks, now=now, min_robust_ev=0.02)
            if not proposed:
                st.info('NO BET — no tickets satisfy current price, timing and uncertainty thresholds.')
            for i, ticket in enumerate(proposed, 1):
                st.write(f'Ticket {i}: odds {ticket.combined_odds:.2f} · model win probability {ticket.probability:.1%} · conservative EV {ticket.robust_ev:.1%}')
                st.dataframe(pd.DataFrame([dict(match=f'{p.home} – {p.away}', market=p.market, selection=p.selection, bookmaker=p.bookmaker, odds=p.odds) for p in ticket.legs]))
            st.caption('Different matches and one bookmaker per ticket. Independence plus a 5% per-additional-leg correlation haircut is a heuristic, not a joint confidence guarantee. Search is bounded and tickets use disjoint matches.')
            st.write('Recorded paper tickets')
            st.dataframe(pd.DataFrame([{k: v for k, v in t.items() if k != 'legs'} for t in tickets]))
        with tabs[3]:
            st.dataframe(pd.DataFrame([{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in f.items()} for f in forecasts]), use_container_width=True)
        with tabs[4]:
            st.write('Frozen forward predictions only; one earliest forecast per match. Historical reports remain separate legacy evaluations with estimated odds timing.')
            by_id = {f['provider_id']: f for f in fixtures}
            seen, probabilities, outcomes = set(), [], []
            for forecast in sorted(forecasts, key=lambda f: f['as_of']):
                fixture = by_id[forecast['provider_id']]
                if fixture['status'] != 'finished' or fixture['provider_id'] in seen:
                    continue
                keys = ['1x2:home', '1x2:draw', '1x2:away']
                if not all(k in forecast['probabilities'] for k in keys):
                    continue
                seen.add(fixture['provider_id'])
                probabilities.append([forecast['probabilities'][k] for k in keys])
                outcomes.append(0 if fixture['hg'] > fixture['ag'] else 1 if fixture['hg'] == fixture['ag'] else 2)
            if probabilities:
                p, y = np.array(probabilities), np.array(outcomes)
                st.json(evaluate.scores(p, y))
                st.dataframe(evaluate.calibration(p, np.eye(3)[y]))
            else:
                st.info('No settled frozen forward predictions yet. Accuracy, calibration and returns are not established.')
        with tabs[5]:
            bankroll = store.bankroll()
            cols = st.columns(4)
            for col, key in zip(cols, ('cash', 'reserved', 'equity', 'profit')):
                col.metric(key.replace('_', ' ').title(), f'{bankroll[key]:.2f}')
            st.json(bankroll)
            st.caption('Equity values unresolved stakes at cost. Drawdown is realised paper equity drawdown; void legs pay at 1.00. This paper settlement policy is not a bookmaker-specific promise.')
    finally:
        store.close()


if __name__ == '__main__':
    main()
