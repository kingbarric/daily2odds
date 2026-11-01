"""Probability-first view of the trained multi-market forecast and actual live odds.

Does not place tickets, alter forecasts, invent missing prices or bypass the
uncertainty gate. Multiple markets on one fixture are alternatives, not
independent opportunities. Model expected goals are not observed shot-based xG.
"""
import argparse
from datetime import datetime,timezone
from html import escape
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from src.football.forward_store import ForwardStore
from src.football.workflow import candidates,load_history,merge_forward_history,predict_fixtures
from src.football.db import DB_PATH

LABELS={'1x2:home':'Home win','1x2:draw':'Draw','1x2:away':'Away win',
        'double_chance:home_draw':'1X (home or draw)','double_chance:draw_away':'X2 (draw or away)',
        'double_chance:home_away':'12 (either team wins)','btts:yes':'BTTS Yes','btts:no':'BTTS No',
        **{f'ou_{line}:{side}':f'{side.title()} {line} goals' for line in (1.5,2.5,3.5) for side in ('over','under')}}


def build(paper_db,training_folder,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    store=object.__new__(ForwardStore)
    store.conn=sqlite3.connect(f'file:{Path(paper_db).resolve()}?mode=ro',uri=True);store.conn.row_factory=sqlite3.Row
    try:
        snapshots=store.forecasts();fixtures={f['provider_id']:f for f in store.fixtures()};latest={}
        for f in snapshots:
            if f['provider_id'] not in latest or f['as_of']>latest[f['provider_id']]['as_of']:latest[f['provider_id']]=f
        picks,_=candidates(store,datetime.now(timezone.utc));rows=[]
        for p in picks:
            key=p.market+':'+p.selection
            rows.append(dict(provider_id=p.match_id,match=f'{p.home} – {p.away}',market=p.market,selection=LABELS[key],
                probability=p.probability,conservative_probability=p.p_lower,odds=p.odds,
                break_even_probability=1/p.odds,nominal_ev=p.probability*p.odds-1,
                bookmaker=p.bookmaker.replace('odds_api:',''),kickoff_utc=p.kickoff_utc.isoformat(),
                odds_captured_at_utc=p.captured_at_utc.isoformat(),forecast_as_of_utc=latest[p.match_id]['as_of']))
        ranked=pd.DataFrame(rows)
        if len(ranked):
            # Choose the best actual quoted price for the SAME event/selection;
            # no synthetic prices or combining bookmakers into a ticket price.
            ranked=ranked.sort_values(['provider_id','market','selection','odds'],ascending=[True,True,True,False])
            ranked=ranked.drop_duplicates(['provider_id','market','selection'])
            ranked=ranked.sort_values(['probability','odds'],ascending=[False,False]).reset_index(drop=True)
        ranked.to_csv(output/'ranked_probability_selections.csv',index=False)
        allrows=[]
        for f in latest.values():
            fixture=fixtures[f['provider_id']]
            allrows.append(dict(provider_id=f['provider_id'],home=fixture['home'],away=fixture['away'],
                league=fixture['league'],kickoff_utc=fixture['kickoff_utc'],as_of_utc=f['as_of'],model_version=f['model_version'],**f['probabilities']))
        pd.DataFrame(allrows).to_csv(output/'all_forecasts.csv',index=False)
        diagnostics={}
        if latest:
            asof=min(f['as_of'] for f in latest.values())
            history,_=load_history(DB_PATH,asof)
            history=merge_forward_history(history,store.fixtures(),asof)
            target=[fixtures[i] for i in latest]
            _,version=predict_fixtures(history,target,asof,diagnostics=diagnostics)
            if not all(f['model_version'].startswith(version) for f in latest.values()):
                raise ValueError('Cannot reproduce recorded training inputs for expected-goals diagnostics')
        goal_rows=[dict(match=f"{fixtures[i]['home']} – {fixtures[i]['away']}",kickoff_utc=fixtures[i]['kickoff_utc'],**d) for i,d in diagnostics.items()]
        goals=pd.DataFrame(goal_rows).sort_values('kickoff_utc') if goal_rows else pd.DataFrame()
        goals.to_csv(output/'model_expected_goals.csv',index=False)
        bank=store.bankroll();ticket_count=len(store.tickets())
    finally:store.close()
    status=json.loads(Path(paper_db).with_suffix('.status.json').read_text())
    evaluation=pd.read_csv(Path(training_folder)/'training_evaluation.csv')
    metadata=json.loads((Path(training_folder)/'training_metadata.json').read_text())
    def table(frame):
        frame=frame.copy()
        for col in ['probability','conservative_probability','break_even_probability','nominal_ev','accuracy']:
            if col in frame:frame[col]=frame[col].map(lambda x:f'{x:.1%}')
        for col in ['log_loss','brier','rps']:
            if col in frame:frame[col]=frame[col].map(lambda x:f'{x:.4f}')
        return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.2f}',escape=True)
    columns=['match','selection','probability','odds','bookmaker','nominal_ev','kickoff_utc']
    top=ranked.drop_duplicates('provider_id').head(15)[columns] if len(ranked) else ranked
    value_shortlist=ranked[ranked.nominal_ev>=.02].drop_duplicates('provider_id').head(15)[columns] if len(ranked) else ranked
    value_shortlist.to_csv(output/'probability_value_shortlist.csv',index=False)
    groups=[]
    if len(ranked):
        for market,g in ranked.groupby('market',sort=False):
            groups.append('<h3>'+escape(market)+'</h3>'+table(g.head(5)[columns]))
    payload=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),status=status,training=metadata,
        fresh_quote_candidates=len(picks),unique_priced_selections=len(ranked),paper_tickets=ticket_count,
        conclusion='Probability-first analysis, not a promise of wins or profitability. NO BET until uncertainty/quality rules qualify.',paper_bankroll=bank)
    (output/'summary.json').write_text(json.dumps(payload,indent=2))
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Football: probability-first forecasts</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:28px auto;padding:0 16px;color:#222}section{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;min-width:650px}th,td{border:1px solid #ccc;padding:7px;text-align:left}th{background:#eee}a{color:#174ea6}</style></head><body>'''
    page+='<h1>Football: probability-first forecasts</h1>'
    page+='<p>Forecast snapshot generated '+escape(payload['generated_at_utc'])+'. Prices are timestamped snapshots and may change; this static report is not a continuously updating odds screen.</p>'
    page+=f'<p>{status["forecasts"]} new forecasts; {status["source_metadata"]["fixtures_with_quotes"]} fixtures with matched The Odds API prices; {len(picks):,} fresh bookmaker offers. UK decimal odds. API credits remaining: {status["source_metadata"]["quota"].get("remaining","unknown")}.</p>'
    page+='<p><strong>Double chance, goal totals and BTTS are included wherever genuine prices are available.</strong> Selections are ranked by estimated probability, not by the largest payout. Each displayed price is an actual offered price from one bookmaker. A high-probability selection can still be overpriced: nominal EV is model probability × odds − 1, not a verified advantage.</p>'
    page+=f'<p><strong>Paper decision: {escape(status["decision"])}.</strong> Tickets must still pass uncertainty and quality checks. No paper stakes were forced; experimental balance £{bank["cash"]:.2f}. The new odds-aware head is experimental and does not overwrite earlier forecasts or count toward the previously frozen strategy validation.</p>'
    page+='<section><h2>Highest probability: one selection per match</h2>'+table(top)+'</section>'
    page+='<section><h2>Highest probability with positive model-estimated value</h2><p>At least 2% nominal model EV, one selection per match. These are analysis candidates, not qualified paper bets: forward uncertainty checks have not passed. Apparent value can reflect model error.</p>'+table(value_shortlist)+'</section>'
    page+='<section><h2>By betting market</h2><p>Markets on the same match are alternatives, not independent opportunities. Missing bookmaker markets are not assigned invented prices.</p>'+''.join(groups)+'</section>'
    page+='<section><h2>Model expected goals</h2><p>Existing Dixon–Coles goal-rate estimates, reproduced from the recorded training inputs. These are model expected goals, not shot-based observed xG or guaranteed scorelines.</p>'+table(goals)+'</section>'
    page+='<section><h2>Historical training evaluation</h2><p>The existing LightGBM was trained with pre-closing bookmaker probabilities and past match features. Target scores and closing prices are excluded from inputs. These seasons are development data and historical quote timestamps remain unverified. The odds-aware model improved on result-only features but did not beat the bookmaker baseline. LightGBM supplies 1X2 and derived double chance; existing Dixon–Coles supplies totals and BTTS, with no numerical model blending.</p>'+table(evaluation)+'</section>'
    page+='<p><a href="ranked_probability_selections.csv">All ranked selections</a> · <a href="all_forecasts.csv">All forecasts</a> · <a href="probability_value_shortlist.csv">Probability/value shortlist</a> · <a href="model_expected_goals.csv">Expected goals</a> · <a href="summary.json">Source, quota and training details</a></p></body></html>'
    (output/'report.html').write_text(page)
    print('Report:',(output/'report.html').resolve())
    if len(ranked):print(ranked[columns].head(12).to_string(index=False))
    print('Priced market counts:',ranked.groupby('market').size().to_dict() if len(ranked) else {})
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper-db',type=Path,default=Path('data/football/paper_odds_api_20261029_v2.db'))
    p.add_argument('--training-folder',type=Path,default=Path('reports/football_odds_training_20261029'))
    p.add_argument('--output',type=Path,default=Path('reports/football_probability_forecast_20261029'))
    a=p.parse_args();build(a.paper_db,a.training_folder,a.output)
