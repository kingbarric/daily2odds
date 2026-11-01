"""Plain HTML/Markdown report from the executed development replay artifacts."""
import argparse
from datetime import datetime,timezone
from html import escape
import json
from math import prod
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from src.football.forward_store import ForwardStore
from src.football.data import LEAGUES


def write_report(folder,paper_db):
    folder=Path(folder)
    groups=pd.read_csv(folder/'market_groups.csv')
    summary=json.loads((folder/'accumulator_summary.json').read_text())
    metrics=pd.read_csv(folder/'metrics.csv')
    all_tickets=[]
    for p in folder.glob('*_tickets.json'):
        all_tickets.extend(json.loads(p.read_text()))
    # Model comparisons use only the identical bookmaker-covered subset.
    comparisons=[]
    for model,g in groups[(groups.market=='1x2')&(groups.status=='supported')].groupby('model'):
        n=g.predictions.sum();bn=g.bookmaker_comparison_n.sum()
        comparisons.append(dict(model=model,predictions=int(n),accuracy=(g.accuracy*g.predictions).sum()/n,
                                log_loss=(g.log_loss*g.predictions).sum()/n,
                                bookmaker_comparison_n=int(bn),
                                matched_model_log_loss=(g.matched_model_log_loss*g.bookmaker_comparison_n).sum()/bn,
                                bookmaker_log_loss=(g.bookmaker_log_loss*g.bookmaker_comparison_n).sum()/bn,
                                bookmaker_accuracy=(g.bookmaker_accuracy*g.bookmaker_comparison_n).sum()/bn))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(folder/'model_comparison.csv',index=False)
    # Full-season market/league ranks, not picks chosen retrospectively for tickets.
    ranked=groups[(groups.model=='Dixon-Coles')&(groups.status=='supported')].copy()
    ranked['league_market']=ranked.league+' / '+ranked.market
    rows=[]
    for (league,market),g in ranked.groupby(['league','market']):
        n=g.predictions.sum()
        rows.append(dict(league=league,market=market,predictions=int(n),
                         accuracy=(g.accuracy*g.predictions).sum()/n,
                         brier=(g.brier*g.predictions).sum()/n,log_loss=(g.log_loss*g.predictions).sum()/n))
    ranks=pd.DataFrame(rows).sort_values(['market','log_loss'])
    ranks.insert(1,'league_name',ranks.league.map(lambda code:LEAGUES.get(code,('',code))[1]))
    ranks.to_csv(folder/'market_league_ranking.csv',index=False)
    stress=[]
    for ticket in all_tickets:
        marginals=[leg['conservative_probability'] for leg in ticket['legs']]
        joint10=prod(marginals)*.9**(len(marginals)-1)
        frechet=max(0.,sum(marginals)-(len(marginals)-1))
        stress.append(dict(ticket_id=ticket['ticket_id'],model=ticket['model'],season=ticket['season'],timing=ticket['timing'],
                           leg_count=len(marginals),nominal_joint=ticket['joint_probability'],
                           heuristic_joint=ticket['conservative_joint_probability'],
                           ten_percent_dependence_stress_ev=joint10*ticket['combined_odds']-1,
                           frechet_ev_from_conservative_marginals=frechet*ticket['combined_odds']-1))
    pd.DataFrame(stress).to_csv(folder/'dependence_stress.csv',index=False)
    curves=[]
    for timing in ('verified','estimated_sensitivity'):
        for model in ('Dixon-Coles','Elo','LightGBM'):
            records=[t for t in all_tickets if t['timing']==timing and t['model']==model]
            for season in (2024,2025):
                records_s=[t for t in records if t['season']==season]
                by_time={}
                for ticket in records_s:
                    by_time.setdefault(ticket['settled_at_utc'],[]).append(ticket)
                net=peak=0
                for when,batch in sorted(by_time.items()):
                    net+=sum(t['net_profit'] for t in batch);peak=max(peak,net)
                    curves.append(dict(model=model,timing=timing,season=season,settled_at_utc=when,
                                       net_profit=net,drawdown=peak-net,tickets_settled=len(batch)))
    pd.DataFrame(curves).to_csv(folder/'equity_curve.csv',index=False)
    prospective=[];forward_summary={};fixtures=[]
    if Path(paper_db).exists():
        store=object.__new__(ForwardStore)
        store.conn=sqlite3.connect(f'file:{Path(paper_db).resolve()}?mode=ro',uri=True);store.conn.row_factory=sqlite3.Row
        try:
            fixtures=store.fixtures();by_id={f['provider_id']:f for f in fixtures}
            quote_rows=store.odds()
            quote_frame=pd.DataFrame(quote_rows)
            if len(quote_frame):
                now=pd.Timestamp.now(tz='UTC')
                quote_frame['source_fresh_within_60m']=(now-pd.to_datetime(quote_frame.source_updated_at,utc=True))<=pd.Timedelta(minutes=60)
                quote_frame.to_csv(folder/'current_odds_snapshots.csv',index=False)
            latest={}
            for forecast in store.forecasts():
                if forecast['provider_id'] not in latest or forecast['as_of']>latest[forecast['provider_id']]['as_of']:
                    latest[forecast['provider_id']]=forecast
            for f in latest.values():
                fixture=by_id[f['provider_id']]
                prospective.append(dict(provider_id=f['provider_id'],league=fixture['league'],home=fixture['home'],away=fixture['away'],
                    kickoff_utc=fixture['kickoff_utc'],forecast_created_at_utc=f['as_of'],model_version=f['model_version'],
                    **f['probabilities']))
            forward_summary=store.bankroll()
        finally:store.close()
    current=pd.DataFrame(prospective)
    current.to_csv(folder/'current_forward_predictions.csv',index=False)
    status_path=Path(paper_db).parent/'latest_forward_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    manifest=json.loads((folder/'manifest.json').read_text())
    lock=Path(paper_db).parent/'forward_policy_lock_v1.json'
    policy=json.loads(lock.read_text()) if lock.exists() else {}
    if policy:(folder/'prospective_policy_lock.json').write_text(json.dumps(policy,indent=2))
    payload=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),model_comparison=comparisons,
                 accumulator_summary=summary,forward_status=status,paper_bankroll=forward_summary,
                 prospective_policy_lock=policy,profitability_conclusion='No statistically credible, verified-price profitable strategy established.')
    (folder/'report_summary.json').write_text(json.dumps(payload,indent=2))
    def html_table(frame,percent_columns=()):
        frame=frame.copy()
        for c in percent_columns:
            if c in frame:
                frame[c]=frame[c].map(lambda x:f'{x:.1%}' if pd.notna(x) else 'Unavailable')
        return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}',na_rep='Unavailable',escape=True)
    financial=[]
    for s in summary:
        ci=s['roi_ci95']
        financial.append(dict(model=s['model'],season=f"{s['season']}/{str(s['season']+1)[-2:]}",timing=s['timing'],
            tickets=s['tickets'],wins=s['wins'],losses=s['losses'],stakes_gbp=s['stakes'],payouts_gbp=round(s['payouts'],2),
            net_profit_gbp=round(s['net_profit'],2),roi=s['roi'],hit_rate=s['hit_rate'],
            maximum_drawdown_gbp=round(s['maximum_drawdown'],2),maximum_losing_streak=s['maximum_losing_streak'],
            roi_ci95=f'{ci[0]:.1%} to {ci[1]:.1%}' if ci else 'Insufficient decision-day clusters'))
    finance=pd.DataFrame(financial)
    forward_display=current[[c for c in ['league','home','away','kickoff_utc','1x2:home','1x2:draw','1x2:away','forecast_created_at_utc'] if c in current]] if len(current) else current
    links=[]
    for name in ['predictions.csv','metrics.csv','market_groups.csv','calibration.csv','coverage.csv','singles.csv',
                 'accumulator_summary.csv','equity_curve.csv','dependence_stress.csv','current_forward_predictions.csv','manifest.json','prospective_policy_lock.json','validation.md','tests.log','preservation_manifest.json','current_odds_snapshots.csv','report_summary.json']:
        if (folder/name).exists():links.append(f'<li><a href="{escape(name)}">{escape(name)}</a></li>')
    for p in sorted(folder.glob('*_tickets.json')):
        links.append(f'<li><a href="{escape(p.name)}">{escape(p.name)} — every ticket and leg</a></li>')
    for p in sorted(folder.glob('*_forecasts.csv')):
        links.append(f'<li><a href="{escape(p.name)}">{escape(p.name)} — every model forecast</a></li>')
    sections=[
        ('Summary','<p><strong>No credible profitable strategy established.</strong> Both replay seasons are development data. '
         'There are zero verified historical price timestamps: all strict accumulator runs return NO BET, with no stakes and undefined ROI. '
         'Estimated-timing sensitivity returns below are hypothetical and do not prove price availability.</p>'),
        ('Data integrity','<p>Results/features are censored at decision-day midnight; head-to-head, rolling form, rest, Elo and season indicators '
         'do not inspect later results or future fixture membership. Calibration uses only earlier settled matches from the same league. '
         'Models were not retuned on these runs. Source database and pre-existing reports are preserved byte-for-byte.</p>'),
        ('Existing models against bookmakers — 1X2',html_table(comparison,['accuracy','bookmaker_accuracy'])+
         '<p>Lower log loss/Brier is better. Bookmaker probabilities remove the complete-market overround proportionally. '
         'Comparisons use matched price-covered rows; these historical price benchmarks are descriptive, not proof of executable odds.</p>'),
        ('Market and league results',html_table(ranks,['accuracy'])+
         '<p>Ranked within each market by log loss. High accuracy on Over/Under 1.5 or double chance reflects easier/base-rate events; '
         'it is not a profitability ranking. Full model and selection-level scores, calibration and coverage are linked below. '
         'Elo and LightGBM support 1X2/double chance only. Other pairs are explicitly unsupported. '
         'Only 1X2 and Over/Under 2.5 have actual historical odds. Double-chance bookmaker event probabilities are derived from de-vigged 1X2; no double-chance prices are fabricated.</p>'),
        ('£10 accumulator backtests',html_table(finance,['roi','hit_rate'])+
         '<p>2–5 legs, distinct matches, same bookmaker, 4.75–5.25 odds, at most three match-disjoint tickets per decision. '
         'Wilson95 probability-bin bounds require 100 prior settled matches per league/selection, with a 5% dependence haircut per additional leg and at least 2% conservative EV. '
         'The dependence penalty is a heuristic, not a joint confidence guarantee. dependence_stress.csv shows a stronger 10% haircut and the Fréchet worst-case event-overlap stress using conservative marginal estimates; these are diagnostics, not extra independently validated strategies. Beam250/top60 candidates bound search; global optimality is not guaranteed. '
         'Exchange/market-average/max quotes are not used as bettable prices. Payouts round to pennies HALF_UP. '
         'Drawdown uses realised settlement batches; losing streaks use settlement order and deterministic ticket IDs for ties. '
         'ROI intervals bootstrap decision-day clusters and do not correct for tuning, model selection or unverified prices. Small positive sensitivity samples are not evidence of profitability.</p>'),
        ('Current five-day forward forecasts',html_table(forward_display,['1x2:home','1x2:draw','1x2:away'])+
         f'<p>Decision: <strong>{escape(status.get("decision","Unavailable"))}</strong>. '
         f'Coverage: {escape(status.get("fixture_coverage","Unknown"))}. '
         'The free API-Football plan does not allow current-season access. Daily public scoreboards supply fixtures across the ten configured leagues; '
         'explicit league IDs and canonical team aliases exclude cup games and prevent duplicate match IDs. The publisher CSV supplies actual available quotes for its fixture batch only. '
         'Upstream schedule completeness is not contractually guaranteed. Quotes record real receipt and batch publication separately; stale batch quotes cannot qualify. '
         'No paper tickets were forced. These current forecasts have no settled predictive-performance sample. '
         'Actual quote receipts and source freshness flags are included in current_odds_snapshots.csv.</p>'),
        ('Paper bankroll',html_table(pd.DataFrame([forward_summary])) if forward_summary else '<p>No paper records.</p>'),
        ('Final validation and automation','<p>2026/27 was already inspected in earlier reports and is not wholly untouched. '
         'This development run excludes it. Final validation is reserved for prospective immutable forecasts after the recorded policy lock. '
         'The existing Dixon–Coles deployment remains in place; no selection of the luckiest backtest model was made.</p>'+
         '<pre>.venv/bin/python football_forward.py --provider auto --watch --stake 10\n.venv/bin/python -m streamlit run football_dashboard.py</pre>'+
         '<p>The persistent worker schedules collection, immutable forecasts, paper placement and result settlement every eight hours. '
         'No OS background service is silently installed. Use an entitled API-Football account with --provider api-football for broader/current odds coverage.</p>'),
        ('Files','<ul>'+''.join(links)+'</ul>')]
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+\
        '<title>Football development backtests 2024/25–2025/26</title><style>body{font:16px/1.5 system-ui,sans-serif;margin:28px auto;padding:0 18px;max-width:1400px;color:#222}table{border-collapse:collapse;min-width:500px}th,td{border:1px solid #ccc;padding:7px;text-align:left}th{background:#eee}section{overflow-x:auto;margin:28px 0}a{color:#174ea6}pre{background:#f5f5f5;padding:12px}</style></head><body>'+\
        '<h1>Football development backtests: 2024/25 and 2025/26</h1>'+''.join(f'<section><h2>{escape(title)}</h2>{body}</section>' for title,body in sections)+'</body></html>'
    (folder/'report.html').write_text(html)
    text=['# Football development backtests: 2024/25 and 2025/26','',
          '**No credible profitable strategy established.** Strict verified-price accumulator runs: NO BET; stakes £0; ROI undefined.',
          '', '## 1X2 models against bookmaker baseline','',comparison.to_markdown(index=False),
          '', '## £10 accumulator results','',finance.to_markdown(index=False),
          '', 'These are development backtests. Estimated-timing results do not establish executable prices. '
          'Positive small-sample sensitivity results do not justify selecting or deploying a strategy.',
          '', '## Current forecasts','',forward_display.to_markdown(index=False) if len(current) else 'Unavailable',
          '', f"Current decision: {status.get('decision','Unavailable')}; {status.get('fixture_coverage','Unknown coverage')}.",
          '', 'See report.html for market/league results, interpretation, financial definitions, calibration/coverage links and all winning/losing tickets.',
          '', 'Existing history and reports preserved. Only prospective data after the policy lock can count toward new independent validation.']
    (folder/'report.md').write_text('\n'.join(text)+'\n')
    print('Wrote',folder/'report.html',flush=True)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=Path('reports/football_development_2024_2025_integrity_v3'))
    p.add_argument('--paper-db',type=Path,default=Path('data/football/paper_forward.db'))
    a=p.parse_args();write_report(a.folder,a.paper_db)
