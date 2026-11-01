"""Evaluate existing models/markets and replay £10 development accumulators.

Strict verified-price evidence is always written separately from explicitly
hypothetical estimated-timing sensitivity. Never overwrites existing reports.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.football.db import DB_PATH
from src.football.historical import load,load_odds,generate,VERSION,fingerprint
from src.football.market_evaluation import evaluate_markets,grouped_market_metrics
from src.football.accumulator_backtest import replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=DB_PATH)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--phase',choices=['markets','accumulators','all'],default='all')
    args=parser.parse_args()
    output=args.output or Path('reports')/('football_development_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True,exist_ok=True)
    df=load(args.db);model_frames,failures=generate(df,'data/football/development_cache')
    odds=load_odds(args.db,df[df.season.isin([2024,2025])].match_id)
    targets=df[df.season.isin([2024,2025])]
    print(f'Loaded {len(targets)} target matches; verified individual prices: {int(odds.timestamp_verified.sum())}',flush=True)
    manifest=dict(version=VERSION,source_fingerprint=fingerprint(df),development_seasons=[2024,2025],
                  excluded_from_development_run=[2026],
                  independent_validation='Prospective forecasts after the recorded policy lock only; 2026/27 was already inspected in legacy reports',stake=10,selection_policy='one selection per match; same bookmaker; 4.75–5.25; at most 3 disjoint tickets per decision',
                  uncertainty_policy='league/selection/probability-bin Wilson95, min100 earlier resolved matches, 5% per-additional-leg dependence haircut, minimum conservative EV2%',
                  candidate_policy='individual conservative EV strictly positive; beam250, top60 candidates; bounded not exhaustive',
                  fit_failures=failures,verified_historical_price_rows=int(odds.timestamp_verified.sum()),
                  evidence_warning='2024/25 and 2025/26 are development backtests. Estimated timing is never proof of price availability. No profitable strategy established.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.phase in ('markets','all'):
        tables=evaluate_markets(df,model_frames,odds)
        tables['market_groups']=grouped_market_metrics(df,model_frames,odds)
        for name,table in tables.items():
            table.to_csv(output/(name+'.csv'),index=False)
        for name,frame in model_frames.items():
            scored=df[df.season>=2023][['match_id','season','div','date','decision','kickoff_utc','estimated_schedule_utc','home','away','hg','ag']].join(frame)
            scored.to_csv(output/(name.lower().replace('-','_')+'_forecasts.csv'),index=False)
        print('Phase 2 complete: all market metrics, calibration, actual-price coverage and singles sensitivity written',flush=True)
    if args.phase in ('accumulators','all'):
        summaries=[]
        for name,frame in model_frames.items():
            for mode in ['verified','estimated_sensitivity']:
                result=replay(df,frame,odds,model_name=name,mode=mode,stake=10)
                prefix=name.lower().replace('-','_')+'_'+mode
                (output/(prefix+'_tickets.json')).write_text(json.dumps(result['tickets'],indent=2))
                pd.DataFrame(result['decisions']).to_csv(output/(prefix+'_decisions.csv'),index=False)
                summaries.extend(result['summaries'])
                flat=[]
                for ticket in result['tickets']:
                    for leg in ticket['legs']:
                        flat.append({**{k:v for k,v in ticket.items() if k!='legs'},**leg})
                pd.DataFrame(flat).to_csv(output/(prefix+'_legs.csv'),index=False)
                print(name,mode,[(s['season'],s['tickets'],s['net_profit']) for s in result['summaries']],flush=True)
        pd.DataFrame(summaries).to_csv(output/'accumulator_summary.csv',index=False)
        (output/'accumulator_summary.json').write_text(json.dumps(summaries,indent=2))
        print('Phases 3–4 complete: chronological tickets, legs, no-bet decisions, settlement and financial metrics written',flush=True)
    print('Output:',output.resolve(),flush=True)


if __name__=='__main__':
    main()
