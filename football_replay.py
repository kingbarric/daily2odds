"""Historical daily replay of the existing forward model; never writes live records.

Existing settings were selected using seasons including 2024/25. Consequently this
is a retrospective replay with chronological training, not an untouched test of
hyperparameter selection. No odds, ticket ROI or verified pre-kickoff records are
manufactured from historical fixtures.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.football import db, evaluate, markets
from src.football.data import LEAGUES
from src.football.workflow import load_history, predict_fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2024)
    parser.add_argument('--leagues', nargs='+', choices=list(LEAGUES), default=list(LEAGUES))
    parser.add_argument('--db', type=Path, default=db.DB_PATH)
    args = parser.parse_args()
    history, _ = load_history(args.db, datetime(args.season+2, 1, 1, tzinfo=timezone.utc))
    targets = history[(history.season == args.season) & history['div'].isin(args.leagues)].copy()
    if targets.empty:
        raise SystemExit('No historical matches for requested season/leagues')
    out, failures = [], []
    for league in args.leagues:
        target = targets[targets['div'] == league]
        for day, group in target.groupby('date', sort=True):
            simulated = day.to_pydatetime().replace(tzinfo=timezone.utc)
            fixtures = [dict(provider_id=r.match_id, league=league, home=r.home, away=r.away)
                        for r in group.itertuples()]
            try:
                predictions, version = predict_fixtures(history, fixtures, simulated)
            except ValueError as exc:
                failures.append(dict(league=league, day=str(day.date()), reason=str(exc)))
                continue
            for r in group.itertuples():
                p = predictions.get(r.match_id)
                if p is None:
                    failures.append(dict(league=league, day=str(day.date()), reason=f'No forecast for {r.match_id}'))
                    continue
                result = dict(match_id=r.match_id, league=league, home=r.home, away=r.away,
                              date=str(day.date()), simulated_as_of_utc=simulated.isoformat(),
                              training_dates_strictly_before=str(day.date()), model_version=version,
                              actual_home_goals=int(r.hg), actual_away_goals=int(r.ag), **p)
                q = [p['1x2:'+s] for s in ('home','draw','away')]
                result['predicted_result'] = ('home','draw','away')[int(np.argmax(q))]
                result['actual_result'] = ('home','draw','away')[int(r.result)]
                result['correct'] = result['predicted_result'] == result['actual_result']
                out.append(result)
        print(f'{league}: {sum(r["league"] == league for r in out)}/{len(target)} forecasts', flush=True)
    if not out:
        raise SystemExit('No forecasts generated')
    frame = pd.DataFrame(out).sort_values(['date','league','home']).reset_index(drop=True)
    folder = Path('reports') / f'football_replay_{args.season}'
    folder.mkdir(exist_ok=True)
    frame.to_csv(folder/'predictions.csv', index=False)
    metrics, market_rows = [], []
    for league, group in frame.groupby('league', sort=False):
        p = group[markets.columns('1x2')].to_numpy()
        y = np.array([('home','draw','away').index(r) for r in group.actual_result])
        scores = evaluate.scores(p,y)
        scores.update(league=league, league_name=LEAGUES[league][1], correct=int(group.correct.sum()),
                      always_home_accuracy=float((y==0).mean()))
        metrics.append(scores)
    p = frame[markets.columns('1x2')].to_numpy()
    y = np.array([('home','draw','away').index(r) for r in frame.actual_result])
    overall = evaluate.scores(p,y)
    overall.update(correct=int(frame.correct.sum()), always_home_accuracy=float((y==0).mean()))
    for market in markets.EXCLUSIVE:
        p = frame[markets.columns(market)].to_numpy()
        y = markets.outcome(market,frame.actual_home_goals,frame.actual_away_goals)
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1):
            raise RuntimeError(f'Invalid {market} probability distribution')
        market_rows.append(dict(market=market, **evaluate.scores(p,y)))
    summary = dict(season=f'{args.season}/{(args.season+1)%100:02d}',
                   mode='historical_replay_not_live', overall=overall, leagues=metrics,
                   markets=market_rows, failures=failures,
                   limitation='Existing hyperparameters were tuned on seasons including 2024/25; no untouched test or profit claim.')
    (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    table = pd.DataFrame(metrics)[['league_name','matches','correct','accuracy','always_home_accuracy','log_loss','brier']]
    sample = frame[frame.league=='E0'].head(10).copy()
    if sample.empty:
        sample=frame.head(10).copy()
    sample['match']=sample.home+' – '+sample.away
    sample['score']=sample.actual_home_goals.astype(str)+'–'+sample.actual_away_goals.astype(str)
    samples=sample[['date','match','1x2:home','1x2:draw','1x2:away','predicted_result','score','correct']]
    lines=[f'# Football {summary["season"]} historical replay','',
           'Generated by `football_replay.py`. This is a replay, not live or timestamp-verified forward predictions.', '',
           'The existing country-specific Dixon–Coles model was refit for each match day using only earlier match dates. '
           '365-day half-life for result/double chance; 730 days for goal markets; history window 3,650 days. '
           'Actual target scores were used only for scoring forecasts, never as model inputs. '
           'Existing settings were selected on seasons including 2024/25, so this is not an untouched out-of-sample test of model selection.', '',
           f'Predicted {len(frame):,} of {len(targets):,} matches. Correct 1X2 picks: {int(frame.correct.sum()):,} '
           f'({overall["accuracy"]:.1%}); always predicting a home win: {overall["always_home_accuracy"]:.1%}.', '',
           table.to_markdown(index=False, floatfmt='.4f'), '',
           '## Markets', '',pd.DataFrame(market_rows).to_markdown(index=False,floatfmt='.4f'), '',
           'Market accuracy means selecting the highest-probability outcome. Different markets have different base rates; '
           'high totals accuracy is not evidence of a profitable betting strategy.', '',
           '## First ten Premier League matches (chronological, not cherry-picked)', '',
           samples.to_markdown(index=False,floatfmt='.3f'), '',
           '## Limitations', '',
           'No bookmaker odds or historical receipt timestamps were fabricated. No accumulators or ROI are reported. '
           'No replay records were inserted into the live forecast/paper database. Existing evaluation reports remain unchanged. '
           f'Withheld/missing forecasts: {len(targets)-len(frame)}. Failure details are in `summary.json`; all individual predictions are in `predictions.csv`.', '']
    (folder/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary['overall'],indent=2), flush=True)
    print(f'Wrote {folder}/report.md, predictions.csv, summary.json', flush=True)


if __name__=='__main__':
    main()
