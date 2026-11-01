"""Retrospective result-blind replay: save predictions before opening the result file.

Uses a real archived pre-kickoff publisher snapshot. Exact live freshness rules
and an explicitly stale-price inference sensitivity are both shown. This is not
an independent unseen experiment: the games have already been discussed.
"""
import argparse
from datetime import datetime,timedelta,timezone
from email.utils import format_datetime
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import joblib
import pandas as pd

from src.football import markets
from src.football.db import DB_PATH
from src.football.public_provider import PublicCSV
from src.football.odds_training import predictor
from src.football.workflow import instant,load_history
from football_odds_report import LABELS


def blind_history(history,asof):
    cutoff=pd.Timestamp(instant(asof)).tz_localize(None).normalize()
    return history[(history.date<cutoff)&history.hg.notna()&history.ag.notna()].copy().reset_index(drop=True)


def archived_inputs(csv_file,metadata_file,asof,day,canonical):
    metadata=json.loads(Path(metadata_file).read_text());captured=instant(metadata['captured_at_utc'])
    if captured>instant(asof):raise ValueError('Archive was not available before simulated prediction time')
    publication=instant(metadata['batch_published_at_utc']) if metadata.get('batch_published_at_utc') else None
    raw=Path(csv_file).read_bytes()
    class Response:
        content=raw
        headers={'Last-Modified':format_datetime(publication,usegmt=True)} if publication else {}
        def raise_for_status(self):pass
    class Session:
        def get(self,*args,**kwargs):return Response()
    provider=PublicCSV(session=Session(),clock=lambda:captured,canonical_teams=canonical)
    fixtures=provider.fixtures(day,day)
    if not fixtures or any(instant(f['kickoff_utc'])<=instant(asof) for f in fixtures):
        raise ValueError('Need archived fixtures strictly after prediction time')
    # Restrict to fixture/quote fields. Actual scores are never passed as inputs.
    for fixture in fixtures:fixture['hg']=fixture['ag']=None
    return provider,fixtures,metadata


def predict_blinded(history,fixtures,provider,model_path,asof):
    history=blind_history(history,asof)
    meta=joblib.load(model_path)['metadata']
    target_season=min(int(f['season']) for f in fixtures)
    if max(meta['training_seasons'])>=target_season or int(meta['calibration_season'])>=target_season:
        raise ValueError('Model training/calibration overlaps target season; blind replay refused')
    records=[]
    for mode,max_age in [('strict_live_freshness',timedelta(hours=1)),('archived_known_price_sensitivity',None)]:
        p,version=predictor(model_path,provider,max_source_age=max_age)(history,fixtures,asof)
        for fixture in fixtures:
            identifier=fixture['provider_id']
            if identifier not in p:continue
            quotes=provider.odds(identifier)
            for key,value in p[identifier].items():
                market,selection=key.split(':')
                price_rows=[q for q in quotes if q['market']==market and q['selection']==selection]
                best=max(price_rows,key=lambda q:q['odds']) if price_rows else None
                records.append(dict(mode=mode,provider_id=identifier,home=fixture['home'],away=fixture['away'],
                    kickoff_utc=fixture['kickoff_utc'],simulated_as_of_utc=instant(asof).isoformat(),
                    model_source=provider.prediction_sources[identifier],model_version=version,
                    market=market,selection=selection,probability=float(value),
                    archived_odds=best['odds'] if best else None,bookmaker=best['bookmaker'] if best else None,
                    actual_archive_receipt_utc=best['captured_at_utc'] if best else None,
                    publisher_batch_time_utc=best['source_updated_at'] if best else None,
                    price_fresh_for_live_policy=(instant(asof)-instant(best['source_updated_at'])<=timedelta(hours=1)) if best else False))
    return pd.DataFrame(records)


def append_actuals(predictions,result_file):
    # Called only after prediction file is written and its hash recorded.
    payload=json.loads(Path(result_file).read_text())
    by_id={r['provider_id']:r for r in payload['results'] if r['status']=='finished'}
    rows=[]
    for row in predictions.to_dict('records'):
        actual=by_id.get(row['provider_id'])
        if not actual:
            rows.append(dict(row,actual_score='Unavailable',selection_won=None));continue
        win=bool(markets.won(row['market'],row['selection'],actual['hg'],actual['ag']))
        rows.append(dict(row,actual_score=f"{actual['hg']}–{actual['ag']}",selection_won=win))
    return pd.DataFrame(rows),payload['checked_at_utc']


def best_per_market(predictions):
    # Select before seeing actual outcomes. One top event per match/family.
    return (predictions.sort_values(['probability','selection'],ascending=[False,True],kind='stable')
            .drop_duplicates(['mode','provider_id','market']))


def run(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    frozen=output/'predictions_before_results.csv'
    if frozen.exists():raise ValueError('Prediction snapshot exists: choose a new output directory')
    day=pd.Timestamp(args.day).date();asof=datetime.combine(day,datetime.min.time(),tzinfo=timezone.utc)+timedelta(minutes=18)
    history,_=load_history(args.db,asof)
    history=blind_history(history,asof)
    provider,fixtures,source=archived_inputs(args.archive,args.archive_metadata,asof,day,set(history.home)|set(history.away))
    predictions=predict_blinded(history,fixtures,provider,args.model,asof)
    # Freeze both variants AND the deterministic selections before any actuals.
    predictions.to_csv(frozen,index=False)
    choices=best_per_market(predictions);choices.to_csv(output/'selections_before_results.csv',index=False)
    top_choices=choices.sort_values('probability',ascending=False,kind='stable').drop_duplicates(['mode','provider_id'])
    top_choices.to_csv(output/'top_selections_before_results.csv',index=False)
    digest=sha256(frozen.read_bytes()).hexdigest()
    manifest=dict(mode='retrospective_result_blind_not_independent_validation',
        simulated_as_of_utc=asof.isoformat(),generated_at_utc=datetime.now(timezone.utc).isoformat(),
        archive=source,model_training_metadata=joblib.load(args.model)['metadata'],
        training_history_max_date=str(history.date.max()),predictions_sha256=digest,
        predictions_saved_before_result_file_read=True,
        price_warning='Publisher batch was already older than 60 minutes. No verified-current prices for trading; missing goal/DC/BTTS prices are not invented.',
        strict_policy='Stale prices cannot enter live odds head; Dixon-Coles fallback',
        sensitivity_policy='Known pre-kickoff stale prices allowed as inference features only, not a bettable-price claim')
    (output/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Outcome access starts HERE, after the predictions and choices are saved.
    joined,checked=append_actuals(predictions,args.results)
    selected,_=append_actuals(choices,args.results)
    top_scored,_=append_actuals(top_choices,args.results)
    assert sha256(frozen.read_bytes()).hexdigest()==digest
    joined.to_csv(output/'all_predictions_with_actuals.csv',index=False)
    selected.to_csv(output/'selected_markets_with_actuals.csv',index=False)
    summary=selected.groupby(['mode','market']).agg(matches=('provider_id','size'),correct=('selection_won','sum')).reset_index()
    summary['hit_rate']=summary.correct/summary.matches
    summary.to_csv(output/'summary.csv',index=False)
    def present(frame):
        frame=frame.copy();frame['match']=frame.home+' – '+frame.away
        frame['prediction']=[LABELS[r.market+':'+r.selection] for r in frame.itertuples()]
        frame['estimated_probability']=frame.probability.map(lambda x:f'{x:.1%}')
        frame['correct']=frame.selection_won.map({True:'Yes',False:'No'})
        return frame[['match','prediction','estimated_probability','archived_odds','actual_score','correct']].rename(columns={'archived_odds':'archived price (stale; not bettable)'})
    parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>28 October: blinded replay vs actual results</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:28px auto;padding:0 16px;color:#222}section{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;min-width:650px}td,th{border:1px solid #ccc;padding:7px;text-align:left}th{background:#eee}</style></head><body>',
        '<h1>28 October 2026: result-blind replay</h1>',
        '<p>Results from 28 October and later are excluded from model inputs. Only an archived publisher snapshot actually captured before kickoff is used. Predictions and highest-probability selections were saved and hashed before this script opened the actual-results file.</p>',
        '<p><strong>This is a retrospective replay, not an independent unseen test.</strong> We already know these results. The model artifact was built later, but its training/calibration labels end in 2025/26, before these games. No parameters were retuned on these five outcomes.</p>',
        '<p>The archived quotes were already stale under the live 60-minute rule. The strict replay therefore uses the existing goal model as fallback. A separately labelled sensitivity uses those known older quotes as odds-head inputs; neither is presented as a qualified bet. No prices are fabricated for unpriced markets.</p>']
    for mode,title in [('strict_live_freshness','Strict live-rule replay'),('archived_known_price_sensitivity','Odds-aware archived-price sensitivity')]:
        sample=selected[selected['mode']==mode]
        parts.append('<section><h2>'+title+'</h2>')
        totals=summary[summary['mode']==mode][['market','matches','correct','hit_rate']].copy();totals['hit_rate']=totals.hit_rate.map(lambda v:f'{v:.0%}')
        parts.append(totals.to_html(index=False,border=0))
        for market in markets.MARKETS:
            parts.append('<h3>'+escape(market)+'</h3>'+present(sample[sample.market==market]).to_html(index=False,border=0,na_rep='Not archived',escape=True))
        # One highest-probability event per match, selected without outcomes.
        top=top_scored[top_scored['mode']==mode]
        parts.append('<h3>Highest probability per match, any market</h3>'+present(top).to_html(index=False,border=0,na_rep='Not archived',escape=True))
        parts.append('</section>')
    parts.extend(['<p>Same-match markets overlap: these are five matches, not thirty independent betting opportunities. No ROI or profitable-strategy claim is made.</p>',
        '<p><a href="selected_markets_with_actuals.csv">Selected markets and actual scores</a> · <a href="all_predictions_with_actuals.csv">Every selection</a> · <a href="predictions_before_results.csv">Frozen replay predictions</a> · <a href="prediction_manifest.json">Timing and blinding audit</a></p></body></html>'])
    (output/'report.html').write_text(''.join(parts))
    print(summary.to_string(index=False));print('Report:',(output/'report.html').resolve())
    print(present(selected[(selected['mode']=='archived_known_price_sensitivity')&(selected.market=='1x2')]).to_string(index=False))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--day',default='2026-10-28')
    p.add_argument('--db',type=Path,default=DB_PATH)
    p.add_argument('--archive',type=Path,default=Path('data/football/forward_sources/20261028T000841912562Z-a267ea27.csv'))
    p.add_argument('--archive-metadata',type=Path,default=Path('data/football/forward_sources/20261028T000841912562Z-a267ea27.json'))
    p.add_argument('--model',type=Path,default=Path('reports/football_odds_training_20261029/lightgbm_with_odds.joblib'))
    p.add_argument('--results',type=Path,default=Path('reports/football_development_2024_2025_integrity_v3/forward_result_check_20261028.json'))
    p.add_argument('--output',type=Path,default=Path('reports/football_blind_replay_20261028'))
    run(p.parse_args())
