"""Date-range jobs reusing the established forecast and settlement components."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from . import historical, markets
from .app_training import inputs, fit
from .db import DB_PATH
from .workflow import predict_fixtures, load_history, merge_forward_history, run_cycle


def historical_range(start,end,years,output,progress=print):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    df=historical.load(DB_PATH,last_season=2100)
    targets=df[df.date.between(pd.Timestamp(start),pd.Timestamp(end))].copy()
    if targets.empty:raise ValueError('No historical fixtures in the database for this range. Use the saved replay for 28 October, or choose a covered date.')
    all_d,X=inputs(DB_PATH); heads={};pmap={}
    quotes=historical.load_odds(DB_PATH,targets.match_id)
    quotes=quotes[quotes.bookmaker=='market_avg']
    quote_map={(r.match_id,r.market,r.selection):r for r in quotes.itertuples()}
    for season in sorted(targets.season.unique()):
        progress(f'Fitting earlier-season LightGBM for historical season {season}')
        model,meta=fit(X,all_d,int(season),years)
        mask=all_d.season.eq(season)&all_d.date.between(pd.Timestamp(start),pd.Timestamp(end))
        if mask.any():
            probs=markets.from_1x2(model.predict_proba(X.loc[mask]))
            pmap.update({mid:row for mid,row in zip(all_d.loc[mask,'match_id'],probs.to_dict('records'))})
        heads[str(season)]=meta
    rows=[]
    for day,group in targets.groupby('date',sort=True):
        progress(f'Predicting {day.date()} with earlier results only')
        asof=day.tz_localize('UTC').to_pydatetime()
        history=df[df.date<day].copy()
        fixtures=[dict(provider_id=r.match_id,league=r.div,season=int(r.season),home=r.home,away=r.away,
                       kickoff_utc=(day+pd.Timedelta(hours=12)).tz_localize('UTC').isoformat()) for r in group.itertuples()]
        diagnostics={}
        probabilities,_=predict_fixtures(history,fixtures,asof,diagnostics=diagnostics)
        for f in fixtures:
            mid=f['provider_id']
            if mid not in probabilities:continue
            probabilities[mid].update(pmap.get(mid,{}))
            for key,p in probabilities[mid].items():
                market,selection=key.split(':')
                quote=quote_map.get((mid,market,selection))
                rows.append(dict(provider_id=mid,date=str(day.date()),home=f['home'],away=f['away'],league=f['league'],
                    market=market,selection=selection,probability=p,odds=quote.odds if quote is not None else np.nan,
                    bookmaker='market_avg' if quote is not None else None,price_status='Historical timestamp unverified' if quote is not None else 'No archived price',
                    as_of=asof.isoformat(),model_source='Historical odds LightGBM + Dixon-Coles' if mid in pmap else 'Dixon-Coles',
                    **diagnostics.get(mid,{})))
    if not rows:raise ValueError('No forecasts: insufficient earlier team history')
    predictions=pd.DataFrame(rows)
    predictions.to_csv(output/'predictions_before_results.csv',index=False)
    manifest=dict(mode='development_historical_replay',training=heads,
        odds_warning='Historical pre-closing odds used as inference features have unverified timestamps; no executable-price or profit claim.',
        predictions_sha256=hashlib.sha256((output/'predictions_before_results.csv').read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Join target outcomes only after persisting predictions.
    actual=targets.set_index('match_id')
    predictions['actual_score']=[f'{int(actual.loc[mid,"hg"])}–{int(actual.loc[mid,"ag"])}' for mid in predictions.provider_id]
    predictions['selection_won']=[bool(markets.won(r.market,r.selection,actual.loc[r.provider_id,'hg'],actual.loc[r.provider_id,'ag'])) for r in predictions.itertuples()]
    predictions.to_csv(output/'predictions.csv',index=False)
    return dict(output=str(output.resolve()),rows=len(predictions),mode=manifest['mode'],decision='Historical analysis; no bets',warning=manifest['odds_warning'])


def live_range(start,end,model_path,output,progress=print):
    import os
    from .forward_store import ForwardStore
    from .scoreboard_provider import ScoreboardCSV
    from .odds_api import OddsAPI,OddsAPIProvider
    from .odds_training import predictor
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc)
    artifact=joblib.load(model_path)
    if int(artifact['metadata']['calibration_season'])>=now.year-(now.month<7):
        raise ValueError('Model calibration overlaps current prediction season')
    with sqlite3.connect(f'file:{DB_PATH.resolve()}?mode=ro',uri=True) as c:
        names={r[0] for r in c.execute('SELECT name FROM teams')}
    key=os.getenv('ODDS_API_KEY')
    if not key:raise ValueError('ODDS_API_KEY is not configured in the project environment')
    store=ForwardStore(output/'paper.db')
    schedule=ScoreboardCSV(canonical_teams=names,snapshot_dir=DB_PATH.parent/'forward_sources')
    api=OddsAPI(key,canonical_teams=names,snapshot_dir=DB_PATH.parent/'odds_api/snapshots',budget_db=DB_PATH.parent/'odds_api/usage.db',cycle_budget=30)
    provider=OddsAPIProvider(api,schedule=schedule,extra_events=3)
    basefixtures=provider.fixtures
    def selected_fixtures(a,b):
        return [f for f in basefixtures(a,b) if start<=pd.Timestamp(f['kickoff_utc']).tz_convert('Europe/London').date().isoformat()<=end]
    provider.fixtures=selected_fixtures
    inference=predictor(model_path,provider);diagnostics={}
    def predict(history,fixtures,asof):
        predict_fixtures(history,fixtures,asof,diagnostics=diagnostics)
        return inference(history,fixtures,asof)
    try:
        progress('Retrieving schedules and real bookmaker prices within the shared API budget')
        result=run_cycle(store,provider,DB_PATH,predictor=predict,rank_by='probability',place_paper=True,stake=10)
        fixtures={f['provider_id']:f for f in store.fixtures()};rows=[]
        quotes=pd.read_sql_query('SELECT * FROM live_odds',store.conn)
        for snapshot in store.forecasts():
            f=fixtures[snapshot['provider_id']]
            for key,p in snapshot['probabilities'].items():
                market,selection=key.split(':');q=quotes[(quotes.provider_id==f['provider_id'])&(quotes.market==market)&(quotes.selection==selection)] if len(quotes) else quotes
                best=q.sort_values('odds',ascending=False).iloc[0] if len(q) else None
                rows.append(dict(provider_id=f['provider_id'],date=pd.Timestamp(f['kickoff_utc']).tz_convert('Europe/London').date().isoformat(),
                    league=f['league'],home=f['home'],away=f['away'],market=market,selection=selection,probability=p,
                    odds=float(best.odds) if best is not None else np.nan,as_of=snapshot['as_of'],
                    bookmaker=best.bookmaker if best is not None else None,
                    odds_captured_at_utc=best.captured_at_utc if best is not None else None,
                    price_status=('Fresh at prediction' if (pd.Timestamp(snapshot['as_of'])-pd.Timestamp(best.source_updated_at)).total_seconds()<=3600 else 'Stale at prediction') if best is not None else 'No quoted price',
                    conservative_probability=snapshot['lower'][key],
                    actual_score='Pending',selection_won=None,**diagnostics.get(f['provider_id'],{})))
        pd.DataFrame(rows,columns=['provider_id','date','league','home','away','market','selection','probability','odds','as_of','bookmaker','odds_captured_at_utc','price_status','conservative_probability','actual_score','selection_won','expected_home_goals','expected_away_goals','expected_total_goals']).to_csv(output/'predictions.csv',index=False)
        (output/'tickets.json').write_text(json.dumps(store.tickets(),indent=2,default=str))
        result['source_metadata']=getattr(provider,'latest_metadata',{})
        (output/'status.json').write_text(json.dumps(result,indent=2,default=str))
        return dict(output=str(output.resolve()),rows=len(rows),mode='forward_paper',**result)
    finally:store.close()
