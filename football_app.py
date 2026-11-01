"""Local interactive football training and prediction website.
Run: .venv/bin/streamlit run football_app.py
"""
from datetime import date,timedelta
import json
from pathlib import Path
import sqlite3
from src.football.db import DB_PATH
import pandas as pd
import streamlit as st
from src.football.app_jobs import list_jobs,submit,ROOT
from src.football.data import LEAGUES
from src.football.app_results import overlay,payout

st.set_page_config(page_title='Football predictions',layout='wide')
st.title('Football predictions')
st.caption('Train the existing models, forecast a date range, and explore probabilities and expected goals.')
st.info('Development models: no statistically credible profitable edge has been established. Historical price timestamps are unverified. Model expected goals are not observed shot-based xG.')

@st.cache_data(ttl=60)
def historical_coverage():
    with sqlite3.connect(f'file:{DB_PATH.resolve()}?mode=ro',uri=True) as conn:
        return conn.execute('SELECT min(match_date), max(match_date) FROM matches WHERE fthg IS NOT NULL AND ftag IS NOT NULL').fetchone()

coverage=historical_coverage()
jobs=list_jobs()
trained=[j for j in jobs if j['kind']=='train' and j['status']=='completed']
train_tab,predict_tab,explore_tab=st.tabs(['Train','Predict','Explore results'],default='Explore results')
with train_tab:
    st.subheader('Train on historical teams, results and odds')
    st.write('Existing LightGBM supplies match results and double chance. Dixon–Coles supplies totals, BTTS and expected goals. Training uses completed earlier seasons, with a separate later calibration season.')
    years=st.selectbox('Training window (seasons, excluding calibration)',[3,5,8],index=1)
    compare=st.checkbox('Also compare 3, 5 and 8 seasons on 2024/25 and 2025/26',value=True)
    st.caption('Early eight-season folds can contain fewer available seasons; the comparison reports actual coverage. These repeatedly inspected seasons are development data.')
    if st.button('Train model',type='primary'):
        try:
            identifier=submit('train',dict(years=years,compare=compare));st.success('Training started: '+identifier[:8]);st.rerun()
        except ValueError as exc:st.error(str(exc))
    for j in trained[:1]:
        st.success('Latest trained model: '+j['id'][:8])
        st.json(j['result'].get('metadata',{}),expanded=False)
        comparison=Path(j['result']['output'])/'comparison.csv'
        if comparison.exists():st.dataframe(pd.read_csv(comparison),hide_index=True,width='stretch')
with predict_tab:
    st.subheader('Generate predictions')
    col1,col2=st.columns(2)
    start=col1.date_input('Start date',value=date.today())
    end=col2.date_input('End date',value=date.today()+timedelta(days=4))
    model_ids=[j['id'] for j in trained]
    selected=st.selectbox('Trained model for future forecasts',model_ids,index=0 if model_ids else None,
        format_func=lambda v:next(f"{j['id'][:8]} · {j['config'].get('years',5)} seasons · {j['created'][:16]}" for j in trained if j['id']==v))
    replay_years=st.selectbox('Training window for historical replay',[3,5,8],index=1)
    st.caption(f'Historical database coverage: {coverage[0]} to {coverage[1]}; the saved 28 October replay is available under Explore results. Future: today plus four days, real quotes within the shared provider budget. Historical: up to 32 days per job; models are refitted on earlier seasons. Predictions are frozen before results are joined. Future runs save paper records and settle recorded completed matches; each run has a separate experimental ledger.')
    if st.button('Generate predictions',type='primary'):
        try:
            identifier=submit('predict',dict(start=start.isoformat(),end=end.isoformat(),years=replay_years,model_job=selected));st.success('Prediction job started: '+identifier[:8]);st.rerun()
        except ValueError as exc:st.error(str(exc))


def result_sources(current_jobs):
    sources={}
    for j in current_jobs:
        if j['kind']=='predict' and j['status']=='completed':
            path=Path(j['result']['output'])/'predictions.csv'
            if path.exists():sources[f"{j['config']['start']} to {j['config']['end']} · {j['id'][:8]}"]=path
    saved=ROOT/'reports/football_probability_forecast_20261029'
    if (saved/'all_forecasts.csv').exists():sources['Saved forward snapshot · 29 October 2026']=saved/'all_forecasts.csv'
    replay=ROOT/'reports/football_blind_replay_20261028/all_predictions_with_actuals.csv'
    if replay.exists():sources['Saved blind replay · 28 October 2026']=replay
    return sources


def read_predictions(path):
    frame=pd.read_csv(path)
    if 'mode' in frame:
        frame=frame[frame['mode']=='strict_live_freshness'].copy()
    if 'probability' not in frame:
        columns=[c for c in frame if ':' in c]
        identifiers=[c for c in frame if c not in columns]
        frame=frame.melt(id_vars=identifiers,value_vars=columns,var_name='key',value_name='probability')
        frame[['market','selection']]=frame['key'].str.split(':',n=1,expand=True)
        folder=path.parent
        prices=folder/'ranked_probability_selections.csv'
        if prices.exists():
            quotes=pd.read_csv(prices)
            from football_odds_report import LABELS
            reverse={v:k for k,v in LABELS.items()}
            quotes['key']=quotes.selection.map(reverse)
            frame=frame.merge(quotes[['provider_id','key','odds','bookmaker']],on=['provider_id','key'],how='left')
        goals=folder/'model_expected_goals.csv'
        if goals.exists():
            g=pd.read_csv(goals);frame['match']=frame.home+' – '+frame.away
            frame=frame.merge(g.drop(columns=['kickoff_utc'],errors='ignore'),on='match',how='left')
    if 'league' not in frame:
        # Saved replay IDs encode the canonical league, e.g. fd:I1:2026:...
        frame['league']=frame.provider_id.str.split(':').str[1]
    if 'date' not in frame:
        frame['date']=pd.to_datetime(frame.kickoff_utc,utc=True).dt.tz_convert('Europe/London').dt.date.astype(str)
    if 'odds' not in frame:frame['odds']=frame.get('archived_odds',float('nan'))
    return overlay(frame,path.parent)

with explore_tab:
    st.subheader('Predicted games')
    st.caption('Every completed prediction run is saved here. Return to the same run and click Check results; no retraining or new prediction search is needed.')
    sources=result_sources(jobs)
    if sources:
        source=st.selectbox('Result set',list(sources))
        frame=read_predictions(sources[source])
        saved_job=next((j for j in jobs if j['kind']=='predict' and j['status']=='completed' and Path(j['result']['output']).resolve()==sources[source].parent.resolve()),None)
        if saved_job:
            if st.button('Check results',type='primary'):
                try:
                    submit('check_results',{'prediction_job':saved_job['id']});st.rerun()
                except ValueError as exc:st.error(str(exc))
            checks=[j for j in jobs if j['kind']=='check_results' and j['config'].get('prediction_job')==saved_job['id']]
            if checks:
                latest=checks[0]
                if latest['status']=='completed':st.success(f"Last checked: {latest['result'].get('checked_at','')} · {latest['result'].get('finished_matches',0)} finished matches. Unavailable results remain pending.")
                elif latest['status']=='failed':st.error(latest['message'])
                else:st.info('Checking match results… refresh when complete, or wait for the job to finish.')
        else:
            st.caption('This is a preserved legacy report. Use a saved prediction run above for Check results.')
        status_file=sources[source].parent/'status.json'
        if status_file.exists():
            status=json.loads(status_file.read_text());st.write('Paper decision: '+status.get('decision','Unknown'))
            if status.get('candidates')==0:st.warning('No fresh eligible bookmaker offers in this run. Forecasts are available; prices and model sources must be checked before using them.')
            with st.expander('Source coverage and API budget'):st.json(status)
        st.caption('Saved snapshots retain their original times. Prices may now be stale. Overlapping markets on one match are alternatives, not independent opportunities.')
        a,b,c=st.columns(3)
        leagues=a.multiselect('Leagues',sorted(frame.league.dropna().unique()),default=sorted(frame.league.dropna().unique()),format_func=lambda x:f'{x} · {LEAGUES[x][0]}' if x in LEAGUES else x)
        families=b.multiselect('Markets',sorted(frame.market.unique()),default=sorted(frame.market.unique()))
        probability=c.slider('Selection probability (%)',0,100,(60,100))
        dates=sorted(frame.date.unique());chosen_dates=st.multiselect('Prediction dates',dates,default=dates)
        frame=frame[frame.league.isin(leagues)&frame.market.isin(families)&frame.date.isin(chosen_dates)&frame.probability.between(probability[0]/100,probability[1]/100)].copy()
        a,b,c=st.columns(3)
        xg=a.slider('Model expected total goals',0.0,10.0,(0.0,10.0),step=.1)
        keep_missing=b.checkbox('Include fixtures without expected-goals diagnostics',value=True)
        priced=c.checkbox('Only selections with bookmaker prices',value=False)
        if 'expected_total_goals' in frame:
            mask=frame.expected_total_goals.between(*xg)
            if keep_missing:mask|=frame.expected_total_goals.isna()
            frame=frame[mask]
        elif not keep_missing:frame=frame.iloc[:0]
        odds=st.slider('Decimal odds range',1.0,30.0,(1.0,30.0),step=.05)
        frame=frame[frame.odds.between(*odds)|((not priced)&frame.odds.isna())]
        best=st.checkbox('Highest-probability selection per match',value=True)
        frame=frame.sort_values('probability',ascending=False,kind='stable')
        if best:frame=frame.drop_duplicates('provider_id')
        st.metric('Matching selections',len(frame))
        frame['probability_percent']=frame.probability*100
        cols=[c for c in ['date','league','home','away','market','selection','probability_percent','odds','bookmaker','expected_home_goals','expected_away_goals','expected_total_goals','actual_score','selection_won','match_status','results_checked_at','as_of','odds_captured_at_utc','price_status','conservative_probability','model_source'] if c in frame]
        display=frame[cols].copy().reset_index(drop=True)
        display.insert(0,'Select',False)
        edited=st.data_editor(display,hide_index=True,width='stretch',disabled=cols,
            column_config={'Select':st.column_config.CheckboxColumn('Select',default=False),
                'probability_percent':st.column_config.NumberColumn('Probability (%)',format='%.1f')},
            key='selected_games_'+str(sources[source]))
        st.subheader('Potential payout calculator')
        st.caption('Tick a selection above and enter a stake. This is a hypothetical accumulator calculation: every selected leg must win. Displayed prices may be historical, stale or from different bookmakers; no bet is placed and availability is not guaranteed.')
        stake=st.number_input('Stake (£)',min_value=0.0,max_value=1000000.0,value=50.0,step=5.0,key='calculator_stake')
        selected_rows=frame.reset_index(drop=True).loc[edited['Select'].to_numpy(dtype=bool)].to_dict('records')
        try:
            calculation=payout(selected_rows,stake)
            a,b,c=st.columns(3)
            a.metric('Selected games',calculation['selections'])
            b.metric('Combined decimal odds',f"{calculation['combined_odds']:.3f}")
            c.metric('Potential return (includes stake)',f"£{calculation['payout']:,.2f}")
            if selected_rows:st.write(f"Potential profit if all win: £{calculation['profit']:,.2f}. If any leg loses, accumulator return is £0. Fees, bonuses and voids are not included.")
        except (ValueError,ArithmeticError) as exc:st.warning(str(exc))
        st.caption('Checkboxes apply to the currently displayed rows. Changing filters or result sets may reset them.')
        st.download_button('Download filtered CSV',frame.to_csv(index=False),'football_predictions.csv','text/csv')
        selected_path=sources[source].parent
        tickets=selected_path/'tickets.json'
        if tickets.exists():
            settlement=selected_path/'settlement_snapshot.json'
            payload=json.loads(settlement.read_text())['tickets'] if settlement.exists() else json.loads(tickets.read_text())
            st.subheader('Accumulator tickets')
            st.write('NO BET' if not payload else 'Tickets passing the existing uncertainty and 4.75–5.25 odds rules')
            st.json(payload,expanded=False)
        st.caption('Ticket joint probability differs from per-selection probability. The optimiser applies a dependence haircut and conservative uncertainty bounds; high individual probability alone does not qualify a ticket.')
    else:st.write('Train a model and generate a prediction range to view results.')

def progress_panel():
    current=list_jobs()
    active=[j for j in current if j['status'] in ('queued','running')]
    with st.expander('Jobs — '+(f'{len(active)} running' if active else 'all complete'),expanded=bool(active)):
        st.caption('Completed predictions are displayed above in Explore results. Select a result set to see its matches and probabilities.')
        if st.button('Refresh results'):st.rerun()
        if not current:st.caption('No jobs yet.');return
        for j in current[:8]:
            st.write(f"{j['kind'].title()} · {j['id'][:8]} · {j['status']} — {j['message']}")
        if any(j['status']=='completed' and j['id'] not in {r['id'] for r in jobs if r['status']=='completed'} for j in current):
            st.rerun()

# Poll only while work is active; idle pages retain the full results view.
if any(j['status'] in ('queued','running') for j in jobs):
    st.fragment(run_every=3)(progress_panel)()
else:
    progress_panel()
