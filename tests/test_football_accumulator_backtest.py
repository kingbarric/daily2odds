from datetime import datetime,timedelta,timezone
import numpy as np
import pandas as pd
import pytest
from src.football.accumulator_backtest import PastCalibration,performance,replay
from src.football.optimiser import Pick,optimise


def test_estimated_prices_require_explicit_sensitivity_mode():
    now=datetime(2024,8,16,16,tzinfo=timezone.utc)
    picks=[Pick(str(i),str(i),'Book','1x2','home',2.25,.9,.85,now,now+timedelta(days=1),timestamp_verified=False) for i in range(2)]
    assert optimise(picks,now=now)==[]
    tickets=optimise(picks,now=now,allow_estimated_timestamps=True)
    assert len(tickets)==1 and np.isclose(tickets[0].combined_odds,5.0625)


def test_financial_statistics_settlement_order_and_losing_streak():
    records=[]
    for i,win in enumerate([False,False,True,False]):
        payout=50 if win else 0
        records.append(dict(ticket_id=str(i),stake=10,payout=payout,net_profit=payout-10,
                            outcome='won' if win else 'lost',decision_utc=f'2024-08-{i+1:02d}',
                            settled_at_utc=f'2024-08-{i+2:02d}'))
    p=performance(records)
    assert p['stakes']==40 and p['payouts']==50 and p['net_profit']==10
    assert p['maximum_losing_streak']==2 and p['maximum_drawdown']==20
    assert p['roi']==.25 and p['hit_rate']==.25
    assert performance([])['roi'] is None


def history():
    rows=[];prices=[]
    # Three earlier decision groups with results known by the target decision.
    for i,date in enumerate(['2023-08-05','2023-08-12','2023-08-19','2024-08-17','2024-08-17']):
        d=pd.Timestamp(date);decision=d-pd.Timedelta(days=1)
        rows.append(dict(match_id=str(i),div='E0',season=2023 if i<3 else 2024,
                         date=d,decision=decision,estimated_schedule_utc=decision+pd.Timedelta(hours=16),
                         kickoff_utc=d+pd.Timedelta(hours=14),home='A'+str(i),away='B'+str(i),hg=1,ag=0))
        prices.append(dict(match_id=str(i),bookmaker='bet365',market='1x2',selection='home',odds=2.25,
                           timestamp_verified=False,captured_at_utc=pd.NaT))
    return pd.DataFrame(rows),pd.DataFrame({'1x2:home':[.9]*5}),pd.DataFrame(prices)


def test_replay_records_stake_outcomes_and_no_verified_receipt_for_sensitivity():
    df,p,o=history()
    # Enough earlier events for this controlled example; tiny min n is not production policy.
    for i in range(3):
        df.loc[i,'hg']=1
    r=replay(df,p,o,model_name='test',mode='estimated_sensitivity',minimum_samples=1,min_robust_ev=0)
    # Wilson with only three wins can still be too uncertain: never force a ticket.
    assert all(t['stake']==10 for t in r['tickets'])
    assert all(l['captured_at_utc'] is None and not l['price_timestamp_verified'] for t in r['tickets'] for l in t['legs'])
    strict=replay(df,p,o,model_name='test',mode='verified',minimum_samples=1)
    assert not strict['tickets'] and strict['summaries'][0]['stakes']==0


def test_future_results_cannot_change_prior_ticket_selections():
    df,p,o=history()
    # Replicate genuinely distinct earlier matches to establish conservative bins.
    old=[]
    for i in range(150):
        r=df.iloc[0].copy();r['match_id']='old'+str(i);old.append(r.to_dict())
    before=pd.DataFrame(old+df.iloc[3:].to_dict('records')).reset_index(drop=True)
    probs=pd.DataFrame({'1x2:home':[.9]*len(before)})
    base=replay(before,probs,o,model_name='test',mode='estimated_sensitivity')
    changed=before.copy();changed.loc[changed.season==2024,['hg','ag']]=[0,9]
    after=replay(changed,probs,o,model_name='test',mode='estimated_sensitivity')
    assert len(base['tickets'])==len(after['tickets'])==1
    assert base['tickets'][0]['ticket_id']==after['tickets'][0]['ticket_id']
    assert base['tickets'][0]['outcome']=='won' and after['tickets'][0]['outcome']=='lost'
    assert base['tickets'][0]['stake']==10 and base['tickets'][0]['payout']==pytest.approx(50.63)
    assert base['tickets'][0]['legs'][0]['calibration_n']==150


def test_calibration_is_league_specific_and_requires_minimum_samples():
    c=PastCalibration(100)
    for _ in range(100):c.observe('E0','btts:yes',.7,True)
    assert c.lower('E0','btts:yes',.7)==(.7,100)
    assert c.lower('E1','btts:yes',.7)==(0,0)


def test_simultaneous_settlement_does_not_create_artificial_drawdown():
    records=[dict(ticket_id='a',stake=10,payout=0,net_profit=-10,outcome='lost',decision_utc='2024-08-01',settled_at_utc='2024-08-02'),
             dict(ticket_id='b',stake=10,payout=30,net_profit=20,outcome='won',decision_utc='2024-08-01',settled_at_utc='2024-08-02')]
    assert performance(records)['maximum_drawdown']==0
