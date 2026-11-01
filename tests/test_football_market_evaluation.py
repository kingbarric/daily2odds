import numpy as np
import pandas as pd
from src.football.market_evaluation import fair_probabilities,selection_metrics,evaluate_markets
from src.football import markets


def data():
    frame=pd.DataFrame(dict(match_id=['a','b'],season=[2024,2024],div=['E0','E0'],
                            hg=[2,0],ag=[0,0],decision=pd.to_datetime(['2024-08-09','2024-08-16']),
                            estimated_schedule_utc=pd.to_datetime(['2024-08-09 16:00','2024-08-16 16:00'])))
    rows=[]
    for match in ['a','b']:
        for book in ['bet365','market_avg']:
            for selection,price in [('home',2),('draw',3.5),('away',4)]:
                rows.append(dict(match_id=match,bookmaker=book,market='1x2',selection=selection,odds=price,
                                 timestamp_verified=False,captured_at_utc=pd.NaT))
    odds=pd.DataFrame(rows)
    p=markets.from_1x2(np.array([[.6,.2,.2],[.3,.4,.3]]))
    return frame,odds,{'Elo':p}


def test_fair_probabilities_remove_margin_and_double_chance_is_event_only():
    _,o,_=data();q=fair_probabilities(o)
    one=q[(q.match_id=='a')&(q.bookmaker=='bet365')]
    assert np.isclose(one[one.market=='1x2'].fair_probability.sum(),1)
    home=one[(one.market=='1x2')&(one.selection=='home')].fair_probability.iloc[0]
    draw=one[(one.market=='1x2')&(one.selection=='draw')].fair_probability.iloc[0]
    hd=one[(one.market=='double_chance')&(one.selection=='home_draw')].fair_probability.iloc[0]
    assert hd==home+draw
    assert 'odds' not in q


def test_binary_metrics_and_unsupported_markets_are_explicit():
    x=selection_metrics([.8,.3],[1,0])
    assert x['accuracy']==1 and np.isclose(x['brier'],.065)
    df,odds,models=data();result=evaluate_markets(df,models,odds)
    m=result['metrics']
    assert (m[m.market=='btts'].status=='unsupported_model_market').all()
    assert (m[m.market=='double_chance'].predictions==2).all()
    singles=result['singles']
    assert singles[singles.timing=='verified'].bets.sum()==0
    assert singles[singles.market=='double_chance'].available_odds.sum()==0
    assert singles[(singles.timing=='estimated_sensitivity')&(singles.selection=='home')].bets.sum()>0
    assert len(result['calibration'])>0


def test_incomplete_or_invalid_prices_are_not_a_bookmaker_baseline():
    _,odds,_=data()
    assert fair_probabilities(odds[odds.selection!='draw']).empty
    odds.loc[0,'odds']=np.nan
    result=fair_probabilities(odds)
    assert result[(result.match_id=='a')&(result.bookmaker=='bet365')].empty
