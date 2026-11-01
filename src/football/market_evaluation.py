"""Selection-level diagnostics. Overlapping double-chance events are binary events,
not a mutually exclusive 3-class distribution. No invented bookmaker prices.
"""
import numpy as np
import pandas as pd

from . import markets, evaluate


def fair_probabilities(odds):
    """Remove overround only from a complete, coherent exclusive price set.

    Double-chance fair event probabilities can be derived from de-vigged 1X2,
    but are NOT obtainable double-chance prices and must never be used as odds.
    """
    columns=['match_id','bookmaker','market','selection','fair_probability']
    if odds.empty:
        return pd.DataFrame(columns=columns)
    o=odds[odds.market.isin(markets.EXCLUSIVE)].copy()
    keys=['match_id','bookmaker','market']
    valid=np.isfinite(o.odds)&(o.odds>1)
    # An invalid member invalidates the complete market; do not drop it first.
    allowed=o.market.str.cat(o.selection,sep=':').isin([f'{m}:{x}' for m in markets.EXCLUSIVE for x in markets.MARKETS[m]])
    o['_valid']=valid&allowed
    group=o.groupby(keys,sort=False)
    required=o.market.map(lambda m:len(markets.MARKETS[m]))
    complete=(group.selection.transform('size')==required)&(group.selection.transform('nunique')==required)&group._valid.transform('all')
    o=o[complete&allowed].copy()
    if o.empty:
        return pd.DataFrame(columns=columns)
    o['_inverse']=1/o.odds
    o['fair_probability']=o._inverse/o.groupby(keys,sort=False)._inverse.transform('sum')
    result=o[columns]
    one=o[o.market=='1x2'].pivot(index=['match_id','bookmaker'],columns='selection',values='fair_probability')
    if len(one):
        dc=pd.DataFrame({'home_draw':one.home+one['draw'],'home_away':one.home+one.away,'draw_away':one['draw']+one.away})
        dc=dc.rename_axis(columns='selection').reset_index().melt(id_vars=['match_id','bookmaker'],value_name='fair_probability')
        dc['market']='double_chance'
        result=pd.concat([result,dc[columns]],ignore_index=True)
    return result.reset_index(drop=True)


def selection_metrics(probability,observed):
    p=np.asarray(probability,float);y=np.asarray(observed,int)
    ok=np.isfinite(p);p=p[ok];y=y[ok]
    if not len(p):
        return dict(predictions=0,accuracy=None,brier=None,log_loss=None)
    q=np.column_stack([p,1-p]);target=1-y
    return dict(predictions=len(p),accuracy=float(((p>=.5)==y.astype(bool)).mean()),
                brier=float(((p-y)**2).mean()),log_loss=float(evaluate.match_log_loss(q,target).mean()))


def evaluate_markets(df, models, odds, *, stake=10):
    fair=fair_probabilities(odds)
    # Market-average probabilities are a descriptive benchmark, not a bettable bookmaker.
    benchmark=fair[fair.bookmaker=='market_avg'].pivot(index='match_id',columns=['market','selection'],values='fair_probability') if len(fair) else pd.DataFrame()
    book=odds[odds.bookmaker=='bet365']
    prices={(m,s):g.set_index('match_id').odds for (m,s),g in book.groupby(['market','selection'])}
    verified={(m,s):g.set_index('match_id').timestamp_verified for (m,s),g in book.groupby(['market','selection'])}
    metrics=[];calibration=[];singles=[];coverage=[]
    for (season,league),g in df[df.season.isin([2024,2025])].groupby(['season','div'],sort=True):
        for market,selections in markets.MARKETS.items():
            for selection in selections:
                key=f'{market}:{selection}';y=markets.won(market,selection,g.hg,g.ag).astype(int)
                b=benchmark[(market,selection)].reindex(g.match_id).to_numpy() if (market,selection) in benchmark else np.full(len(g),np.nan)
                o=prices.get((market,selection),pd.Series(dtype=float)).reindex(g.match_id).to_numpy(float)
                v=verified.get((market,selection),pd.Series(dtype=bool)).reindex(g.match_id).fillna(False).to_numpy(bool)
                known=g.estimated_schedule_utc.notna().to_numpy()
                if np.isfinite(b).any():
                    baseline_cal=evaluate.calibration(b,y)
                    for bucket,r in baseline_cal.iterrows():
                        calibration.append(dict(season=int(season),league=league,model='Bookmaker (margin removed)',
                                                market=market,selection=selection,bin=bucket,n=int(r.n),
                                                predicted=float(r.predicted),observed=float(r.observed)))
                for name,frame in models.items():
                    p=frame.loc[g.index,key].to_numpy() if key in frame else np.full(len(g),np.nan)
                    row=dict(season=int(season),league=league,model=name,market=market,selection=selection,
                             status='supported' if key in frame else 'unsupported_model_market',**selection_metrics(p,y))
                    matched=np.isfinite(p)&np.isfinite(b)
                    bm=selection_metrics(b[matched],y[matched]);mm=selection_metrics(p[matched],y[matched])
                    row.update(bookmaker_comparison_n=int(matched.sum()),bookmaker_log_loss=bm['log_loss'],
                               bookmaker_brier=bm['brier'],bookmaker_accuracy=bm['accuracy'],
                               matched_model_log_loss=mm['log_loss'],matched_model_brier=mm['brier'])
                    if matched.any():
                        a=-np.log(np.clip(np.where(y[matched],p[matched],1-p[matched]),1e-12,1))
                        bb=-np.log(np.clip(np.where(y[matched],b[matched],1-b[matched]),1e-12,1))
                        delta,se=evaluate.mean_se(a-bb)
                        row.update(delta_log_loss=float(delta),delta_log_loss_se=float(se))
                    else:
                        row.update(delta_log_loss=None,delta_log_loss_se=None)
                    metrics.append(row)
                    if np.isfinite(p).any():
                        cal=evaluate.calibration(p,y)
                        for bucket,r in cal.iterrows():
                            calibration.append(dict(season=int(season),league=league,model=name,market=market,
                                                    selection=selection,bin=bucket,n=int(r.n),
                                                    predicted=float(r.predicted),observed=float(r.observed)))
                    # Binary event p>=0.5 accuracy can be high for rare events. ROI uses
                    # an actual price and positive EV, not classification accuracy.
                    for mode in ('verified','estimated_sensitivity'):
                        eligible=np.isfinite(p)&np.isfinite(o)&(o>1)
                        if mode=='verified':
                            # Verified quotes must actually precede the decision, not just kickoff.
                            cap=book[(book.market==market)&(book.selection==selection)].set_index('match_id').captured_at_utc.reindex(g.match_id).to_numpy()
                            eligible &= v & (cap <= g.estimated_schedule_utc.to_numpy())
                        else:
                            eligible &= known
                        selected=eligible&(p*o-1>=.02)
                        profit=np.where(y[selected],stake*(o[selected]-1),-stake)
                        singles.append(dict(season=int(season),league=league,model=name,market=market,selection=selection,
                                            timing=mode,available_odds=int(eligible.sum()),bets=int(selected.sum()),
                                            stakes=float(stake*selected.sum()),profit=float(profit.sum()),
                                            roi=float(profit.sum()/(stake*selected.sum())) if selected.any() else None,
                                            hit_rate=float(y[selected].mean()) if selected.any() else None))
                coverage.append(dict(season=int(season),league=league,market=market,selection=selection,
                                     bet365_quotes=int(np.isfinite(o).sum()),verified_quotes=int(v.sum()),
                                     bookmaker_probability_n=int(np.isfinite(b).sum()),
                                     baseline='derived_de_vigged_1x2_event' if market=='double_chance' else 'de_vigged_market_average'))
    return dict(metrics=pd.DataFrame(metrics),calibration=pd.DataFrame(calibration),singles=pd.DataFrame(singles),coverage=pd.DataFrame(coverage))


def grouped_market_metrics(df,models,odds):
    """Top-selection hit rate plus coherent multiclass/binary scores per market.

    Double chance is scored as three separate binary events, never normalised to
    sum one; hit rate is for the highest-probability selection on each match.
    """
    fair=fair_probabilities(odds)
    bm=fair[fair.bookmaker=='market_avg'] if len(fair) else fair
    rows=[]
    for (season,league),g in df[df.season.isin([2024,2025])].groupby(['season','div']):
        for market in markets.MARKETS:
            cols=markets.columns(market)
            outcome=np.column_stack([markets.won(market,s,g.hg,g.ag) for s in markets.MARKETS[market]])
            base=bm[bm.market==market].pivot(index='match_id',columns='selection',values='fair_probability') if len(bm) else pd.DataFrame()
            b=base.reindex(index=g.match_id,columns=markets.MARKETS[market]).to_numpy(float)
            for name,frame in models.items():
                info=dict(season=int(season),league=league,model=name,market=market)
                if not set(cols)<=set(frame.columns):
                    rows.append(dict(**info,status='unsupported_model_market',predictions=0));continue
                p=frame.loc[g.index,cols].to_numpy();ok=np.isfinite(p).all(1)
                if not ok.any():
                    rows.append(dict(**info,status='no_predictions',predictions=0));continue
                def score(prob,observed):
                    n=len(prob)
                    hit=float(observed[np.arange(n),prob.argmax(1)].mean()) if n else None
                    if market=='double_chance':
                        binary=selection_metrics(prob.ravel(),observed.ravel())
                        return dict(accuracy=hit,brier=binary['brier'],log_loss=binary['log_loss'])
                    y=observed.argmax(1)
                    return {k:v for k,v in evaluate.scores(prob,y).items() if k in ['accuracy','brier','log_loss']}
                allscore=score(p[ok],outcome[ok]);matched=ok&np.isfinite(b).all(1)
                row=dict(**info,status='supported',predictions=int(ok.sum()),**allscore,
                         bookmaker_comparison_n=int(matched.sum()))
                if matched.any():
                    bs=score(b[matched],outcome[matched]);ms=score(p[matched],outcome[matched])
                    row.update({f'bookmaker_{k}':v for k,v in bs.items()})
                    row.update({f'matched_model_{k}':v for k,v in ms.items()})
                rows.append(row)
    return pd.DataFrame(rows)
