"""Chronological accumulator replay. Strict evidence and hypothetical timing never mix.

Only 2024/25 and 2025/26 are development targets. Calibration warms up from
2023/24, and is updated only after the latest plausible full-time result cutoff.
Historical price availability is unknown for estimated rows: sensitivity tickets
record NULL verified receipt and a separately labelled assumed decision timestamp.
"""
from collections import defaultdict
from datetime import timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from itertools import groupby
from math import prod
from math import sqrt
import hashlib
import heapq
import json

import numpy as np
import pandas as pd

from . import markets
from .optimiser import Pick, optimise


def result_available(row):
    """Conservative: no earlier than next UTC date or kickoff+3h, whichever later."""
    day=pd.Timestamp(row.date).normalize()+pd.Timedelta(days=1)
    kickoff=pd.Timestamp(row.kickoff_utc)
    return max(day,kickoff+pd.Timedelta(hours=3)) if pd.notna(kickoff) else day


class PastCalibration:
    def __init__(self, minimum_samples=100):
        self.minimum_samples=minimum_samples
        self.counts=defaultdict(lambda:[0,0])

    @staticmethod
    def key(league,selection,probability):
        return league,selection,min(int(probability*10),9)

    def observe(self,league,selection,probability,outcome):
        wins,n=self.counts[self.key(league,selection,probability)]
        self.counts[self.key(league,selection,probability)]=[wins+bool(outcome),n+1]

    def lower(self,league,selection,probability):
        wins,n=self.counts[self.key(league,selection,probability)]
        if n<self.minimum_samples:
            return 0.,n
        p=wins/n;z=1.96
        bound=(p+z*z/(2*n)-z*sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
        return min(probability,max(0.,bound)),n


def performance(tickets, stake=10):
    """Realised equity in settlement order; win/lose streak also by settlement.

    Date-clustered bootstrap intervals are descriptive development uncertainty,
    not a correction for model/strategy selection or unverified prices.
    """
    records=sorted(tickets,key=lambda t:(t['settled_at_utc'],t['ticket_id']))
    equity=peak=drawdown=0.;run=worst=0
    # Same-cutoff settlements are one batch: arbitrary ticket-ID ordering must
    # not manufacture a transient equity trough within a simultaneous batch.
    for _,batch in groupby(records,key=lambda t:t['settled_at_utc']):
        equity+=sum(t['net_profit'] for t in batch)
        peak=max(peak,equity);drawdown=max(drawdown,peak-equity)
    for t in records:
        run=run+1 if t['outcome']=='lost' else 0;worst=max(worst,run)
    n=len(records);stakes=sum(t['stake'] for t in records);payout=sum(t['payout'] for t in records)
    result=dict(tickets=n,stake_per_ticket=stake,stakes=stakes,payouts=payout,net_profit=payout-stakes,
                roi=(payout-stakes)/stakes if stakes else None,
                hit_rate=sum(t['outcome']=='won' for t in records)/n if n else None,
                wins=sum(t['outcome']=='won' for t in records),losses=sum(t['outcome']=='lost' for t in records),
                maximum_losing_streak=worst,maximum_drawdown=drawdown,roi_ci95=None,decision_days=0)
    if records:
        days=defaultdict(lambda:[0.,0.])
        for t in records:
            days[t['decision_utc'][:10]][0]+=t['net_profit']
            days[t['decision_utc'][:10]][1]+=t['stake']
        a=np.array(list(days.values()));result['decision_days']=len(a)
        if len(a)>=10:
            rng=np.random.default_rng(2024)
            means=[]
            for _ in range(1000):
                sample=a[rng.integers(0,len(a),len(a))].sum(0)
                means.append(sample[0]/sample[1])
            result['roi_ci95']=[float(x) for x in np.quantile(means,[.025,.975])]
    return result


def replay(df, probabilities, odds, *, model_name, mode='verified', stake=10,
           minimum_samples=100, beam_width=250, max_candidates=60, min_robust_ev=.02):
    if mode not in ('verified','estimated_sensitivity'):
        raise ValueError('invalid timing mode')
    if stake<=0 or not np.isfinite(stake):
        raise ValueError('positive finite stake required')
    events=[];cal=PastCalibration(minimum_samples);tickets=[];decisions=[]
    meta={r.match_id:r for r in df.itertuples()}
    odds=odds[~odds.bookmaker.isin(['market_avg','market_max','betfair_exchange'])].copy()
    odds_by_match={mid:g for mid,g in odds.groupby('match_id',sort=False)}
    sequence=0
    # Use the same scheduled collection groups as the historical feature cutoff.
    # Strict quote rows are separately checked against actual verified receipt.
    for day,group in df[df.season>=2023].groupby('decision',sort=True):
        while events and events[0][0]<day:
            _,_,league,key,p,win=heapq.heappop(events)
            cal.observe(league,key,p,win)
        picks=[];evidence={};known={}
        for row in group.itertuples():
            for key in probabilities.columns:
                p=float(probabilities.at[row.Index,key])
                if np.isfinite(p):
                    market,selection=key.split(':')
                    win=bool(markets.won(market,selection,row.hg,row.ag))
                    heapq.heappush(events,(result_available(row),sequence,row.div,key,p,win));sequence+=1
            if row.season not in (2024,2025) or row.match_id not in odds_by_match:
                continue
            quote_rows=odds_by_match[row.match_id]
            for q in quote_rows.itertuples():
                key=f'{q.market}:{q.selection}'
                if key not in probabilities:
                    continue
                p=float(probabilities.at[row.Index,key])
                if not np.isfinite(p):
                    continue
                kickoff=pd.Timestamp(row.kickoff_utc)
                verified=bool(q.timestamp_verified)
                if mode=='verified':
                    captured=pd.Timestamp(q.captured_at_utc)
                    decision=pd.Timestamp(row.estimated_schedule_utc)
                    if pd.isna(decision):
                        decision=day
                    if not verified or pd.isna(captured) or pd.isna(kickoff) or not captured<=decision<kickoff:
                        continue
                else:
                    decision=pd.Timestamp(row.estimated_schedule_utc)
                    if pd.isna(decision) or pd.isna(kickoff) or not decision<kickoff:
                        continue
                    # Deliberately hypothetical. Original receipt remains NULL in output.
                    captured=decision
                lower,n=cal.lower(row.div,key,p)
                if lower<=0 or lower*q.odds<=1:
                    continue
                identifier=hashlib.sha256(f'{row.match_id}|{q.bookmaker}|{key}|{q.odds}'.encode()).hexdigest()[:24]
                picks.append(Pick(snapshot_id=identifier,match_id=row.match_id,bookmaker=q.bookmaker,
                                  market=q.market,selection=q.selection,odds=float(q.odds),probability=p,p_lower=lower,
                                  captured_at_utc=captured.to_pydatetime().replace(tzinfo=timezone.utc),
                                  kickoff_utc=kickoff.to_pydatetime().replace(tzinfo=timezone.utc),
                                  home=row.home,away=row.away,league=row.div,
                                  timestamp_verified=(verified if mode=='verified' else False)))
                evidence[identifier]=dict(calibration_n=n,actual_captured_at_utc=str(q.captured_at_utc) if verified else None,
                                          estimated_timestamps=not verified)
                known[identifier]=decision
        if not group.season.isin([2024,2025]).any():
            continue
        # Calendar collection groups can contain different schedule times only for
        # invalid/holiday quotes, already excluded. Use exact assumed quote time.
        # Features/calibration stay more conservative: results strictly before day.
        now=max((p.captured_at_utc for p in picks),default=day.to_pydatetime().replace(tzinfo=timezone.utc))
        selected=optimise(picks,now=now,beam_width=beam_width,max_candidates=max_candidates,
                          min_robust_ev=min_robust_ev,allow_estimated_timestamps=(mode=='estimated_sensitivity'))
        season=int(group.season.max())
        decisions.append(dict(model=model_name,timing=mode,season=season,decision_utc=now.isoformat(),
                              candidates=len(picks),tickets=len(selected),decision='TICKETS' if selected else 'NO BET'))
        for i,t in enumerate(selected):
            legs=[]
            for pick in t.legs:
                row=meta[pick.match_id];win=bool(markets.won(pick.market,pick.selection,row.hg,row.ag))
                legs.append(dict(match_id=pick.match_id,league=row.div,home=row.home,away=row.away,
                                 market=pick.market,selection=pick.selection,probability=pick.probability,
                                 conservative_probability=pick.p_lower,price=pick.odds,bookmaker=pick.bookmaker,
                                 price_timestamp_verified=pick.timestamp_verified,
                                 captured_at_utc=evidence[pick.snapshot_id]['actual_captured_at_utc'],
                                 assumed_available_at_utc=pick.captured_at_utc.isoformat() if mode!='verified' else None,
                                 calibration_n=evidence[pick.snapshot_id]['calibration_n'],
                                 kickoff_utc=pick.kickoff_utc.isoformat(),score=f'{row.hg}-{row.ag}',won=win))
            win=all(l['won'] for l in legs)
            decimal_odds=prod(Decimal(str(p.odds)) for p in t.legs)
            payout=float((Decimal(str(stake))*decimal_odds).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)) if win else 0.
            identity=hashlib.sha256(json.dumps([model_name,mode,now.isoformat(),[p.snapshot_id for p in t.legs]]).encode()).hexdigest()[:24]
            tickets.append(dict(ticket_id=identity,model=model_name,timing=mode,season=season,
                                decision_utc=now.isoformat(),settled_at_utc=max(result_available(meta[p.match_id]) for p in t.legs).isoformat()+'+00:00',
                                stake=stake,combined_odds=t.combined_odds,joint_probability=t.probability,
                                conservative_joint_probability=t.p_lower,dependence_haircut=.05,
                                nominal_ev=t.ev,conservative_ev=t.robust_ev,
                                outcome='won' if win else 'lost',payout=payout,net_profit=payout-stake,legs=legs))
    summaries=[]
    for season in [2024,2025]:
        records=[t for t in tickets if t['season']==season]
        summaries.append(dict(season=season,model=model_name,timing=mode,
                              **performance(records,stake),no_bet_days=sum(d['season']==season and d['tickets']==0 for d in decisions),
                              losing_streak_order='settlement cutoff, then deterministic ticket id for ties',
                              drawdown_method='realised equity at conservative settlement batches',
                              payout_rounding='penny, ROUND_HALF_UP'))
    return dict(tickets=tickets,decisions=decisions,summaries=summaries)
