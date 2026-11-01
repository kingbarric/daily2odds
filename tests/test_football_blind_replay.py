from datetime import datetime,timezone,timedelta
import json
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import pytest

import football_blind_replay as replay
from src.football.markets import from_1x2
from src.football.odds_training import live_price_features

ASOF=datetime(2026,10,28,0,18,tzinfo=timezone.utc)


def history():
    return pd.DataFrame(dict(date=pd.to_datetime(['2026-10-26','2026-10-28','2026-10-29']),
        home=['A']*3,away=['B']*3,hg=[1,9,8],ag=[0,8,9]))


def test_target_day_and_later_results_are_always_removed():
    before=history();changed=history();changed.loc[1:,['hg','ag']]=[90,70]
    pd.testing.assert_frame_equal(replay.blind_history(before,ASOF),replay.blind_history(changed,ASOF))
    assert len(replay.blind_history(before,ASOF))==1


def test_future_archive_is_refused(tmp_path):
    metadata=tmp_path/'meta.json';metadata.write_text(json.dumps({'captured_at_utc':'2026-10-29T00:00:00+00:00'}))
    with pytest.raises(ValueError,match='not available'):
        replay.archived_inputs(tmp_path/'unread.csv',metadata,ASOF,ASOF.date(),{'A','B'})


def test_stale_sensitivity_never_allows_future_prices():
    class Provider:
        captured=ASOF-timedelta(minutes=5);source=ASOF-timedelta(hours=12)
        def odds(self,identifier):
            return [dict(market='1x2',selection=s,odds=p,bookmaker='Book',source_updated_at=self.source,captured_at_utc=self.captured)
                    for s,p in [('home',2),('draw',3.5),('away',4)]]
    p=Provider();fixtures=[{'provider_id':'a'}]
    assert live_price_features(p,fixtures,ASOF).isna().all().all()
    assert live_price_features(p,fixtures,ASOF,max_source_age=None).notna().all().all()
    p.captured=ASOF+timedelta(seconds=1)
    assert live_price_features(p,fixtures,ASOF,max_source_age=None).isna().all().all()
    p.captured=ASOF-timedelta(minutes=1);p.source=ASOF+timedelta(seconds=1)
    assert live_price_features(p,fixtures,ASOF,max_source_age=None).isna().all().all()


def test_predictions_saved_before_actuals_are_loaded(tmp_path,monkeypatch):
    csv=tmp_path/'archive.csv';csv.write_text('Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\nE0,28/10/2026,17:30,A,B,999,999,2,3.5,4\n')
    meta=tmp_path/'meta.json';meta.write_text(json.dumps(dict(captured_at_utc='2026-10-28T00:08:00+00:00',batch_published_at_utc='2026-10-27T12:00:00+00:00')))
    model=tmp_path/'model.joblib';joblib.dump({'metadata':{'training_seasons':[2024],'calibration_season':2025}},model)
    actual=tmp_path/'actual.json';actual.write_text(json.dumps({'checked_at_utc':'2026-10-28T23:56:00+00:00','results':[
        {'provider_id':'fd:E0:2026:A|B','status':'finished','hg':0,'ag':2}]}))
    def factory(path,provider,**kwargs):
        def predict(hist,fixtures,asof):
            assert len(hist)==1 and hist.hg.max()==1
            assert all(f['hg'] is None and f['ag'] is None for f in fixtures)
            provider.prediction_sources={f['provider_id']:'mock-blind' for f in fixtures}
            return {f['provider_id']:from_1x2(np.array([[.6,.25,.15]])).iloc[0].to_dict() for f in fixtures},'mock'
        return predict
    monkeypatch.setattr(replay,'predictor',factory)
    monkeypatch.setattr(replay,'load_history',lambda *a:(history(),None))
    output=tmp_path/'report';original_join=replay.append_actuals
    def audit_join(predictions,path):
        assert (output/'predictions_before_results.csv').exists()
        assert (output/'selections_before_results.csv').exists()
        assert 'actual_score' not in pd.read_csv(output/'predictions_before_results.csv').columns
        return original_join(predictions,path)
    monkeypatch.setattr(replay,'append_actuals',audit_join)
    args=SimpleNamespace(day='2026-10-28',db=tmp_path/'unused.db',archive=csv,archive_metadata=meta,model=model,results=actual,output=output)
    replay.run(args)
    joined=pd.read_csv(output/'selected_markets_with_actuals.csv')
    assert (joined.actual_score=='0–2').all()
    assert not joined[joined.market=='1x2'].selection_won.any()
    assert json.loads((output/'prediction_manifest.json').read_text())['predictions_saved_before_result_file_read']
