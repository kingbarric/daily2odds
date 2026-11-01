from datetime import datetime,timezone,timedelta
import pandas as pd
import pytest
from src.football.app_training import split_masks
from src.football import app_jobs


def test_windows_are_disjoint_and_earlier():
    df=pd.DataFrame({'season':list(range(2010,2028))})
    for years in (3,5,8):
        train,cal,test=split_masks(df,2026,years)
        assert train.sum()==years
        assert df.loc[train,'season'].max()==2024
        assert df.loc[cal,'season'].tolist()==[2025]
        assert not (train&cal).any() and not (train&test).any()
        assert df.loc[test,'season'].tolist()==[2026]


def test_job_validation():
    with pytest.raises(ValueError):app_jobs.validate('shell',{})
    with pytest.raises(ValueError):app_jobs.validate('train',{'command':'echo key'})
    with pytest.raises(ValueError):app_jobs.validate('train',{'years':2})
    with pytest.raises(ValueError):app_jobs.validate('predict',{'start':'2025-01-03','end':'2025-01-01'})
    today=datetime.now(timezone.utc).date()
    with pytest.raises(ValueError):app_jobs.validate('predict',{'start':str(today),'end':str(today+timedelta(days=5)),'model_job':'x'})
    with pytest.raises(ValueError):app_jobs.validate('predict',{'start':str(today),'end':str(today)})


def test_single_background_job(monkeypatch,tmp_path):
    monkeypatch.setattr(app_jobs,'JOBS',tmp_path)
    class Proc:pid=123
    monkeypatch.setattr(app_jobs.subprocess,'Popen',lambda *a,**kw:Proc())
    app_jobs.submit('train',{'years':5})
    with pytest.raises(ValueError,match='already running'):app_jobs.submit('train',{'years':3})


def test_site_renders_and_filters_saved_results():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(app_jobs.ROOT/'football_app.py')).run(timeout=30)
    assert not at.exception
    assert [t.label for t in at.tabs]==['Train','Predict','Explore results']
    assert {'Train model','Generate predictions'}.issubset({b.label for b in at.button})
    probability=next(s for s in at.slider if s.label=='Selection probability (%)')
    probability.set_value((99,100));at.run()
    assert not at.exception
    assert at.metric[0].value=='0'


def test_site_saved_replay_source():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(app_jobs.ROOT/'football_app.py')).run(timeout=30)
    source=next(s for s in at.selectbox if s.label=='Result set')
    source.select('Saved blind replay · 28 October 2026');at.run()
    assert not at.exception
    assert 'actual_score' in at.dataframe[-1].value.columns
