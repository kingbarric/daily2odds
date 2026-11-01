from streamlit.testing.v1 import AppTest
from src.football import app_jobs


def test_train_button_starts_allowlisted_job(monkeypatch):
    calls=[]
    monkeypatch.setattr(app_jobs,'list_jobs',lambda:[])
    monkeypatch.setattr(app_jobs,'submit',lambda kind,config:calls.append((kind,config)) or 'test-id')
    app=AppTest.from_file(str(app_jobs.ROOT/'football_app.py')).run()
    next(b for b in app.button if b.label=='Train model').click().run()
    assert not app.exception
    assert calls==[('train',{'years':5,'compare':True})]


def test_prediction_button_passes_selected_dates(monkeypatch):
    calls=[]
    monkeypatch.setattr(app_jobs,'list_jobs',lambda:[])
    monkeypatch.setattr(app_jobs,'submit',lambda kind,config:calls.append((kind,config)) or 'test-id')
    app=AppTest.from_file(str(app_jobs.ROOT/'football_app.py')).run()
    from datetime import date
    app.date_input[0].set_value(date(2025,10,25))
    app.date_input[1].set_value(date(2025,10,26))
    next(b for b in app.button if b.label=='Generate predictions').click().run()
    assert not app.exception
    assert calls[0][0]=='predict'
    assert calls[0][1]['start']=='2025-10-25' and calls[0][1]['end']=='2025-10-26'


def test_check_results_button_targets_saved_run(monkeypatch):
    calls=[]
    saved=next(j for j in app_jobs.list_jobs() if j['kind']=='predict' and j['status']=='completed')
    monkeypatch.setattr(app_jobs,'list_jobs',lambda:[saved])
    monkeypatch.setattr(app_jobs,'submit',lambda kind,config:calls.append((kind,config)) or 'check-id')
    app=AppTest.from_file(str(app_jobs.ROOT/'football_app.py')).run()
    next(b for b in app.button if b.label=='Check results').click().run()
    assert not app.exception
    assert calls==[('check_results',{'prediction_job':saved['id']})]
