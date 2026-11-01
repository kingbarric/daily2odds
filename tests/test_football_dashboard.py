"""Run the actual Streamlit app against isolated databases, never the user's file."""
from pathlib import Path

import pytest

streamlit = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

from src.football.forward_store import ForwardStore
from tests.test_football_workflow import historical, NOW, RecordedProvider, predictor
from src.football.workflow import run_cycle

APP = str(Path(__file__).resolve().parents[1]/'football_dashboard.py')


def test_dashboard_missing_database_has_no_sample_data(tmp_path, monkeypatch):
    path = tmp_path/'missing.db'
    monkeypatch.setenv('FOOTBALL_DB_PATH', str(path))
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert 'No database' in app.info[0].value
    assert not path.exists()


def test_dashboard_preserves_historical_database_without_migration(tmp_path, monkeypatch):
    path = tmp_path/'historical.db'
    historical(path)
    before = path.read_bytes()
    monkeypatch.setenv('FOOTBALL_DB_PATH', str(path))
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert 'Historical database found' in app.info[0].value
    assert path.read_bytes() == before


def test_dashboard_renders_all_tabs_and_no_bet_read_only(tmp_path, monkeypatch):
    path = tmp_path/'forward.db'
    historical(path)
    store = ForwardStore(path)
    run_cycle(store, RecordedProvider(), path, clock=lambda: NOW, predictor=predictor)
    store.close()
    before = path.read_bytes()
    monkeypatch.setenv('FOOTBALL_DB_PATH', str(path))
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert len(app.tabs) == 6
    assert any('NO BET' in element.value for element in app.info)
    assert len(app.metric) == 4
    assert path.read_bytes() == before
