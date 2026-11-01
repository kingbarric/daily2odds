"""Local, persistent allowlisted background jobs; no shell commands from the UI."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
JOBS=ROOT/'data/football/app_jobs'


def connect():
    JOBS.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(JOBS/'jobs.sqlite',timeout=30)
    c.row_factory=sqlite3.Row
    c.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, kind TEXT, status TEXT, created TEXT, config TEXT, result TEXT, message TEXT, pid INTEGER)')
    return c


def list_jobs():
    with connect() as c:
        rows=[dict(r) for r in c.execute('SELECT * FROM jobs ORDER BY created DESC')]
    for r in rows:
        r['config']=json.loads(r['config']);r['result']=json.loads(r['result'] or '{}')
        if r['status'] in ('queued','running') and r['pid']:
            try: os.kill(r['pid'],0)
            except PermissionError: pass  # macOS sandbox can deny probes of a running worker
            except ProcessLookupError:
                update(r['id'],status='failed',message='Worker stopped unexpectedly; start a new run')
                r['status']='failed';r['message']='Worker stopped unexpectedly'
    return rows


def validate(kind,config):
    from datetime import date
    if kind not in ('train','predict','check_results'):raise ValueError('Unknown job')
    if set(config)-{'years','compare','start','end','model_job','live','prediction_job'}:raise ValueError('Unknown setting')
    if config.get('years',5) not in (3,5,8):raise ValueError('Invalid training window')
    if kind=='check_results':
        target=next((j for j in list_jobs() if j['id']==config.get('prediction_job') and j['kind']=='predict' and j['status']=='completed'),None)
        if target is None:raise ValueError('Select a completed prediction run')
    if kind=='predict':
        start=date.fromisoformat(config['start']);end=date.fromisoformat(config['end'])
        if end<start or (end-start).days>31:raise ValueError('Choose an ordered range of at most 32 days')
        today=datetime.now(timezone.utc).date()
        if start<today<=end:raise ValueError('Run historical and future date ranges separately')
        if end>=today and (end-today).days>4:raise ValueError('Live forecasts support today and the next four days')
        if end>=today and not config.get('model_job'):raise ValueError('Train and select a model first')
    return dict(config)


def submit(kind,config):
    config=validate(kind,config);identifier=uuid.uuid4().hex
    folder=JOBS/identifier;folder.mkdir(parents=True)
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running')").fetchone():
            raise ValueError('A job is already running; wait for it to finish')
        c.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',(identifier,kind,'queued',datetime.now(timezone.utc).isoformat(),json.dumps(config),'{}','Starting',None))
    try:
        with (folder/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(ROOT/'football_app_worker.py'),identifier],cwd=ROOT,
                stdout=log,stderr=log,start_new_session=True,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        with connect() as c:c.execute('UPDATE jobs SET pid=? WHERE id=?',(proc.pid,identifier))
    except Exception:
        update(identifier,status='failed',message='Could not start worker');raise
    return identifier


def update(identifier,**values):
    if set(values)-{'status','result','message'}:raise ValueError('Invalid update')
    if 'result' in values:values['result']=json.dumps(values['result'])
    with connect() as c:
        c.execute('UPDATE jobs SET '+','.join(f'{k}=?' for k in values)+' WHERE id=?',(*values.values(),identifier))
