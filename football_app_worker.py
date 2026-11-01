"""Allowlisted worker entry point used by the local Streamlit site."""
import sys
from src.football.app_jobs import list_jobs,update,ROOT
from src.football.db import DB_PATH


def run(identifier):
    job=next(j for j in list_jobs() if j['id']==identifier)
    config=job['config'];output=ROOT/'reports/football_app'/identifier
    update(identifier,status='running',message='Loading data')
    def progress(message):
        print(message,flush=True);update(identifier,message=message)
    try:
        if job['kind']=='train':
            from src.football.app_training import train
            result=train(DB_PATH,output,config.get('years',5),config.get('compare',True),progress)
        elif job['kind']=='check_results':
            from src.football.app_results import check_results
            saved=next(j for j in list_jobs() if j['id']==config['prediction_job'] and j['kind']=='predict' and j['status']=='completed')
            progress('Checking completed match scores; original predictions stay frozen')
            result=check_results(saved['result']['output'])
        elif config['end']<__import__('datetime').datetime.now(__import__('datetime').timezone.utc).date().isoformat():
            from src.football.app_predictions import historical_range
            result=historical_range(config['start'],config['end'],config.get('years',5),output,progress)
        else:
            from src.football.app_predictions import live_range
            model=next(j for j in list_jobs() if j['id']==config['model_job'] and j['kind']=='train' and j['status']=='completed')
            result=live_range(config['start'],config['end'],__import__('pathlib').Path(model['result']['output'])/'lightgbm_with_odds.joblib',output,progress)
        update(identifier,status='completed',result=result,message='Complete')
    except Exception as exc:
        # Provider exceptions can contain URLs with API credentials. Never persist their text.
        message='Job failed ('+type(exc).__name__+'). Check data coverage, chronological history, network access and API budget.'
        if isinstance(exc,ValueError) and 'http' not in str(exc).lower() and 'key' not in str(exc).lower():message=str(exc)
        print(message,flush=True);update(identifier,status='failed',message=message)

if __name__=='__main__':run(sys.argv[1])
