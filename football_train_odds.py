"""Train the existing LightGBM with historical pre-closing odds; preserve prior reports."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from src.football.db import DB_PATH
from src.football.workflow import load_history
from src.football.odds_training import historical_odds,train

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,default=DB_PATH)
    p.add_argument('--output',type=Path,default=Path('reports/football_odds_training_20261029'))
    a=p.parse_args()
    if (a.output/'lightgbm_with_odds.joblib').exists():p.error('choose a new output directory to preserve the previous trained artifact')
    history,_=load_history(a.db,datetime.now(timezone.utc))
    train(history,historical_odds(a.db,history),a.output)
