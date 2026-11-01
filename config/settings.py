"""Standalone Daily2Odds paths; secrets are loaded from the local environment."""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/'data'
MODELS_DIR=PROJECT_ROOT/'models'
LOGS_DIR=PROJECT_ROOT/'logs'
for directory in (DATA_DIR,MODELS_DIR,LOGS_DIR):
    directory.mkdir(parents=True,exist_ok=True)
