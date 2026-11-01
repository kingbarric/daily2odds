"""Owner-only nightly training entrypoint. Website visitors read saved outputs only."""
import argparse
from src.football.commercial_nightly import run
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-model',help='Recovery only: reuse an already completed local training artifact')
    args=parser.parse_args();run(reuse_model=args.reuse_model)
