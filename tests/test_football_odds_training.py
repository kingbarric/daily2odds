import numpy as np
import pandas as pd
from src.football.odds_training import price_features,historical_odds
from src.football import db
from tests.test_football_bot import parsed


def test_odds_features_are_margin_removed_and_missing_prices_stay_missing():
    f=price_features([[2,3.5,4],[2,np.nan,4]])
    assert np.isclose(f.iloc[0,:3].sum(),1)
    assert np.isclose(f.iloc[0].odds_overround,.5+1/3.5+.25)
    assert f.iloc[1].isna().all()


def test_closing_prices_cannot_enter_training_features(tmp_path):
    m,o,_=parsed(tmp_path);path=tmp_path/'history.db'
    with db.engine(path).begin() as c:
        db.ingest(c,m,o,'recorded')
        # Fixture source has bet365 prices; copy it to the model's consensus source.
        c.exec_driver_sql("UPDATE odds_snapshots SET bookmaker='market_avg' WHERE bookmaker='bet365'")
    frame=pd.DataFrame({'match_id':m.match_id})
    before=historical_odds(path,frame)
    with db.engine(path).begin() as c:
        rows=[dict(r) for r in c.execute(db.odds_snapshots.select().where(db.odds_snapshots.c.bookmaker=='market_avg',db.odds_snapshots.c.snapshot=='pre_closing')).mappings()]
        for row in rows:row.update(snapshot='closing',odds=99)
        c.execute(db.odds_snapshots.insert(),rows)
    pd.testing.assert_frame_equal(before,historical_odds(path,frame))
