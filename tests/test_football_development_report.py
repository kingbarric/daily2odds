import json
import pandas as pd
from football_development_report import write_report
from src.football.market_evaluation import grouped_market_metrics
from tests.test_football_market_evaluation import data


def test_group_metrics_keep_double_chance_as_binary_events():
    df,o,models=data()
    result=grouped_market_metrics(df,models,o)
    dc=result[(result.model=='Elo')&(result.market=='double_chance')].iloc[0]
    assert dc.predictions==2 and 0<=dc.brier<=1 and dc.accuracy==1
    unsupported=result[(result.model=='Elo')&(result.market=='btts')].iloc[0]
    assert unsupported.status=='unsupported_model_market'


def test_plain_report_contains_limits_and_escapes_untrusted_names(tmp_path):
    groups=pd.DataFrame([dict(season=2024,league='<script>bad</script>',model='Dixon-Coles',market='1x2',
        status='supported',predictions=1,accuracy=1.,brier=.4,log_loss=.5,bookmaker_comparison_n=1,
        matched_model_log_loss=.5,bookmaker_log_loss=.4,bookmaker_accuracy=1.)])
    groups.to_csv(tmp_path/'market_groups.csv',index=False)
    pd.DataFrame([{'x':1}]).to_csv(tmp_path/'metrics.csv',index=False)
    (tmp_path/'manifest.json').write_text('{}')
    summary=[dict(model='Dixon-Coles',season=2024,timing='verified',tickets=0,wins=0,losses=0,
        stakes=0,payouts=0,net_profit=0,roi=None,hit_rate=None,maximum_drawdown=0,maximum_losing_streak=0,roi_ci95=None)]
    (tmp_path/'accumulator_summary.json').write_text(json.dumps(summary))
    payload=write_report(tmp_path,tmp_path/'missing.db')
    html=(tmp_path/'report.html').read_text()
    assert '<script>bad</script>' not in html and '&lt;script&gt;' in html
    assert 'zero verified historical price timestamps' in html
    assert payload['profitability_conclusion'].startswith('No statistically credible')
    assert not (tmp_path/'missing.db').exists()
