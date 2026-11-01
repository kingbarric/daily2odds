"""Conservative calibration bounds from previously frozen, settled paper forecasts only.

Wilson bounds describe bin hit-rate uncertainty, not a guarantee about an individual
match. Empty/undersampled bins fail closed. No historical report is treated as live evidence.
"""
import math

from src.football.markets import won


def lower_bounds(probabilities, forecasts, fixtures, as_of, *, minimum_samples=100, z=1.96, league=None):
    from src.football.forward_store import utc
    cutoff = utc(as_of)
    by_id = {str(f['provider_id']): f for f in fixtures}
    bounds, counts = {}, {}
    for key, probability in probabilities.items():
        market, selection = key.split(':')
        bucket = min(int(probability * 10), 9)
        observations = {}
        seen = set()
        # Only one forecast per match/market/selection enters calibration. Later revisions
        # are not extra independent samples. Source result must have arrived before as_of.
        for forecast in sorted(forecasts, key=lambda f: f['as_of']):
            fixture = by_id.get(str(forecast['provider_id']))
            if not fixture or fixture['status'] != 'finished':
                continue
            if league is not None and fixture.get('league') != league:
                continue
            if utc(forecast['as_of']) >= cutoff or utc(fixture['updated_at']) >= cutoff:
                continue
            if utc(forecast['kickoff_utc']) <= utc(forecast['as_of']):
                continue
            if utc(fixture['kickoff_utc']) >= cutoff:
                continue
            q = forecast['probabilities'].get(key)
            if q is None or fixture['provider_id'] in seen:
                continue
            seen.add(fixture['provider_id'])  # choose earliest before assigning its probability bin
            if min(int(q * 10), 9) != bucket:
                continue
            if fixture['hg'] is None or fixture['ag'] is None:
                continue
            observations.setdefault(fixture['provider_id'], bool(won(market, selection, fixture['hg'], fixture['ag'])))
        n = len(observations)
        counts[key] = n
        if n < minimum_samples:
            bounds[key] = 0.0
            continue
        observed = sum(observations.values()) / n
        denominator = 1 + z*z/n
        lower = (observed + z*z/(2*n) - z*math.sqrt(observed*(1-observed)/n + z*z/(4*n*n))) / denominator
        bounds[key] = min(float(probability), max(0.0, lower))
    return bounds, counts
