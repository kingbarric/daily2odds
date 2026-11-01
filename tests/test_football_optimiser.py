"""Snapshot eligibility, conservative arithmetic and bounded ticket selection."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.football.optimiser import Pick, Ticket, optimise

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def pick(match="a", **changes):
    values = dict(snapshot_id=f"snapshot-{match}", match_id=match,
                  bookmaker="book", market="1x2", selection="home", odds=2.2,
                  probability=0.8, p_lower=0.7,
                  captured_at_utc=NOW - timedelta(minutes=10),
                  kickoff_utc=NOW + timedelta(hours=2))
    values.update(changes)
    return Pick(**values)


def run(picks, **kwargs):
    return optimise(picks, now=NOW, **kwargs)


def test_exact_interface_and_arithmetic():
    a, b = pick(odds=2), pick("b", odds=2.5, probability=0.75, p_lower=0.6)
    result = run([a, b], correlation_haircut=0.1)
    assert len(result) == 1
    t = result[0]
    assert isinstance(t, Ticket)
    assert set(t.legs) == {a, b}
    assert t.combined_odds == 5
    assert t.probability == pytest.approx(0.8 * 0.75)
    assert t.p_lower == pytest.approx(0.7 * 0.6 * 0.9)
    assert t.ev == pytest.approx(5 * 0.8 * 0.75 - 1)
    assert t.robust_ev == pytest.approx(5 * 0.7 * 0.6 * 0.9 - 1)


@pytest.mark.parametrize("odds,qualifies", [(4.75, True), (5.25, True),
                                           (4.749, False), (5.251, False)])
def test_inclusive_price_boundaries(odds, qualifies):
    assert bool(run([pick(odds=2), pick("b", odds=odds / 2)])) == qualifies


def test_duplicate_matches_even_across_markets_never_form_ticket():
    assert run([pick(), pick(selection="away"), pick(market="btts", selection="yes")]) == []


def test_same_bookmaker_required_no_synthetic_prices():
    assert run([pick(), pick("b", bookmaker="other")]) == []
    picks = [pick(), pick("b"), pick("c", bookmaker="other"),
             pick("d", bookmaker="other")]
    for ticket in run(picks):
        assert len({p.bookmaker for p in ticket.legs}) == 1
        assert all(p in picks for p in ticket.legs)


def test_uncertainty_can_reject_nominally_profitable_ticket():
    assert run([pick(p_lower=0.3), pick("b", p_lower=0.3)]) == []
    assert run([pick(), pick("b")], min_robust_ev=2) == []


def test_robust_ev_threshold_is_inclusive():
    picks = [pick(odds=2, p_lower=0.5), pick("b", odds=2.5, p_lower=0.5)]
    assert len(run(picks, correlation_haircut=0, min_robust_ev=0.25)) == 1
    assert run(picks, correlation_haircut=0, min_robust_ev=0.250001) == []


def test_haircut_per_additional_leg():
    picks = [pick(str(i), odds=5 ** (1 / 3)) for i in range(3)]
    t = run(picks, correlation_haircut=0.1)[0]
    assert len(t.legs) == 3
    assert t.p_lower == pytest.approx(0.7 ** 3 * 0.9 ** 2)
    assert run(picks, correlation_haircut=1) == []


@pytest.mark.parametrize("field,value", [
    ("captured_at_utc", NOW - timedelta(minutes=60, microseconds=1)),
    ("captured_at_utc", NOW + timedelta(microseconds=1)),
    ("kickoff_utc", NOW),
    ("kickoff_utc", NOW - timedelta(microseconds=1)),
])
def test_ineligible_timestamps_are_excluded(field, value):
    assert run([pick(**{field: value}), pick("b")]) == []


@pytest.mark.parametrize("capture", [NOW, NOW - timedelta(minutes=60)])
def test_age_endpoints_are_inclusive(capture):
    assert run([pick(captured_at_utc=capture), pick("b")])


def test_custom_max_age():
    picks = [pick(), pick("b")]
    assert run(picks, max_odds_age=timedelta(minutes=9)) == []
    assert run(picks, max_odds_age=timedelta(minutes=10))


@pytest.mark.parametrize("field,value", [
    ("odds", 1), ("odds", 0), ("odds", float("inf")),
    ("odds", float("nan")), ("odds", True),
    ("probability", -0.1), ("probability", 1.1),
    ("probability", float("nan")), ("probability", float("inf")),
    ("p_lower", -0.1), ("p_lower", 0.81), ("p_lower", float("nan")),
    ("p_lower", float("inf")), ("snapshot_id", ""), ("match_id", " "),
    ("bookmaker", None), ("market", ""), ("selection", ""),
    ("home", []), ("away", 1), ("league", {}),
    ("captured_at_utc", NOW.replace(tzinfo=None)),
    ("kickoff_utc", "2026-10-27"),
    ("kickoff_utc", NOW.astimezone(timezone(timedelta(hours=1)))),
])
def test_invalid_pick_rejected(field, value):
    with pytest.raises(ValueError):
        pick(**{field: value})


def test_zero_and_one_probabilities_valid():
    assert pick(probability=0, p_lower=0).probability == 0
    assert run([pick(probability=0, p_lower=0), pick("b")]) == []
    assert run([pick(probability=1, p_lower=1), pick("b", probability=1, p_lower=1)])


@pytest.mark.parametrize("kwargs", [
    {"min_odds": float("nan")}, {"max_odds": float("inf")},
    {"min_odds": 6}, {"min_legs": 1}, {"max_legs": 6},
    {"min_legs": 4, "max_legs": 3}, {"max_tickets": 4},
    {"max_tickets": True}, {"beam_width": 0}, {"max_candidates": 0},
    {"correlation_haircut": -0.1}, {"correlation_haircut": 1.1},
    {"correlation_haircut": float("nan")},
    {"min_robust_ev": float("inf")}, {"min_robust_ev": -1},
    {"max_odds_age": timedelta(seconds=-1)}, {"max_odds_age": 60},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        run([], **kwargs)


def test_now_must_be_explicit_aware_utc():
    with pytest.raises(TypeError):
        optimise([])
    with pytest.raises(ValueError):
        optimise([], now=NOW.replace(tzinfo=None))


def test_no_qualifying_inputs():
    assert run([]) == []
    assert run([pick()]) == []
    assert run([pick(), pick("b")], max_tickets=0) == []
    with pytest.raises(ValueError):
        run([None])


def test_max_three_tickets_all_matches_disjoint_and_deterministic():
    picks = [pick(str(i)) for i in range(10)]
    result = run(iter(picks))
    assert len(result) == 3
    matches = [p.match_id for t in result for p in t.legs]
    assert len(matches) == len(set(matches))
    assert result == run(reversed(picks))
    assert len(run(picks, max_tickets=1)) == 1
    assert all(2 <= len(t.legs) <= 5 for t in result)


def test_duplicate_snapshots_do_not_change_results():
    picks = [pick(), pick("b")]
    assert run(picks * 5) == run(picks)


def test_five_legs_and_configurable_bounds():
    picks = [pick(str(i), odds=5 ** 0.2, probability=1, p_lower=1) for i in range(6)]
    result = run(picks, min_legs=5)
    assert len(result) == 1 and len(result[0].legs) == 5
    assert run(picks, max_legs=4) == []
    assert run([pick(), pick("b")], min_odds=6, max_odds=7) == []


def test_bounded_search_and_no_input_mutation():
    picks = [pick(str(i)) for i in range(200)]
    original = list(picks)
    assert len(run(picks, max_candidates=2, beam_width=1)) == 1
    assert picks == original
    assert replace(picks[0], home="Home", away="Away", league="League").home == "Home"


def test_probability_ranking_does_not_prefer_larger_payout_over_win_probability():
    from datetime import datetime,timedelta,timezone
    from src.football.optimiser import Pick,optimise
    now=datetime(2026,10,29,12,tzinfo=timezone.utc)
    values=[]
    for book,price,p in [('A',2.2,.8),('B',2.29,.79)]:
        for i in range(2):
            values.append(Pick(book+str(i),book+str(i),book,'1x2','home',price,p,p,now,now+timedelta(days=1)))
    assert optimise(values,now=now,max_tickets=1,rank_by='probability')[0].legs[0].bookmaker=='A'
    assert optimise(values,now=now,max_tickets=1,rank_by='ev')[0].legs[0].bookmaker=='B'
