"""Pure accumulator selection from supplied bookmaker snapshots and probabilities.

No probabilities or prices are fitted here. Joint probabilities use an independence
product. The lower product is multiplied by (1 - correlation_haircut)**(legs - 1).
This correlation penalty is an explicit heuristic, NOT a confidence guarantee:
individual lower estimates do not imply a joint lower bound under dependence.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from heapq import nlargest
from math import isfinite, prod
from numbers import Real
from typing import Iterable


def _number(name: str, value: float, low: float, high: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def _utc(name: str, value: datetime) -> None:
    if (not isinstance(value, datetime) or value.utcoffset() != timedelta(0)):
        raise ValueError(f"{name} must be an aware UTC datetime")


@dataclass(frozen=True)
class Pick:
    snapshot_id: str
    match_id: str
    bookmaker: str
    market: str
    selection: str
    odds: float
    probability: float
    p_lower: float
    captured_at_utc: datetime
    kickoff_utc: datetime
    home: str | None = None
    away: str | None = None
    league: str | None = None
    timestamp_verified: bool = True

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "match_id", "bookmaker", "market", "selection"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.timestamp_verified) is not bool:
            raise ValueError("timestamp_verified must be boolean")
        for name in ("home", "away", "league"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        _number("odds", self.odds, 1.0, float("inf"))
        if self.odds <= 1:
            raise ValueError("odds must be greater than 1")
        _number("probability", self.probability, 0, 1)
        _number("p_lower", self.p_lower, 0, self.probability)
        _utc("captured_at_utc", self.captured_at_utc)
        _utc("kickoff_utc", self.kickoff_utc)


@dataclass(frozen=True)
class Ticket:
    legs: tuple[Pick, ...]
    combined_odds: float
    probability: float
    p_lower: float
    ev: float
    robust_ev: float


def optimise(
    picks: Iterable[Pick],
    *,
    now: datetime,
    min_odds: float = 4.75,
    max_odds: float = 5.25,
    min_legs: int = 2,
    max_legs: int = 5,
    max_tickets: int = 3,
    min_robust_ev: float = 0.0,
    correlation_haircut: float = 0.05,
    max_odds_age: timedelta = timedelta(minutes=60),
    beam_width: int = 1000,
    max_candidates: int = 100,
    allow_estimated_timestamps: bool = False,
    rank_by: str = "ev",
) -> list[Ticket]:
    """Return up to three match-disjoint tickets, or [] (NO BET).

    Odds and robust-EV thresholds are inclusive. Prices must be captured in
    [now - max_odds_age, now], with kickoff strictly after now. Invalid inputs
    raise ValueError; valid but ineligible snapshots are excluded. All picks are
    validated before filtering, including those beyond the search cap.

    Candidates are ranked by individual conservative return, then searched in a
    bounded beam. Tickets are ranked by robust EV, then nominal EV; selection is
    greedy and match-disjoint. Caps bound search work, so neither exhaustive
    coverage nor a globally optimal diversified portfolio is guaranteed.
    EV is net expected return per unit stake: probability * decimal odds - 1.
    """
    _utc("now", now)
    if rank_by not in ("ev", "probability"):
        raise ValueError("rank_by must be ev or probability")
    _number("min_odds", min_odds, 1, float("inf"))
    _number("max_odds", max_odds, min_odds, float("inf"))
    _number("min_robust_ev", min_robust_ev, 0, float("inf"))
    _number("correlation_haircut", correlation_haircut, 0, 1)
    for name, value, low, high in (
        ("min_legs", min_legs, 2, 5), ("max_legs", max_legs, min_legs, 5),
        ("max_tickets", max_tickets, 0, 3),
        ("beam_width", beam_width, 1, 100_000),
        ("max_candidates", max_candidates, 1, 10_000),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if not isinstance(max_odds_age, timedelta) or max_odds_age < timedelta(0):
        raise ValueError("max_odds_age must be a nonnegative timedelta")
    eligible = []
    for pick in picks:
        if not isinstance(pick, Pick):
            raise ValueError("picks must contain Pick instances")
        pick.__post_init__()
        if not pick.timestamp_verified and not allow_estimated_timestamps:
            continue
        if (timedelta(0) <= now - pick.captured_at_utc <= max_odds_age
                and pick.kickoff_utc > now and pick.odds <= max_odds):
            eligible.append(pick)
    latest = {}
    for pick in eligible:
        key = (pick.bookmaker, pick.match_id, pick.market, pick.selection)
        previous = latest.get(key)
        if previous is None or (pick.captured_at_utc, pick.snapshot_id) > (previous.captured_at_utc, previous.snapshot_id):
            latest[key] = pick
    eligible = list(latest.values())
    # Full tie-breaker makes output independent of input ordering; identical
    # snapshots cannot create duplicate tickets because match IDs are unique.
    eligible = sorted(set(eligible), key=lambda p: (
        -p.odds * p.p_lower, -p.odds * p.probability,
        p.bookmaker, p.match_id, p.market, p.selection, p.snapshot_id,
        p.captured_at_utc, p.kickoff_utc, p.odds, p.probability, p.p_lower,
        p.home or "", p.away or "", p.league or "",
    ))[:max_candidates]
    if not max_tickets:
        return []

    def ticket(indices: tuple[int, ...]) -> Ticket:
        legs = tuple(eligible[i] for i in indices)
        odds = prod(p.odds for p in legs)
        probability = prod(p.probability for p in legs)
        lower = prod(p.p_lower for p in legs) * (1 - correlation_haircut) ** (len(legs) - 1)
        return Ticket(legs, odds, probability, lower,
                      odds * probability - 1, odds * lower - 1)

    beam = [()]
    qualifying = []
    for size in range(1, max_legs + 1):
        def extensions():
            for indices in beam:
                used = {eligible[i].match_id for i in indices}
                book = eligible[indices[0]].bookmaker if indices else None
                for i in range(indices[-1] + 1 if indices else 0, len(eligible)):
                    pick = eligible[i]
                    if pick.match_id in used or (book is not None and pick.bookmaker != book):
                        continue
                    candidate = indices + (i,)
                    value = ticket(candidate)
                    if value.combined_odds <= max_odds:
                        yield candidate, value

        states = nlargest(beam_width, extensions(),
                          key=lambda state: ((state[1].p_lower, state[1].probability, state[1].robust_ev) if rank_by=="probability" else (state[1].robust_ev, state[1].ev)))
        if not states:
            break
        if size >= min_legs:
            qualifying.extend(value for _, value in states
                              if value.combined_odds >= min_odds
                              and value.robust_ev >= min_robust_ev)
        beam = [indices for indices, _ in states]
    result = []
    used_matches = set()
    for value in sorted(qualifying, key=lambda t: ((-t.p_lower,-t.probability,-t.robust_ev) if rank_by=="probability" else (-t.robust_ev,-t.ev))):
        matches = {p.match_id for p in value.legs}
        if used_matches.isdisjoint(matches):
            result.append(value)
            used_matches.update(matches)
            if len(result) == max_tickets:
                break
    return result
