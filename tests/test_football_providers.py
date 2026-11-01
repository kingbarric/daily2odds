"""Offline provider contracts: no credentials and no network access."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest
import requests

from src.football.providers import APIFootball, ProviderError, LEAGUE_IDS
from src.football.data import LEAGUES

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def fixture(status="NS", ident=123, league=39):
    return {"fixture": {"id": ident, "date": "2026-10-28T16:00:00+01:00", "status": {"short": status}},
            "league": {"id": league, "season": 2026},
            "teams": {"home": {"name": "Manchester United"}, "away": {"name": "Arsenal"}},
            "goals": {"home": 4, "away": 3}, "score": {"fulltime": {"home": 1, "away": 1}}}


def envelope(rows, current=1, total=1, errors=None):
    return {"errors": errors or [], "paging": {"current": current, "total": total}, "response": rows}


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, deepcopy(kwargs)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        class Response:
            def raise_for_status(self):
                if isinstance(item, int):
                    raise requests.HTTPError(str(item))

            def json(self):
                if item == "invalid json":
                    raise ValueError()
                return item
        return Response()


def adapter(*responses, **kwargs):
    session = Session(*responses)
    return APIFootball("test-key", session=session, clock=lambda: NOW,
                      team_mapping={"Manchester United": "Man United", "Arsenal": "Arsenal"}, **kwargs), session


def quote():
    return {"fixture": {"id": 123, "date": "2026-10-28T15:00:00Z"}, "league": {"id": 39},
            "update": "2026-10-27T12:30:00+01:00", "bookmakers": [{"name": "Bet365", "bets": [
                {"id": 1, "name": "Match Winner", "values": [{"value": x, "odd": "2.1"} for x in ("Home", "Draw", "Away")]},
                {"id": 12, "name": "Double Chance", "values": [{"value": x, "odd": "1.3"} for x in ("Home/Draw", "Home/Away", "Draw/Away")]},
                {"id": 5, "name": "Goals Over/Under", "values": [{"value": f"{side} {line}", "odd": "1.9"} for side in ("Over", "Under") for line in ("1.5", "2.5", "3.5", "4.5")]},
                {"id": 8, "name": "Both Teams Score", "values": [{"value": x, "odd": "1.8"} for x in ("Yes", "No")]},
                {"id": 3, "name": "Second Half Winner", "values": [{"value": "Home", "odd": "2.0"}]},
            ]}]}


def test_fixture_contract_dates_leagues_and_transport():
    api, session = adapter(envelope([fixture(), fixture(league=999)]), envelope([]))
    assert api.fixtures("2026-10-27", "2026-10-28") == [dict(
        provider_id="123", league="E0", season=2026, kickoff_utc="2026-10-28T15:00:00+00:00",
        home="Man United", away="Arsenal", status="scheduled", hg=None, ag=None)]
    assert set(LEAGUE_IDS.values()) == set(LEAGUES)
    assert [call[1]["params"]["date"] for call in session.calls] == ["2026-10-27", "2026-10-28"]
    assert session.calls[0][0].endswith("/fixtures")
    assert session.calls[0][1]["headers"] == {"x-apisports-key": "test-key"}
    assert session.calls[0][1]["timeout"] == 20


@pytest.mark.parametrize("status,expected", [("NS", "scheduled"), ("TBD", "scheduled"), ("1H", "live"),
    ("HT", "live"), ("2H", "live"), ("ET", "live"), ("P", "live"), ("BT", "live"), ("INT", "live"),
    ("LIVE", "live"), ("PST", "postponed"), ("SUSP", "postponed"), ("CANC", "cancelled"), ("ABD", "abandoned"),
    ("FT", "finished"), ("AET", "finished"), ("PEN", "finished")])
def test_status_and_regulation_scores(status, expected):
    api, _ = adapter(envelope([fixture(status)]))
    match = api.results([123])[0]
    assert match["status"] == expected
    assert (match["hg"], match["ag"]) == ((4, 3) if status == "FT" else (1, 1) if status in ("AET", "PEN") else (None, None))


def test_unknown_names_and_missing_extra_time_regulation_score_fail():
    row = fixture()
    row["teams"]["home"]["name"] = "Unmapped"
    api, _ = adapter(envelope([row]))
    with pytest.warns(UserWarning, match="Skipping fixture.*Unmapped"):
        assert api.results([123]) == []
    row = fixture("PEN")
    row["score"]["fulltime"]["home"] = None
    api, _ = adapter(envelope([row]))
    with pytest.raises(ProviderError):
        api.results([123])


def test_results_batches_deduplicates_and_empty_has_no_request():
    api, session = adapter(envelope([]), envelope([]))
    assert api.results([]) == []
    api.results(list(range(1, 22)) + [1])
    assert len(session.calls) == 2
    assert session.calls[0][1]["params"]["ids"] == "-".join(map(str, range(1, 21)))
    assert session.calls[1][1]["params"]["ids"] == "21"


def test_all_supported_odds_and_provider_vs_capture_time():
    api, _ = adapter(envelope([quote()]))
    rows = api.odds(123)
    assert len(rows) == 14
    assert {r["market"] for r in rows} == {"1x2", "double_chance", "ou_1.5", "ou_2.5", "ou_3.5", "btts"}
    assert {r["selection"] for r in rows if r["market"] == "double_chance"} == {"home_draw", "home_away", "draw_away"}
    for row in rows:
        assert row["source_updated_at"] == "2026-10-27T11:30:00+00:00"
        assert row["captured_at_utc"] == NOW.isoformat()
        assert row["provider_id"] == "123" and row["bookmaker"] == "Bet365"
        assert isinstance(row["odds"], float)


@pytest.mark.parametrize("updated", [None, "bad", "2026-10-27T11:00:00", "2026-10-27T13:00:00Z"])
def test_update_required_aware_and_not_future(updated):
    row = quote()
    row["update"] = updated
    api, _ = adapter(envelope([row]))
    with pytest.raises(ProviderError):
        api.odds(123)


@pytest.mark.parametrize("price", ["NaN", "inf", "1", "0", "-2", "bad", True])
def test_bad_prices_rejected(price):
    row = quote()
    row["bookmakers"][0]["bets"][0]["values"][0]["odd"] = price
    api, _ = adapter(envelope([row]))
    with pytest.raises(ProviderError):
        api.odds(123)


@pytest.mark.parametrize("change", ["kickoff", "live", "status", "unsupported"])
def test_inplay_and_unsupported_leagues_omitted(change):
    row = quote()
    if change == "kickoff":
        row["fixture"]["date"] = NOW.isoformat()
    elif change == "live":
        row["live"] = True
    elif change == "status":
        row["fixture"]["status"] = {"short": "1H"}
    else:
        row["league"]["id"] = 999
    api, _ = adapter(envelope([row]))
    assert api.odds(123) == []


def test_pagination_and_bounded_calls():
    api, session = adapter(envelope([], 1, 2), envelope([quote()], 2, 2))
    assert len(api.odds(123)) == 14
    assert [c[1]["params"]["page"] for c in session.calls] == [1, 2]
    api, session = adapter(envelope([], 1, 2), max_requests=1)
    with pytest.raises(ProviderError, match="budget"):
        api.odds(123)
    assert len(session.calls) == 1


@pytest.mark.parametrize("response", [429, 500, requests.Timeout(), requests.ConnectionError(), "invalid json",
    envelope([], errors={"token": "invalid"}), {}, envelope([], 2, 2), envelope([], 1, 1000)])
def test_http_json_provider_and_paging_failures(response):
    api, session = adapter(response)
    with pytest.raises(ProviderError):
        api.odds(123)
    assert len(session.calls) == 1


def test_unexpected_ids_and_statuses_rejected():
    api, _ = adapter(envelope([fixture(ident=456)]))
    with pytest.raises(ProviderError):
        api.results([123])
    api, _ = adapter(envelope([fixture("AWD")]))
    with pytest.raises(ProviderError):
        api.results([123])
    row = quote()
    row["fixture"]["id"] = 456
    api, _ = adapter(envelope([row]))
    with pytest.raises(ProviderError):
        api.odds(123)


def test_validation_before_network_and_unknown_market_ids():
    api, session = adapter()
    for ids in ([0], [True], ["1-2"]):
        with pytest.raises(ValueError):
            api.results(ids)
    with pytest.raises(ValueError):
        api.fixtures("2026-10-28", "2026-10-27")
    assert not session.calls
    row = quote()
    row["bookmakers"][0]["bets"] = [{"id": 999, "name": "Match Winner", "values": [{"value": "Home", "odd": "2"}]}]
    api, _ = adapter(envelope([row]))
    assert api.odds(123) == []


def test_canonical_mapping_verified_against_historical_clubs():
    api, _ = adapter(envelope([fixture("FT")]), canonical_teams=["Man United", "Arsenal"])
    assert api.results([123])[0]["hg"] == 4
    with pytest.raises(ValueError, match="historical club list"):
        adapter(canonical_teams=["Arsenal"])
