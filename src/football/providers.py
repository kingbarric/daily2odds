"""API-Football v3 ingestion (https://www.api-football.com/documentation-v3).

APIFootball(api_key, *, session=None, clock=None, team_mapping=None,
            canonical_teams=None, timeout=20, max_requests=100, max_pages=20).
Dates passed to fixtures are inclusive UTC calendar dates (ISO dates or date objects).
Results accepts positive fixture IDs, batched in groups of 20. All methods return
lists of dictionaries and raise ProviderError on transport/provider/schema errors;
no partial batch is returned. Limits apply separately to each public call, with
no automatic retries. Unsupported leagues/markets and non-prematch odds are omitted.
Team mapping is provider name -> canonical name, including explicit identity entries;
unmapped names warn and skip the fixture. When canonical_teams is supplied, mapping
targets must belong to that historical club list. No fuzzy or implicit identity matching.
Odds bookmaker names are provider names; selections follow football.markets.
Only finished fixtures expose regulation goals; other statuses have hg/ag=None.
"""
from datetime import date, datetime, timedelta, timezone
import math
import warnings

import requests

from .data import LEAGUES

# API-Football league IDs -> the existing football-data codes.
LEAGUE_IDS = {39: "E0", 40: "E1", 41: "E2", 42: "E3", 140: "SP1",
              78: "D1", 135: "I1", 61: "F1", 88: "N1", 94: "P1"}
STATUS = {**dict.fromkeys(("TBD", "NS"), "scheduled"),
          **dict.fromkeys(("1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"), "live"),
          **dict.fromkeys(("FT", "AET", "PEN"), "finished"),
          "PST": "postponed", "SUSP": "postponed", "CANC": "cancelled", "ABD": "abandoned"}


class ProviderError(RuntimeError):
    """Request failed, exceeded its budget, or contained unsafe/malformed data."""


class UnknownTeamError(ProviderError):
    """An explicit provider-to-canonical team mapping is required."""


def _utc(value):
    try:
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("timezone required")
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ProviderError("Invalid timezone-aware timestamp") from exc


def _id(value):
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdigit() or int(value) <= 0:
        raise ValueError("fixture IDs must be positive integers")
    return str(int(value))


class APIFootball:
    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, api_key, *, session=None, clock=None, team_mapping=None,
                 canonical_teams=None, timeout=20, max_requests=100, max_pages=20):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if any(type(n) is not int or n <= 0 for n in (max_requests, max_pages)):
            raise ValueError("request and page limits must be positive integers")
        self.session = session if session is not None else requests.Session()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.team_mapping = dict(team_mapping or {})
        if canonical_teams is not None:
            known = set(canonical_teams)
            invalid = [name for name in self.team_mapping.values() if name not in known]
            if invalid:
                raise ValueError(f"Canonical mapping targets absent from historical club list: {invalid!r}")
        self.headers = {"x-apisports-key": api_key}
        self.timeout, self.max_requests, self.max_pages = timeout, max_requests, max_pages

    def _fetch(self, endpoint, params, budget):
        rows, total = [], None
        page = 1
        while True:
            if budget[0] >= self.max_requests:
                raise ProviderError("Request budget exceeded")
            budget[0] += 1
            query = dict(params)
            if endpoint == "odds":
                query["page"] = page
            try:
                response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=query,
                                            headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                # Do not echo request objects/URLs/headers that might contain secrets.
                raise ProviderError("API-Football request failed") from None
            if not isinstance(payload, dict) or payload.get("errors"):
                raise ProviderError("API-Football returned errors or an invalid envelope")
            batch, paging = payload.get("response"), payload.get("paging")
            if not isinstance(batch, list) or not isinstance(paging, dict):
                raise ProviderError("Malformed response/paging")
            current, pages = paging.get("current"), paging.get("total")
            if (type(current) is not int or type(pages) is not int or current != page
                    or pages < 1 or pages > self.max_pages or (total is not None and pages != total)):
                raise ProviderError("Invalid or excessive pagination")
            if endpoint != "odds" and pages != 1:
                raise ProviderError("Unexpected fixture pagination")
            total = pages
            rows.extend(batch)
            if page == total:
                return rows
            page += 1

    def _team(self, name):
        mapped = self.team_mapping.get(name)
        if not isinstance(mapped, str) or not mapped.strip():
            raise UnknownTeamError(f"Explicit canonical mapping required for {name!r}")
        return mapped

    def _fixture(self, row):
        league = LEAGUE_IDS.get(row["league"]["id"])
        if league not in LEAGUES:
            return None
        fixture = row["fixture"]
        status = STATUS.get(fixture["status"]["short"])
        if status is None:
            raise ProviderError("Unsupported fixture status")
        season = row["league"]["season"]
        if type(season) is not int:
            raise ProviderError("Invalid season")
        hg = ag = None
        if status == "finished":
            goals = row["goals"] if fixture["status"]["short"] == "FT" else row["score"]["fulltime"]
            hg, ag = goals["home"], goals["away"]
            if any(type(g) is not int or g < 0 for g in (hg, ag)):
                raise ProviderError("Missing or invalid regulation score")
        return dict(provider_id=_id(fixture["id"]), league=league, season=season,
                    kickoff_utc=_utc(fixture["date"]).isoformat(),
                    home=self._team(row["teams"]["home"]["name"]),
                    away=self._team(row["teams"]["away"]["name"]), status=status, hg=hg, ag=ag)

    def _normalize(self, rows):
        try:
            matches = []
            for row in rows:
                try:
                    match = self._fixture(row)
                except UnknownTeamError as exc:
                    warnings.warn(f"Skipping fixture: {exc}", UserWarning, stacklevel=3)
                    continue
                if match is not None:
                    matches.append(match)
            return matches
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError("Malformed fixture") from exc

    def fixtures(self, start, end):
        """Fetch supported fixtures on inclusive UTC dates; includes changed statuses."""
        def day(value):
            if isinstance(value, datetime):
                return _utc(value).date()
            return value if isinstance(value, date) else date.fromisoformat(value)
        start, end = day(start), day(end)
        days = (end - start).days + 1
        if days <= 0 or days > self.max_requests:
            raise ValueError("Invalid date range or request budget exceeded")
        rows, budget = [], [0]
        for offset in range(days):
            rows.extend(self._fetch("fixtures", {"date": (start + timedelta(days=offset)).isoformat(),
                                                  "timezone": "UTC"}, budget))
        return self._normalize(rows)

    def results(self, ids):
        """Refresh fixture states, including postponed/live/unplayed matches."""
        ids = list(dict.fromkeys(_id(i) for i in ids))
        if len(ids) > 20 * self.max_requests:
            raise ValueError("Fixture IDs exceed request budget")
        rows, budget = [], [0]
        for offset in range(0, len(ids), 20):
            rows.extend(self._fetch("fixtures", {"ids": "-".join(ids[offset:offset + 20]), "timezone": "UTC"}, budget))
        matches = self._normalize(rows)
        if any(m["provider_id"] not in ids for m in matches):
            raise ProviderError("Unexpected fixture ID")
        return matches

    def odds(self, fixture_id):
        """Capture current pre-match quotes; never use the in-play endpoint."""
        fixture_id = _id(fixture_id)
        rows = self._fetch("odds", {"fixture": fixture_id}, [0])
        captured = _utc(self.clock())
        out = []
        try:
            for row in rows:
                fixture = row["fixture"]
                if _id(fixture["id"]) != fixture_id:
                    raise ProviderError("Unexpected odds fixture ID")
                if row.get("live") or fixture.get("live") or fixture.get("status", {}).get("short", "NS") != "NS":
                    continue
                if _utc(fixture["date"]) <= captured:
                    continue
                if LEAGUE_IDS.get(row["league"]["id"]) not in LEAGUES:
                    continue
                updated = _utc(row.get("update"))
                if updated > captured or updated >= _utc(fixture["date"]):
                    raise ProviderError("Provider update is in the future or in-play")
                for book in row["bookmakers"]:
                    if not isinstance(book["name"], str) or not book["name"].strip():
                        raise ProviderError("Invalid bookmaker")
                    for bet in book["bets"]:
                        for value in bet["values"]:
                            market, selection = self._selection(bet, value["value"])
                            if market is None:
                                continue
                            price = float(value["odd"])
                            if isinstance(value["odd"], bool) or not math.isfinite(price) or price <= 1:
                                raise ProviderError("Invalid decimal odds")
                            out.append(dict(bookmaker=book["name"], market=market, selection=selection,
                                            odds=price, source_updated_at=updated.isoformat(),
                                            captured_at_utc=captured.isoformat(), provider_id=fixture_id))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError("Malformed odds") from exc
        return out

    @staticmethod
    def _selection(bet, value):
        # Full-match pre-match IDs only: half-time / extra-time / live IDs differ.
        families = {1: ("Match Winner", "1x2", {"Home": "home", "Draw": "draw", "Away": "away"}),
                    12: ("Double Chance", "double_chance", {"Home/Draw": "home_draw", "Home/Away": "home_away", "Draw/Away": "draw_away"}),
                    8: ("Both Teams Score", "btts", {"Yes": "yes", "No": "no"})}
        if bet.get("id") == 5 and bet.get("name") == "Goals Over/Under":
            choices = {f"{side} {line}": (f"ou_{line}", side.lower())
                       for side in ("Over", "Under") for line in ("1.5", "2.5", "3.5")}
            return choices.get(value, (None, None))
        family = families.get(bet.get("id"))
        if family and bet.get("name") == family[0] and value in family[2]:
            return family[1], family[2][value]
        return None, None
