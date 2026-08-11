"""Ingestion tests.

No network: every source is driven through a fake HTTP client holding canned
payloads in the exact shapes the real feeds return. There's one opt-in live
test at the bottom for when you want to confirm a feed is still up.
"""

import gzip
import io
import json
import os
import urllib.error

import pytest

from pmbot.ingest import (
    AmbiguousPlayer,
    UnknownPlayer,
    build_source,
    default_seasons,
    merge_games,
    write_logs,
)
from pmbot.ingest.base import (
    FetchedLogs,
    PlayerRef,
    as_float,
    build_index,
    fuzzy_candidates,
    name_key,
    normalise_name,
    resolve_one,
)
from pmbot.ingest.hoopr import HoopRSource, file_year, played
from pmbot.ingest.http import FetchError, HttpClient
from pmbot.ingest.mlb_statsapi import MlbStatsApiSource, innings_to_outs
from pmbot.ingest.nba_stats import (
    NbaStatsSource,
    nba_season,
    parse_game_date,
    parse_matchup,
    rows_to_dicts,
)
from pmbot.ingest.nflverse import NflverseSource, week_to_date
from pmbot.ingest.store import SampleDataProtected, path_for, summarise_file


class FakeClient:
    """Stands in for HttpClient; matches canned payloads by URL substring."""

    def __init__(self, json_payloads=None, csv_payloads=None):
        self.json_payloads = json_payloads or {}
        self.csv_payloads = csv_payloads or {}
        self.calls: list[str] = []

    def _match(self, url, table):
        self.calls.append(url)
        for fragment, payload in table.items():
            if fragment in url:
                return payload
        raise FetchError(f"no canned payload for {url}")

    def get_json(self, url, headers=None, ttl=None):
        return self._match(url, self.json_payloads)

    def get_csv(self, url, headers=None, ttl=None):
        return iter(self._match(url, self.csv_payloads))


# ==========================================================================
# name handling
# ==========================================================================
class TestNames:
    def test_accents_and_punctuation_fold_away(self):
        assert normalise_name("Nikola Jokić") == "nikola jokic"
        assert normalise_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"

    def test_suffixes_are_ignored_for_matching(self):
        """Feeds disagree about 'Jr.' constantly; the key must not."""
        assert name_key("Bobby Witt Jr.") == name_key("Bobby Witt")
        assert name_key("Michael Pittman Jr") == name_key("Michael Pittman")

    def test_a_unique_match_resolves(self):
        refs = [PlayerRef("1", "Josh Allen", "nfl", "BUF", "QB")]
        assert resolve_one("Josh Allen", refs).source_id == "1"

    def test_two_players_with_one_name_raise_rather_than_guess(self):
        """There really are two Josh Allens, and picking wrong is silent."""
        refs = [
            PlayerRef("1", "Josh Allen", "nfl", "BUF", "QB"),
            PlayerRef("2", "Josh Allen", "nfl", "JAX", "LB"),
        ]
        with pytest.raises(AmbiguousPlayer) as exc:
            resolve_one("Josh Allen", refs)
        assert "BUF" in str(exc.value) and "JAX" in str(exc.value)

    def test_a_team_breaks_the_tie(self):
        refs = [
            PlayerRef("1", "Josh Allen", "nfl", "BUF", "QB"),
            PlayerRef("2", "Josh Allen", "nfl", "JAX", "LB"),
        ]
        assert resolve_one("Josh Allen", refs, team="buf").source_id == "1"
        assert resolve_one("Josh Allen", refs, position="LB").source_id == "2"

    def test_a_lone_near_match_is_accepted(self):
        """Callers pass fuzzy-search output, so one survivor means one player."""
        refs = [PlayerRef("1", "Patrick Mahomes", "nfl")]
        assert resolve_one("Patrick Mahommes", refs).source_id == "1"

    def test_several_near_matches_and_no_exact_one_suggests_alternatives(self):
        refs = [PlayerRef("1", "Josh Allen", "nfl"), PlayerRef("2", "Josh Allens", "nfl")]
        with pytest.raises(UnknownPlayer) as exc:
            resolve_one("Josh Allan", refs)
        assert "Josh Allen" in str(exc.value)

    def test_nothing_at_all_is_an_unknown_player(self):
        with pytest.raises(UnknownPlayer):
            resolve_one("Someone", [])

    def test_fuzzy_lookup_survives_a_typo(self):
        index = build_index([PlayerRef("1", "Patrick Mahomes", "nfl")])
        assert fuzzy_candidates("Patrick Mahommes", index)
        assert not fuzzy_candidates("Aaron Rodgers", index)


class TestAsFloat:
    @pytest.mark.parametrize(
        "value,expected",
        [(None, 0.0), ("", 0.0), ("NA", 0.0), ("--", 0.0), ("7", 7.0), (3, 3.0), ("12.5", 12.5)],
    )
    def test_junk_becomes_zero_and_numbers_survive(self, value, expected):
        assert as_float(value) == expected

    def test_clock_style_minutes_are_understood(self):
        assert as_float("34:12") == pytest.approx(34.2, abs=0.01)

    def test_unparseable_text_falls_back(self):
        assert as_float("DNP-CD", default=-1.0) == -1.0


# ==========================================================================
# http client
# ==========================================================================
class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestHttpClient:
    def test_responses_are_cached_so_a_refetch_is_free(self, tmp_path, monkeypatch):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(request.full_url)
            return FakeResponse(b'{"ok": true}')

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        client = HttpClient(cache_dir=tmp_path, min_interval=0)
        assert client.get_json("https://example.test/a") == {"ok": True}
        assert client.get_json("https://example.test/a") == {"ok": True}
        assert len(calls) == 1

    def test_gzipped_payloads_are_unwrapped(self, tmp_path, monkeypatch):
        body = gzip.compress(b"col\n1\n")
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda request, timeout=None: FakeResponse(body)
        )
        client = HttpClient(cache_dir=tmp_path, min_interval=0)
        assert list(client.get_csv("https://example.test/x.csv")) == [{"col": "1"}]

    def test_a_404_is_not_retried(self, tmp_path, monkeypatch):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b"nope"))

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        client = HttpClient(cache_dir=tmp_path, min_interval=0)
        with pytest.raises(FetchError, match="404"):
            client.get_bytes("https://example.test/missing")
        assert len(calls) == 1  # a missing player will still be missing next time

    def test_a_500_is_retried_then_reported(self, tmp_path, monkeypatch):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, 503, "Down", {}, io.BytesIO(b"later"))

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        monkeypatch.setattr("time.sleep", lambda _: None)
        client = HttpClient(cache_dir=tmp_path, min_interval=0)
        with pytest.raises(FetchError, match="503"):
            client.get_bytes("https://example.test/flaky", attempts=3)
        assert len(calls) == 3

    def test_offline_mode_refuses_to_reach_out(self, tmp_path):
        client = HttpClient(cache_dir=tmp_path, offline=True)
        with pytest.raises(FetchError, match="offline"):
            client.get_bytes("https://example.test/anything")

    def test_offline_mode_still_serves_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda request, timeout=None: FakeResponse(b'{"v": 1}')
        )
        HttpClient(cache_dir=tmp_path, min_interval=0).get_json("https://example.test/v")
        offline = HttpClient(cache_dir=tmp_path, offline=True)
        assert offline.get_json("https://example.test/v") == {"v": 1}

    def test_garbage_is_not_cached_as_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda request, timeout=None: FakeResponse(b"<html>nope")
        )
        client = HttpClient(cache_dir=tmp_path, min_interval=0)
        with pytest.raises(FetchError, match="did not return JSON"):
            client.get_json("https://example.test/html")
        assert not list(tmp_path.glob("*.bin"))


# ==========================================================================
# NFL / nflverse
# ==========================================================================
NFL_WEEKLY = [
    {
        "player_id": "00-0034857", "player_display_name": "Josh Allen", "position": "QB",
        "season": "2024", "week": "1", "season_type": "REG", "team": "BUF", "opponent_team": "ARI",
        "completions": "18", "attempts": "23", "passing_yards": "232", "passing_tds": "2",
        "carries": "9", "rushing_yards": "39", "rushing_tds": "1",
        "targets": "0", "receptions": "0", "receiving_yards": "0", "receiving_tds": "0",
    },
    {
        "player_id": "00-0034857", "player_display_name": "Josh Allen", "position": "QB",
        "season": "2024", "week": "2", "season_type": "REG", "team": "BUF", "opponent_team": "MIA",
        "completions": "13", "attempts": "19", "passing_yards": "139", "passing_tds": "1",
        "carries": "6", "rushing_yards": "2", "rushing_tds": "0",
        "targets": "", "receptions": "", "receiving_yards": "", "receiving_tds": "",
    },
    {
        "player_id": "00-0034857", "player_display_name": "Josh Allen", "position": "QB",
        "season": "2024", "week": "1", "season_type": "PRE", "team": "BUF", "opponent_team": "CHI",
        "completions": "0", "attempts": "0", "passing_yards": "0", "passing_tds": "0",
        "carries": "0", "rushing_yards": "0", "rushing_tds": "0",
        "targets": "0", "receptions": "0", "receiving_yards": "0", "receiving_tds": "0",
    },
    {
        "player_id": "00-0036358", "player_display_name": "CeeDee Lamb", "position": "WR",
        "season": "2024", "week": "1", "season_type": "REG", "team": "DAL", "opponent_team": "CLE",
        "completions": "0", "attempts": "0", "passing_yards": "0", "passing_tds": "0",
        "carries": "0", "rushing_yards": "0", "rushing_tds": "0",
        "targets": "10", "receptions": "5", "receiving_yards": "61", "receiving_tds": "0",
    },
]

NFL_SCHEDULE = [
    {"season": "2024", "week": "1", "gameday": "2024-09-08", "home_team": "BUF", "away_team": "ARI"},
    {"season": "2024", "week": "2", "gameday": "2024-09-12", "home_team": "MIA", "away_team": "BUF"},
    {"season": "2024", "week": "1", "gameday": "2024-09-08", "home_team": "CLE", "away_team": "DAL"},
]


@pytest.fixture
def nfl_source():
    client = FakeClient(csv_payloads={
        "stats_player_week_2024": NFL_WEEKLY,
        "schedules/games.csv": NFL_SCHEDULE,
    })
    return NflverseSource(client=client)


class TestNflverse:
    def test_fields_map_onto_pmbot_markets(self, nfl_source):
        logs = nfl_source.fetch("Josh Allen", [2024])
        week1 = logs.games[0]
        assert week1["pass_attempts"] == 23
        assert week1["pass_yards"] == 232
        assert week1["pass_tds"] == 2
        assert week1["rush_attempts"] == 9
        assert week1["rush_yards"] == 39

    def test_the_schedule_supplies_real_dates_and_venue(self, nfl_source):
        games = nfl_source.fetch("Josh Allen", [2024]).games
        assert games[0]["date"] == "2024-09-08"
        assert games[0]["home"] is True  # BUF hosted ARI
        assert games[1]["date"] == "2024-09-12"
        assert games[1]["home"] is False  # BUF at MIA

    def test_preseason_is_excluded_by_default(self, nfl_source):
        games = nfl_source.fetch("Josh Allen", [2024]).games
        assert len(games) == 2
        assert all(g["opponent"] != "CHI" for g in games)

    def test_preseason_can_be_asked_for(self):
        client = FakeClient(csv_payloads={
            "stats_player_week_2024": NFL_WEEKLY, "schedules/games.csv": NFL_SCHEDULE,
        })
        source = NflverseSource(client=client, season_types=("REG", "PRE"))
        assert len(source.fetch("Josh Allen", [2024]).games) == 3

    def test_empty_cells_become_zero_not_a_crash(self, nfl_source):
        assert nfl_source.fetch("Josh Allen", [2024]).games[1]["receptions"] == 0.0

    def test_games_come_back_oldest_first(self, nfl_source):
        games = nfl_source.fetch("CeeDee Lamb", [2024]).games
        assert games == sorted(games, key=lambda g: g["date"])

    def test_the_right_player_is_picked_out(self, nfl_source):
        lamb = nfl_source.fetch("CeeDee Lamb", [2024])
        assert lamb.ref.source_id == "00-0036358"
        assert lamb.games[0]["targets"] == 10

    def test_an_unknown_player_is_reported(self, nfl_source):
        with pytest.raises(UnknownPlayer):
            nfl_source.fetch("Nobody Here", [2024])

    def test_an_unpublished_season_does_not_sink_the_published_one(self):
        """Before week 1 the new season file 404s. That must not skip everyone."""
        class Missing(FakeClient):
            def get_csv(self, url, headers=None, ttl=None):
                if "stats_player_week_2025" in url:
                    raise FetchError("HTTP 404 for stats_player_week_2025.csv: Not Found")
                return super().get_csv(url, headers, ttl)

        source = NflverseSource(client=Missing(csv_payloads={
            "stats_player_week_2024": NFL_WEEKLY, "schedules/games.csv": NFL_SCHEDULE,
        }))
        logs = source.fetch("Josh Allen", [2024, 2025])
        assert len(logs.games) == 2  # the 2024 season still came through

    def test_no_data_at_all_says_so_rather_than_blaming_the_name(self):
        class AllMissing(FakeClient):
            def get_csv(self, url, headers=None, ttl=None):
                if "stats_player_week" in url:
                    raise FetchError("HTTP 404: Not Found")
                return super().get_csv(url, headers, ttl)

        source = NflverseSource(client=AllMissing(csv_payloads={"schedules/games.csv": NFL_SCHEDULE}))
        with pytest.raises(FetchError, match="not published yet"):
            source.fetch("Josh Allen", [2025, 2026])

    def test_a_non_404_error_still_propagates(self):
        class Broken(FakeClient):
            def get_csv(self, url, headers=None, ttl=None):
                raise FetchError("HTTP 500 for stats_player_week_2024.csv")

        with pytest.raises(FetchError, match="500"):
            NflverseSource(client=Broken()).fetch("Josh Allen", [2024])

    def test_week_dates_preserve_ordering_without_a_schedule(self):
        assert week_to_date(2024, 1) < week_to_date(2024, 2) < week_to_date(2024, 18)


# ==========================================================================
# MLB / statsapi
# ==========================================================================
MLB_PLAYERS = {"people": [
    {"id": 592450, "fullName": "Aaron Judge",
     "currentTeam": {"id": 147, "abbreviation": "NYY"},
     "primaryPosition": {"abbreviation": "RF"}},
    {"id": 669373, "fullName": "Tarik Skubal",
     "currentTeam": {"id": 116, "abbreviation": "DET"},
     "primaryPosition": {"abbreviation": "P"}},
]}

MLB_TEAMS = {"teams": [
    {"id": 147, "abbreviation": "NYY"}, {"id": 111, "abbreviation": "BOS"},
    {"id": 116, "abbreviation": "DET"}, {"id": 114, "abbreviation": "CLE"},
]}

MLB_HITTING = {"stats": [{"splits": [
    {"date": "2025-04-01", "isHome": True, "opponent": {"id": 111},
     "game": {"gamePk": 1}, "stat": {
         "plateAppearances": 5, "atBats": 4, "hits": 2, "totalBases": 5,
         "homeRuns": 1, "rbi": 3, "runs": 2, "stolenBases": 0,
         "strikeOuts": 1, "baseOnBalls": 1, "doubles": 0, "triples": 0}},
    {"date": "2025-04-02", "isHome": False, "opponent": {"id": 111},
     "game": {"gamePk": 2}, "stat": {
         "plateAppearances": 4, "atBats": 4, "hits": 0, "totalBases": 0,
         "homeRuns": 0, "rbi": 0, "runs": 0, "stolenBases": 0,
         "strikeOuts": 2, "baseOnBalls": 0, "doubles": 0, "triples": 0}},
]}]}

MLB_PITCHING = {"stats": [{"splits": [
    {"date": "2025-04-03", "isHome": True, "opponent": {"id": 114},
     "game": {"gamePk": 9}, "stat": {
         "battersFaced": 25, "strikeOuts": 8, "hits": 4, "earnedRuns": 2,
         "baseOnBalls": 1, "homeRuns": 1, "inningsPitched": "6.1"}},
]}]}


def mlb_source(hitting=MLB_HITTING, pitching=MLB_PITCHING):
    client = FakeClient(json_payloads={
        "sports/1/players": MLB_PLAYERS,
        "teams?sportId=1": MLB_TEAMS,
        "group=hitting": hitting,
        "group=pitching": pitching,
    })
    return MlbStatsApiSource(client=client)


class TestMlbStatsApi:
    def test_hitting_fields_map_onto_pmbot_markets(self):
        games = mlb_source().fetch("Aaron Judge", [2025]).games
        first = games[0]
        assert first["plate_appearances"] == 5
        assert first["hits"] == 2
        assert first["total_bases"] == 5
        assert first["home_runs"] == 1
        assert first["rbis"] == 3

    def test_pitching_fields_map_onto_pmbot_markets(self):
        games = mlb_source().fetch("Tarik Skubal", [2025]).games
        pitched = next(g for g in games if g.get("batters_faced"))
        assert pitched["batters_faced"] == 25
        assert pitched["strikeouts"] == 8
        assert pitched["hits_allowed"] == 4
        assert pitched["earned_runs"] == 2

    def test_innings_pitched_is_not_a_decimal(self):
        """'6.1' is six innings and one out. Reading it as 6.1 corrupts everything."""
        assert innings_to_outs("6.1") == 19
        assert innings_to_outs("6.2") == 20
        assert innings_to_outs("7.0") == 21
        assert innings_to_outs("0.1") == 1
        assert innings_to_outs("") == 0

    def test_outs_are_derived_from_innings(self):
        games = mlb_source().fetch("Tarik Skubal", [2025]).games
        assert next(g for g in games if g.get("batters_faced"))["outs"] == 19

    def test_a_two_way_player_gets_one_row_per_game(self):
        """Ohtani's hitting and pitching lines belong on the same game."""
        both = {"stats": [{"splits": [
            {"date": "2025-04-03", "isHome": True, "opponent": {"id": 114},
             "game": {"gamePk": 9},
             "stat": {"plateAppearances": 4, "hits": 2, "totalBases": 6, "homeRuns": 1,
                      "rbi": 2, "runs": 1, "stolenBases": 0, "atBats": 4,
                      "strikeOuts": 1, "baseOnBalls": 0, "doubles": 0, "triples": 0}},
        ]}]}
        games = mlb_source(hitting=both).fetch("Tarik Skubal", [2025]).games
        assert len(games) == 1
        assert games[0]["hits"] == 2 and games[0]["strikeouts"] == 8

    def test_opponent_ids_become_abbreviations(self):
        assert mlb_source().fetch("Aaron Judge", [2025]).games[0]["opponent"] == "BOS"

    def test_home_and_away_survive(self):
        games = mlb_source().fetch("Aaron Judge", [2025]).games
        assert games[0]["home"] is True and games[1]["home"] is False


# ==========================================================================
# NBA / stats.nba.com
# ==========================================================================
NBA_PLAYERS = {"resultSets": [{
    "headers": ["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ABBREVIATION"],
    "rowSet": [[1630162, "Anthony Edwards", "MIN"], [1628369, "Jayson Tatum", "BOS"]],
}]}

NBA_LOG = {"resultSets": [{
    "headers": ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "FG3M", "BLK", "STL", "TOV",
                "OREB", "DREB", "FGA", "FTA"],
    "rowSet": [
        ["OCT 25, 2024", "MIN vs. SAC", 36, 31, 5, 4, 4, 1, 2, 3, 1, 4, 22, 8],
        ["OCT 27, 2024", "MIN @ DEN", 34, 24, 7, 6, 3, 0, 1, 2, 2, 5, 20, 5],
    ],
}]}


def nba_source():
    client = FakeClient(json_payloads={"commonallplayers": NBA_PLAYERS, "playergamelog": NBA_LOG})
    return NbaStatsSource(client=client, season_types=("Regular Season",))


class TestNbaStats:
    def test_column_oriented_replies_become_dicts(self):
        rows = rows_to_dicts(NBA_PLAYERS)
        assert rows[0]["DISPLAY_FIRST_LAST"] == "Anthony Edwards"
        assert rows_to_dicts({}) == []

    @pytest.mark.parametrize(
        "matchup,expected",
        [("MIN vs. SAC", ("MIN", "SAC", True)), ("MIN @ DEN", ("MIN", "DEN", False))],
    )
    def test_matchup_strings_carry_the_venue(self, matchup, expected):
        assert parse_matchup(matchup) == expected

    def test_nba_dates_are_normalised(self):
        assert parse_game_date("OCT 25, 2024") == "2024-10-25"
        assert parse_game_date("2024-10-25") == "2024-10-25"

    @pytest.mark.parametrize(
        "given,expected",
        [(2024, "2024-25"), ("2024", "2024-25"), ("2024-25", "2024-25"), (1999, "1999-00")],
    )
    def test_season_labels_are_built_correctly(self, given, expected):
        """NBA seasons are named for the year they start. Easy off-by-one."""
        assert nba_season(given) == expected

    def test_fields_map_onto_pmbot_markets(self):
        games = nba_source().fetch("Anthony Edwards", [2024]).games
        assert games[0]["minutes"] == 36
        assert games[0]["points"] == 31
        assert games[0]["rebounds"] == 5
        assert games[0]["assists"] == 4
        assert games[0]["threes"] == 4

    def test_venue_and_opponent_come_from_the_matchup(self):
        games = nba_source().fetch("Anthony Edwards", [2024]).games
        assert (games[0]["opponent"], games[0]["home"]) == ("SAC", True)
        assert (games[1]["opponent"], games[1]["home"]) == ("DEN", False)

    def test_the_endpoint_gets_the_headers_it_demands(self):
        """Without the nba.com headers the endpoint returns nothing at all."""
        source = nba_source()
        source.fetch("Anthony Edwards", [2024])
        from pmbot.ingest.nba_stats import HEADERS

        assert HEADERS["x-nba-stats-origin"] == "stats"
        assert "nba.com" in HEADERS["Referer"]


# ==========================================================================
# NBA / hoopR release CSVs
# ==========================================================================
def hoopr_row(**kw):
    row = {
        "game_id": "1", "season": "2025", "season_type": "2", "game_date": "2024-10-22",
        "athlete_id": "4594268", "athlete_display_name": "Anthony Edwards",
        "athlete_position_abbreviation": "SG", "team_abbreviation": "MIN",
        "opponent_team_abbreviation": "LAL", "home_away": "away",
        "minutes": "36.0", "points": "27", "rebounds": "6", "assists": "5",
        "three_point_field_goals_made": "4", "blocks": "1", "steals": "2", "turnovers": "3",
        "offensive_rebounds": "1", "defensive_rebounds": "5",
        "field_goals_attempted": "20", "free_throws_attempted": "6",
        "did_not_play": "false", "active": "true", "starter": "true",
    }
    row.update(kw)
    return row


HOOPR_ROWS = [
    hoopr_row(),
    # ESPN marks plenty of played games active=false. It is a roster flag, not
    # an appearance flag, and filtering on it drops ~40% of a real season.
    hoopr_row(game_date="2024-10-24", active="false", points="31", home_away="home"),
    hoopr_row(game_date="2024-10-26", did_not_play="true", minutes="", points=""),
    hoopr_row(game_date="2025-04-20", season_type="3", points="22"),   # playoffs
    hoopr_row(game_date="2024-10-05", season_type="1", points="8"),    # preseason
    hoopr_row(athlete_id="3112335", athlete_display_name="Nikola Jokic",
              team_abbreviation="DEN", points="35", rebounds="14", assists="11"),
]


def hoopr_source(season_types=("REG", "POST")):
    client = FakeClient(csv_payloads={"player_box_2025.csv": HOOPR_ROWS})
    return HoopRSource(client=client, season_types=season_types)


class TestHoopR:
    def test_season_is_labelled_by_its_starting_year(self):
        """hoopR files are named for the year a season ends; we aren't."""
        assert file_year(2024) == 2025
        assert file_year(2025) == 2026

    def test_fields_map_onto_pmbot_markets(self):
        game = hoopr_source().fetch("Anthony Edwards", [2024]).games[0]
        assert game["minutes"] == 36.0
        assert game["points"] == 27
        assert game["rebounds"] == 6
        assert game["assists"] == 5
        assert game["threes"] == 4
        assert (game["opponent"], game["home"]) == ("LAL", False)

    def test_games_marked_inactive_are_still_games(self):
        """The regression that silently deleted 40% of a season."""
        games = hoopr_source().fetch("Anthony Edwards", [2024]).games
        assert any(g["date"] == "2024-10-24" and g["points"] == 31 for g in games)

    def test_dnps_are_dropped(self):
        games = hoopr_source().fetch("Anthony Edwards", [2024]).games
        assert all(g["date"] != "2024-10-26" for g in games)
        assert all(g["minutes"] > 0 for g in games)

    def test_played_reads_the_right_columns(self):
        assert played(hoopr_row()) is True
        assert played(hoopr_row(active="false")) is True
        assert played(hoopr_row(did_not_play="true")) is False
        assert played(hoopr_row(minutes="")) is False

    def test_regular_season_and_playoffs_are_kept_preseason_is_not(self):
        games = hoopr_source().fetch("Anthony Edwards", [2024]).games
        dates = {g["date"] for g in games}
        assert "2025-04-20" in dates      # playoffs count
        assert "2024-10-05" not in dates  # preseason does not

    def test_players_are_kept_apart(self):
        jokic = hoopr_source().fetch("Nikola Jokic", [2024])
        assert jokic.ref.source_id == "3112335"
        assert len(jokic.games) == 1 and jokic.games[0]["rebounds"] == 14

    def test_accented_spellings_find_the_same_player(self):
        assert hoopr_source().fetch("Nikola Jokić", [2024]).ref.source_id == "3112335"

    def test_games_come_back_oldest_first(self):
        games = hoopr_source().fetch("Anthony Edwards", [2024]).games
        assert games == sorted(games, key=lambda g: g["date"])

    def test_a_season_that_has_not_tipped_off_is_skipped(self):
        """Searching only the newest season would report everyone as unknown."""
        class Missing(FakeClient):
            def get_csv(self, url, headers=None, ttl=None):
                if "player_box_2026" in url:
                    raise FetchError("HTTP 404 for player_box_2026.csv: Not Found")
                return super().get_csv(url, headers, ttl)

        source = HoopRSource(client=Missing(csv_payloads={"player_box_2025.csv": HOOPR_ROWS}))
        assert source.fetch("Anthony Edwards", [2024, 2025]).games

    def test_no_data_at_all_says_so(self):
        class AllMissing(FakeClient):
            def get_csv(self, url, headers=None, ttl=None):
                raise FetchError("HTTP 404: Not Found")

        with pytest.raises(FetchError, match="not published yet"):
            HoopRSource(client=AllMissing()).fetch("Anthony Edwards", [2025])


# ==========================================================================
# storage
# ==========================================================================
def fetched(games, source="nflverse", player="Josh Allen"):
    return FetchedLogs(
        player=player, sport="nfl", source=source,
        ref=PlayerRef("00-1", player, "nfl", "BUF", "QB"), games=games,
    )


class TestStore:
    def test_logs_land_where_the_models_look(self, tmp_path):
        path = write_logs(fetched([{"date": "2024-09-08", "pass_yards": 232, "team": "BUF"}]), tmp_path)
        assert path == path_for(tmp_path, "nfl", "Josh Allen")
        payload = json.loads(path.read_text())
        assert payload["player"] == "Josh Allen"
        assert payload["source"] == "nflverse"
        assert payload["team"] == "BUF"
        assert len(payload["games"]) == 1

    def test_a_second_fetch_merges_rather_than_truncates(self, tmp_path):
        write_logs(fetched([{"date": "2024-09-08", "pass_yards": 232}]), tmp_path)
        write_logs(fetched([{"date": "2024-09-15", "pass_yards": 263}]), tmp_path)
        games = json.loads(path_for(tmp_path, "nfl", "Josh Allen").read_text())["games"]
        assert [g["date"] for g in games] == ["2024-09-08", "2024-09-15"]

    def test_a_corrected_stat_line_wins(self, tmp_path):
        write_logs(fetched([{"date": "2024-09-08", "pass_yards": 999}]), tmp_path)
        write_logs(fetched([{"date": "2024-09-08", "pass_yards": 232}]), tmp_path)
        games = json.loads(path_for(tmp_path, "nfl", "Josh Allen").read_text())["games"]
        assert len(games) == 1 and games[0]["pass_yards"] == 232

    def test_feeds_are_never_blended(self, tmp_path):
        """Mixing sources produces a file that looks fine and models nothing."""
        write_logs(fetched([{"date": "2026-01-01", "pass_yards": 1}], source="synthetic"), tmp_path)
        write_logs(fetched([{"date": "2024-09-08", "pass_yards": 232}], source="nflverse"), tmp_path)
        payload = json.loads(path_for(tmp_path, "nfl", "Josh Allen").read_text())
        assert payload["source"] == "nflverse"
        assert [g["date"] for g in payload["games"]] == ["2024-09-08"]

    def test_an_unlabelled_file_is_treated_as_a_foreign_feed(self, tmp_path):
        """No recorded source means unknown provenance, so don't merge into it."""
        target = path_for(tmp_path, "nfl", "Josh Allen")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "player": "Josh Allen", "games": [{"date": "2026-01-01", "pass_yards": 1}],
        }))
        write_logs(fetched([{"date": "2024-09-08", "pass_yards": 232}]), tmp_path)
        payload = json.loads(target.read_text())
        assert [g["date"] for g in payload["games"]] == ["2024-09-08"]

    def test_replace_drops_the_history(self, tmp_path):
        write_logs(fetched([{"date": "2024-09-08", "pass_yards": 232}]), tmp_path)
        write_logs(fetched([{"date": "2024-09-15", "pass_yards": 263}]), tmp_path, merge=False)
        games = json.loads(path_for(tmp_path, "nfl", "Josh Allen").read_text())["games"]
        assert [g["date"] for g in games] == ["2024-09-15"]

    def test_merge_sorts_and_deduplicates(self):
        merged = merge_games(
            [{"date": "2024-09-15", "v": 1}, {"date": "2024-09-08", "v": 2}],
            [{"date": "2024-09-15", "v": 3}],
        )
        assert [(g["date"], g["v"]) for g in merged] == [("2024-09-08", 2), ("2024-09-15", 3)]

    def test_written_files_load_back_through_the_normal_provider(self, tmp_path):
        from pmbot.providers import LocalGameLogProvider

        write_logs(
            fetched([
                {"date": "2024-09-08", "pass_attempts": 23, "pass_yards": 232,
                 "opponent": "ARI", "home": True, "team": "BUF"},
            ]),
            tmp_path,
        )
        logs = LocalGameLogProvider(tmp_path).logs_for("Josh Allen", "nfl")
        assert len(logs) == 1
        assert logs[0].stats["pass_yards"] == 232
        assert logs[0].opponent == "ARI" and logs[0].is_home
        assert "home" not in logs[0].stats  # a bool is not a stat

    def test_the_synthetic_fixture_cannot_be_overwritten(self, tmp_path):
        """The guard must sit here, not in the CLI: the scheduler writes too.

        It didn't, once, and the daily job promptly replaced four synthetic
        NBA fixtures with real logs -- which looked fine until the sample
        slate started pricing real form against invented lines.
        """
        sample = tmp_path / "data" / "sample" / "gamelogs"
        with pytest.raises(SampleDataProtected, match="refusing to write"):
            write_logs(fetched([{"date": "2024-09-08"}]), sample)
        assert not (sample / "nfl").exists()

    def test_the_fixture_generator_can_still_write_there(self, tmp_path):
        sample = tmp_path / "data" / "sample" / "gamelogs"
        path = write_logs(fetched([{"date": "2024-09-08"}]), sample, allow_sample=True)
        assert path.exists()

    def test_ordinary_paths_are_unaffected(self, tmp_path):
        assert write_logs(fetched([{"date": "2024-09-08"}]), tmp_path / "gamelogs").exists()

    def test_summary_line_is_informative(self, tmp_path):
        path = write_logs(fetched([{"date": "2024-09-08"}, {"date": "2024-12-01"}]), tmp_path)
        summary = summarise_file(path)
        assert "Josh Allen" in summary and "2 games" in summary and "nflverse" in summary


class TestRegistry:
    @pytest.mark.parametrize("sport,expected", [("nfl", "nflverse"), ("mlb", "mlb-statsapi"), ("nba", "hoopr")])
    def test_every_sport_has_a_default_source(self, sport, expected):
        assert build_source(sport).name == expected

    def test_alternative_sources_can_be_asked_for(self):
        assert build_source("nba", "nba-stats").name == "nba-stats"

    def test_unknown_sports_and_sources_are_rejected(self):
        with pytest.raises(KeyError):
            build_source("cricket")
        with pytest.raises(KeyError, match="unknown nfl source"):
            build_source("nfl", "my-spreadsheet")

    def test_default_seasons_are_recent_and_ordered(self):
        seasons = default_seasons("nfl", count=3)
        assert len(seasons) == 3
        assert seasons == sorted(seasons)


# ==========================================================================
# opt-in live check
# ==========================================================================
live = pytest.mark.skipif(
    os.environ.get("PMBOT_LIVE_TESTS") != "1",
    reason="set PMBOT_LIVE_TESTS=1 to hit the real feeds",
)


@live
def test_live_nflverse_fetch(tmp_path):
    source = build_source("nfl", client=HttpClient(cache_dir=tmp_path))
    logs = source.fetch("Josh Allen", [2024])
    assert len(logs.games) > 10
    assert all(g["pass_attempts"] >= 0 for g in logs.games)
    assert logs.ref.team == "BUF"


@live
def test_live_hoopr_fetch(tmp_path):
    source = build_source("nba", client=HttpClient(cache_dir=tmp_path))
    logs = source.fetch("Anthony Edwards", [2024])
    # A full season plus playoffs; anything near 55 means the DNP filter has
    # started eating real games again.
    assert len(logs.games) > 80
    assert logs.ref.team == "MIN"
    assert 20 < sum(g["points"] for g in logs.games) / len(logs.games) < 35
