import json
from datetime import datetime, timezone

import pytest

from pmbot.config import ApiSettings, Settings
from pmbot.providers import (
    LocalGameLogProvider,
    LocalOddsProvider,
    MatchupBook,
    build_odds_provider,
    slugify,
)
from pmbot.providers.theoddsapi import OddsApiError, TheOddsApiProvider, parse_event_odds, to_snapshot


class TestSlugify:
    @pytest.mark.parametrize(
        "name,slug",
        [
            ("Anthony Edwards", "anthony-edwards"),
            ("Shai Gilgeous-Alexander", "shai-gilgeous-alexander"),
            ("Bobby Witt Jr.", "bobby-witt-jr"),
            ("  Luka  Doncic ", "luka-doncic"),
        ],
    )
    def test_names_become_stable_keys(self, name, slug):
        assert slugify(name) == slug

    def test_accents_fold_so_feeds_agree(self):
        """One feed writes Jokić, another writes Jokic. Same player."""
        assert slugify("Nikola Jokić") == slugify("Nikola Jokic")


class TestLocalProviders:
    def test_props_load_from_the_fixture_slate(self):
        props = LocalOddsProvider("data/props").player_props("nba", ["player_points"])
        assert props
        assert all(p.market == "player_points" for p in props)
        assert all(p.two_way_books for p in props)

    def test_markets_are_filtered(self):
        props = LocalOddsProvider("data/props").player_props("nba", ["player_rebounds"])
        assert {p.market for p in props} == {"player_rebounds"}

    def test_events_can_be_filtered(self):
        provider = LocalOddsProvider("data/props")
        everything = provider.player_props("nba", ["player_points"])
        one = everything[0].event_id
        assert {p.event_id for p in provider.player_props("nba", ["player_points"], [one])} == {one}

    def test_a_stored_slate_is_replayed_as_tonight(self):
        """A fixture checked in months ago must still scan."""
        props = LocalOddsProvider("data/props").player_props("nba", ["player_points"])
        assert all(p.commence_time > datetime.now(timezone.utc) for p in props)

    def test_replay_can_be_turned_off(self, tmp_path):
        (tmp_path / "nba_old.json").write_text(json.dumps({
            "sport": "nba",
            "events": [{
                "id": "e", "commence_time": "2020-01-01T00:00:00Z",
                "home_team": "A", "away_team": "B",
                "props": [{"player": "X", "market": "player_points",
                           "books": [{"book": "pinnacle", "line": 20.5, "over": -110, "under": -110}]}],
            }],
        }))
        props = LocalOddsProvider(tmp_path, replay=False).player_props("nba", ["player_points"])
        assert props[0].commence_time.year == 2020

    def test_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert LocalOddsProvider(tmp_path / "nope").player_props("nba", ["player_points"]) == []

    def test_game_logs_load_and_cache(self):
        provider = LocalGameLogProvider("data/gamelogs")
        logs = provider.logs_for("Anthony Edwards", "nba")
        assert len(logs) > 20
        assert logs is provider.logs_for("Anthony Edwards", "nba")  # cached
        assert all("minutes" in g.stats for g in logs)

    def test_an_unknown_player_has_no_logs(self):
        assert LocalGameLogProvider("data/gamelogs").logs_for("Nobody", "nba") == []

    def test_known_players_are_listable(self):
        assert "Anthony Edwards" in LocalGameLogProvider("data/gamelogs").known_players("nba")


class TestMatchupBook:
    def test_factors_are_applied_and_explained(self):
        book = MatchupBook("data/defense.json")
        context = book.context_for(sport="nba", market="player_points", opponent="MIN", is_home=False)
        assert context.opponent_factor < 1.0  # MIN is a tough defence in the fixture
        assert context.notes

    def test_unknown_teams_get_no_adjustment(self):
        book = MatchupBook("data/defense.json")
        context = book.context_for(sport="nba", market="player_points", opponent="ZZZ", is_home=None)
        assert context.combined == pytest.approx(1.0)

    def test_home_and_away_are_mirror_images(self):
        book = MatchupBook("data/defense.json")
        home = book.context_for(sport="nba", market="player_points", opponent="ZZZ", is_home=True)
        away = book.context_for(sport="nba", market="player_points", opponent="ZZZ", is_home=False)
        assert home.home_factor * away.home_factor == pytest.approx(1.0)

    def test_a_missing_file_means_no_adjustments(self, tmp_path):
        book = MatchupBook(tmp_path / "absent.json")
        assert book.context_for(
            sport="nba", market="player_points", opponent="MIN", is_home=True
        ).combined == pytest.approx(1.0)


EVENT_PAYLOAD = {
    "id": "abc123",
    "commence_time": "2026-08-11T23:40:00Z",
    "home_team": "Minnesota Timberwolves",
    "away_team": "Sacramento Kings",
    "bookmakers": [
        {
            "key": "pinnacle",
            "markets": [
                {
                    "key": "player_points",
                    "last_update": "2026-08-11T21:00:00Z",
                    "outcomes": [
                        {"name": "Over", "description": "Anthony Edwards", "price": -112, "point": 27.5},
                        {"name": "Under", "description": "Anthony Edwards", "price": -108, "point": 27.5},
                        {"name": "Over", "description": "Rudy Gobert", "price": 105, "point": 12.5},
                        {"name": "Under", "description": "Rudy Gobert", "price": -125, "point": 12.5},
                    ],
                }
            ],
        },
        {
            "key": "draftkings",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {"name": "Over", "description": "Anthony Edwards", "price": 130, "point": 27.5},
                        {"name": "Under", "description": "Anthony Edwards", "price": -160, "point": 27.5},
                        # A yes/no market shares the endpoint and must be ignored.
                        {"name": "Yes", "description": "Anthony Edwards", "price": -140},
                    ],
                }
            ],
        },
    ],
}


class TestOddsApiParsing:
    def test_outcomes_regroup_into_two_way_markets(self):
        props = parse_event_odds(EVENT_PAYLOAD, "nba")
        by_player = {p.player: p for p in props}
        assert set(by_player) == {"Anthony Edwards", "Rudy Gobert"}

        edwards = by_player["Anthony Edwards"]
        assert len(edwards.books) == 2
        assert {b.book for b in edwards.books} == {"pinnacle", "draftkings"}
        assert all(b.is_two_way for b in edwards.books)
        assert edwards.event_id == "abc123"
        assert edwards.home_team == "Minnesota Timberwolves"

    def test_prices_land_on_the_right_side(self):
        edwards = next(p for p in parse_event_odds(EVENT_PAYLOAD, "nba") if p.player == "Anthony Edwards")
        dk = next(b for b in edwards.books if b.book == "draftkings")
        assert (dk.over_american, dk.under_american, dk.line) == (130, -160, 27.5)

    def test_non_two_way_outcomes_are_dropped(self):
        edwards = next(p for p in parse_event_odds(EVENT_PAYLOAD, "nba") if p.player == "Anthony Edwards")
        # The "Yes" outcome has no point and no matching side: it must not
        # sneak in as a phantom line.
        assert all(b.line == 27.5 for b in edwards.books)

    def test_an_empty_payload_parses_to_nothing(self):
        assert parse_event_odds({"id": "x", "bookmakers": []}, "nba") == []

    def test_snapshots_round_trip_through_the_local_provider(self, tmp_path):
        props = parse_event_odds(EVENT_PAYLOAD, "nba")
        path = tmp_path / "nba_snap.json"
        path.write_text(json.dumps(to_snapshot(props, "nba")))

        reloaded = LocalOddsProvider(tmp_path).player_props("nba", ["player_points"])
        assert {p.player for p in reloaded} == {"Anthony Edwards", "Rudy Gobert"}
        original = next(p for p in props if p.player == "Rudy Gobert")
        copy = next(p for p in reloaded if p.player == "Rudy Gobert")
        assert [(b.book, b.line, b.over_american) for b in copy.books] == [
            (b.book, b.line, b.over_american) for b in original.books
        ]


class TestProviderSelection:
    def test_local_is_the_default(self):
        assert isinstance(build_odds_provider(Settings()), LocalOddsProvider)

    def test_the_live_provider_needs_a_key(self):
        with pytest.raises(OddsApiError, match="no API key"):
            TheOddsApiProvider(ApiSettings(provider="theoddsapi"))

    def test_an_api_key_switches_the_provider_on(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PMBOT_ODDS_API_KEY", "test-key")
        monkeypatch.delenv("PMBOT_PROVIDER", raising=False)
        settings = Settings.load(tmp_path / "absent.json")
        assert settings.api.provider == "theoddsapi"
        assert isinstance(build_odds_provider(settings), TheOddsApiProvider)

    def test_unknown_providers_are_rejected(self):
        settings = Settings()
        settings.api.provider = "bovada-scraper"
        with pytest.raises(ValueError, match="unknown odds provider"):
            build_odds_provider(settings)
