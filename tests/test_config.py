import json

import pytest

from pmbot.config import Settings


class TestOverlays:
    def test_defaults_are_conservative(self):
        s = Settings()
        assert s.staking.kelly_fraction <= 0.25
        assert s.staking.max_bet_fraction <= 0.05
        assert s.devig.exclude_own_book is True

    def test_a_config_file_overlays_defaults(self, tmp_path):
        path = tmp_path / "pmbot.config.json"
        path.write_text(json.dumps({
            "staking": {"bankroll": 25_000, "kelly_fraction": 0.5},
            "devig": {"method": "shin"},
        }))
        s = Settings.load(path)
        assert s.staking.bankroll == 25_000
        assert s.devig.method == "shin"
        assert s.staking.max_bet_fraction == Settings().staking.max_bet_fraction  # untouched

    def test_lists_become_tuples(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"devig": {"sharp_books": ["pinnacle", "circa"]}}))
        assert Settings.load(path).devig.sharp_books == ("pinnacle", "circa")

    def test_typos_are_rejected_rather_than_ignored(self):
        """A silently ignored key is a config you think is applied and isn't."""
        with pytest.raises(KeyError, match="unknown config key"):
            Settings().apply({"staking": {"kely_fraction": 0.5}})
        with pytest.raises(KeyError, match="unknown config section"):
            Settings().apply({"stakes": {}})

    def test_sections_must_be_objects(self):
        with pytest.raises(TypeError):
            Settings().apply({"staking": 0.5})


class TestEnvironment:
    def test_env_overrides_are_typed(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PMBOT_BANKROLL", "7500")
        monkeypatch.setenv("PMBOT_KELLY_FRACTION", "0.5")
        monkeypatch.setenv("PMBOT_DEVIG", "shin")
        s = Settings.load(tmp_path / "absent.json")
        assert s.staking.bankroll == 7500.0
        assert isinstance(s.staking.bankroll, float)
        assert s.staking.kelly_fraction == 0.5
        assert s.devig.method == "shin"

    def test_an_explicit_provider_beats_the_key_shortcut(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PMBOT_ODDS_API_KEY", "abc")
        monkeypatch.setenv("PMBOT_PROVIDER", "local")
        assert Settings.load(tmp_path / "absent.json").api.provider == "local"

    def test_empty_env_values_are_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PMBOT_BANKROLL", "")
        assert Settings.load(tmp_path / "absent.json").staking.bankroll == Settings().staking.bankroll


class TestDataDir:
    """One env var has to move every writable path, or a volume mount is a trap."""

    def test_data_dir_moves_everything_writable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PMBOT_DATA_DIR", str(tmp_path))
        data = Settings.load(tmp_path / "absent.json").data
        assert data.db == str(tmp_path / "pmbot.sqlite3")
        assert data.gamelogs == str(tmp_path / "gamelogs")
        assert data.props == str(tmp_path / "props")
        assert data.cache.startswith(str(tmp_path))
        assert data.ingest_cache.startswith(str(tmp_path))

    def test_the_repo_defense_table_survives_an_empty_volume(self, monkeypatch, tmp_path):
        """Losing every matchup adjustment silently would be worse than loud."""
        monkeypatch.setenv("PMBOT_DATA_DIR", str(tmp_path))
        assert Settings.load(tmp_path / "absent.json").data.defense == "data/defense.json"

    def test_a_volume_defense_table_wins_when_present(self, monkeypatch, tmp_path):
        (tmp_path / "defense.json").write_text("{}")
        monkeypatch.setenv("PMBOT_DATA_DIR", str(tmp_path))
        assert Settings.load(tmp_path / "absent.json").data.defense == str(tmp_path / "defense.json")

    def test_an_explicit_db_still_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PMBOT_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("PMBOT_DB", "/elsewhere/journal.sqlite3")
        assert Settings.load(tmp_path / "absent.json").data.db == "/elsewhere/journal.sqlite3"

    def test_deployment_knobs_are_settable_from_the_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PMBOT_MC_SIMS", "4000")
        monkeypatch.setenv("PMBOT_MIN_EV", "0.05")
        monkeypatch.setenv("PMBOT_MAX_BET_FRACTION", "0.01")
        s = Settings.load(tmp_path / "absent.json")
        assert s.models.mc_sims == 4000
        assert s.staking.min_ev == 0.05
        assert s.staking.max_bet_fraction == 0.01


class TestSerialisation:
    def test_round_trips_through_disk(self, tmp_path):
        original = Settings()
        original.staking.bankroll = 3210.0
        path = original.save(tmp_path / "out.json")
        assert Settings.load(path).staking.bankroll == 3210.0

    def test_to_dict_is_json_serialisable(self):
        json.dumps(Settings().to_dict(), default=list)
