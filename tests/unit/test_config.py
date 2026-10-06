from __future__ import annotations

from pathlib import Path

from spectra.config import ENV_DATA_DIR, AppConfig, default_data_dir, load_config


def test_default_data_dir_is_outside_the_repository():
    data_dir = default_data_dir()
    assert Path.cwd() not in data_dir.parents
    assert data_dir.name == "Spectra"


def test_env_var_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "custom"))
    assert default_data_dir() == (tmp_path / "custom").resolve()


def test_derived_paths_live_under_the_data_dir(tmp_path):
    config = AppConfig(data_dir=tmp_path)
    for path in (
        config.db_path,
        config.reports_dir,
        config.exports_dir,
        config.drawings_dir,
        config.logs_dir,
        config.key_path,
    ):
        assert tmp_path in path.parents


def test_ensure_directories_creates_the_tree(tmp_path):
    config = AppConfig(data_dir=tmp_path / "Spectra")
    config.ensure_directories()
    assert config.reports_dir.is_dir()
    assert config.key_path.parent.is_dir()


def test_round_trip_through_dict(tmp_path):
    original = AppConfig(data_dir=tmp_path, share_dir=tmp_path / "share", demo_mode=True)
    restored = AppConfig.from_dict(original.to_dict())
    assert restored.data_dir == original.data_dir
    assert restored.share_dir == original.share_dir
    assert restored.demo_mode is True


def test_unknown_keys_are_ignored(tmp_path):
    config = AppConfig.from_dict({"data_dir": str(tmp_path), "legacy_option": 42})
    assert config.data_dir == tmp_path


def test_save_and_load_round_trip(tmp_path):
    config = AppConfig(data_dir=tmp_path, hover_select_secs=2.0, locale="pt_BR")
    config.save()
    assert load_config(tmp_path).hover_select_secs == 2.0


def test_load_config_with_corrupt_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    assert load_config(tmp_path).locale == "pt_BR"


def test_model_path_prefers_the_bundled_copy(tmp_path):
    config = AppConfig(data_dir=tmp_path)
    assert config.model_path().name == "hand_landmarker.task"
