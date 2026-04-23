"""Smoke test for the config module."""

from pathlib import Path

from cms_fwa.config import Settings, settings


def test_settings_defaults() -> None:
    """Settings should load with sensible defaults even without a .env file."""
    s = Settings()
    assert s.cms_data_year == 2022
    assert s.cms_data_state == "TX"
    assert s.api_port == 8000
    assert s.log_level == "INFO"


def test_raw_data_dir_is_path() -> None:
    """Data directory properties should return Path objects."""
    assert isinstance(settings.raw_data_dir, Path)
    assert settings.raw_data_dir.name == "raw"


def test_duckdb_path_is_absolute() -> None:
    """DuckDB absolute path should always be absolute."""
    assert settings.duckdb_abs_path.is_absolute()
