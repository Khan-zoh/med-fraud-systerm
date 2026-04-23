"""
Centralized configuration for the CMS FWA Detection project.

All paths, data parameters, and service settings are loaded from environment
variables (via .env) with sensible defaults for local development. No hardcoded
paths anywhere else in the codebase — import from here.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


PROJECT_ROOT = _project_root()


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- General ---
    cms_fwa_env: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Logging level")

    # --- Data Paths ---
    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    duckdb_path: Path = Field(
        default=Path("data/cms_fwa.duckdb"), description="DuckDB database file"
    )

    # --- CMS Data Configuration ---
    cms_data_year: int = Field(default=2022, description="Year of Part B data to ingest")
    cms_data_state: str = Field(
        default="TX,NM,AZ,NV,OK,CO",
        description="Comma-separated state abbreviations to ingest (e.g. 'TX' or 'TX,NM,AZ')",
    )

    @property
    def cms_data_states(self) -> list[str]:
        """Parse cms_data_state into a list of state codes."""
        return [s.strip().upper() for s in self.cms_data_state.split(",") if s.strip()]

    # --- API Server ---
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(default=8000, description="FastAPI bind port")

    # --- Streamlit ---
    streamlit_port: int = Field(default=8501, description="Streamlit dashboard port")

    # --- Dagster ---
    dagster_home: Path = Field(default=Path("dagster_home"), description="Dagster home directory")

    @property
    def raw_data_dir(self) -> Path:
        """Directory for raw downloaded files."""
        return self._resolve(self.data_dir / "raw")

    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed/transformed data."""
        return self._resolve(self.data_dir / "processed")

    @property
    def models_dir(self) -> Path:
        """Directory for trained model artifacts."""
        return self._resolve(self.data_dir / "models")

    @property
    def duckdb_abs_path(self) -> Path:
        """Absolute path to DuckDB database."""
        return self._resolve(self.duckdb_path)

    def _resolve(self, path: Path) -> Path:
        """Resolve a path relative to PROJECT_ROOT if not already absolute."""
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


# Singleton — import this everywhere
settings = Settings()
