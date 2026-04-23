"""Tests for data source configuration."""

import pytest

from cms_fwa.ingestion.sources import (
    LEIE_COLUMNS,
    NPPES_COLUMN_RENAMES,
    NPPES_COLUMNS,
    PARTB_COLUMNS,
    PARTB_DATASET_UUIDS,
    RAW_TABLES,
    get_partb_api_url,
)


def test_partb_api_url_2022() -> None:
    """Should construct a valid CMS API URL for 2022."""
    url = get_partb_api_url(2022)
    assert "data.cms.gov" in url
    assert PARTB_DATASET_UUIDS[2022] in url


def test_partb_api_url_invalid_year() -> None:
    """Should raise ValueError for unconfigured years."""
    with pytest.raises(ValueError, match="No dataset UUID"):
        get_partb_api_url(1990)


def test_partb_columns_include_npi() -> None:
    """Part B column list must include the NPI join key."""
    assert "Rndrng_NPI" in PARTB_COLUMNS


def test_leie_columns_include_npi() -> None:
    """LEIE column list must include NPI for joining."""
    assert "NPI" in LEIE_COLUMNS


def test_nppes_rename_keys_match_columns() -> None:
    """Every NPPES rename key should correspond to a kept column."""
    assert set(NPPES_COLUMN_RENAMES.keys()) == set(NPPES_COLUMNS)


def test_raw_table_names_are_schema_qualified() -> None:
    """All raw table names should be in the 'raw' schema."""
    assert RAW_TABLES.partb.startswith("raw.")
    assert RAW_TABLES.leie.startswith("raw.")
    assert RAW_TABLES.nppes.startswith("raw.")
