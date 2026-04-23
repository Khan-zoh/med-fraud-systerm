"""Tests for the validation module."""

import json
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from cms_fwa.ingestion.sources import RAW_TABLES
from cms_fwa.ingestion.validators import ValidationResult


def test_validation_result_str_pass() -> None:
    """Passing validation should display PASS."""
    r = ValidationResult(
        table_name="test_table",
        success=True,
        total_expectations=5,
        successful_expectations=5,
        failed_expectations=0,
        failure_details=[],
    )
    assert "[PASS]" in str(r)
    assert "5/5" in str(r)


def test_validation_result_str_fail() -> None:
    """Failing validation should display FAIL."""
    r = ValidationResult(
        table_name="test_table",
        success=False,
        total_expectations=5,
        successful_expectations=3,
        failed_expectations=2,
        failure_details=["Bad NPI", "Missing column"],
    )
    assert "[FAIL]" in str(r)
    assert "3/5" in str(r)
