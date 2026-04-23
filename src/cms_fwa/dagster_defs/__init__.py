"""
Dagster Definitions — the single entry point for `dagster dev`.

Run with: dagster dev -m cms_fwa.dagster_defs
Or:       make dagster-dev
"""

from dagster import Definitions

from cms_fwa.dagster_defs.assets import (
    leie_raw_file,
    leie_raw_table,
    nppes_raw_table,
    partb_raw_file,
    partb_raw_table,
)

defs = Definitions(
    assets=[
        partb_raw_file,
        partb_raw_table,
        leie_raw_file,
        leie_raw_table,
        nppes_raw_table,
    ],
)
