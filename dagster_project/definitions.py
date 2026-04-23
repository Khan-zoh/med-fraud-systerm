"""
Redirect — Dagster definitions live in the main package.

Run with: dagster dev -m cms_fwa.dagster_defs
Or:       make dagster-dev
"""

from cms_fwa.dagster_defs import defs  # noqa: F401
