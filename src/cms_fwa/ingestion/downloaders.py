"""
HTTP download utilities for CMS data sources.

Handles chunked downloads with progress bars, resume support via
Content-Range, and CMS API pagination. All downloads are idempotent —
if the target file already exists and is complete, the download is skipped.
"""

import json
from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm

from cms_fwa.config import settings
from cms_fwa.ingestion.sources import (
    LEIE_CSV_URL,
    NPPES_DOWNLOAD_URL,
    PARTB_API_BASE,
    PARTB_COLUMNS,
    PARTB_PAGE_SIZE,
    get_partb_api_url,
)

# Generous timeout for large government data downloads
HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
DOWNLOAD_CHUNK_SIZE = 1024 * 256  # 256 KB chunks


def _ensure_raw_dir() -> Path:
    """Create and return the raw data directory."""
    raw_dir = settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def download_partb_data(
    year: int | None = None,
    state: str | None = None,
) -> Path:
    """Download CMS Part B data via the data.cms.gov JSON API.

    The API returns paginated JSON. We stream all pages into a single
    JSONL (newline-delimited JSON) file for efficient DuckDB ingestion.

    Args:
        year: Data year (defaults to settings.cms_data_year).
        state: State abbreviation filter (defaults to settings.cms_data_state).

    Returns:
        Path to the downloaded JSONL file.
    """
    year = year or settings.cms_data_year
    state = state or settings.cms_data_state

    raw_dir = _ensure_raw_dir()
    output_file = raw_dir / f"partb_{state.lower()}_{year}.jsonl"
    marker_file = raw_dir / f"partb_{state.lower()}_{year}.complete"

    # Idempotency: skip if already downloaded
    if marker_file.exists():
        logger.info(f"Part B data already downloaded: {output_file}")
        return output_file

    api_url = get_partb_api_url(year)
    logger.info(f"Downloading Part B data: year={year}, state={state}")

    total_rows = 0
    offset = 0

    with open(output_file, "w", encoding="utf-8") as f:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            while True:
                params: dict[str, str | int] = {
                    "size": PARTB_PAGE_SIZE,
                    "offset": offset,
                }
                # Filter by state if specified
                if state:
                    params["filter[Rndrng_Prvdr_State_Abrvtn]"] = state

                logger.debug(f"Fetching offset={offset}")
                resp = client.get(api_url, params=params)
                resp.raise_for_status()

                records = resp.json()

                if not records:
                    break

                for record in records:
                    # Keep only the columns we care about
                    filtered = {k: record.get(k) for k in PARTB_COLUMNS}
                    f.write(json.dumps(filtered) + "\n")

                total_rows += len(records)
                offset += len(records)

                if len(records) < PARTB_PAGE_SIZE:
                    break

                logger.info(f"  ... downloaded {total_rows:,} rows so far")

    # Write completion marker
    marker_file.write_text(str(total_rows))
    logger.info(f"Part B download complete: {total_rows:,} rows -> {output_file}")
    return output_file


def download_leie_data() -> Path:
    """Download the LEIE (OIG exclusions) CSV file.

    Returns:
        Path to the downloaded CSV file.
    """
    raw_dir = _ensure_raw_dir()
    output_file = raw_dir / "leie_updated.csv"
    marker_file = raw_dir / "leie_updated.complete"

    if marker_file.exists():
        logger.info(f"LEIE data already downloaded: {output_file}")
        return output_file

    logger.info(f"Downloading LEIE exclusions from {LEIE_CSV_URL}")

    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        with client.stream("GET", LEIE_CSV_URL) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(output_file, "wb") as f:
                with tqdm(
                    total=total, unit="B", unit_scale=True, desc="LEIE"
                ) as pbar:
                    for chunk in resp.iter_bytes(DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)
                        pbar.update(len(chunk))

    marker_file.write_text("complete")
    logger.info(f"LEIE download complete -> {output_file}")
    return output_file


def download_nppes_data() -> Path:
    """Download the NPPES NPI dissemination ZIP file.

    This is a large file (~7GB). DuckDB will read it lazily, so we only
    need the raw zip on disk. If the download is interrupted, delete the
    partial file and re-run — the marker file ensures idempotency.

    Returns:
        Path to the downloaded ZIP file.
    """
    raw_dir = _ensure_raw_dir()
    output_file = raw_dir / "nppes_dissemination.zip"
    marker_file = raw_dir / "nppes_dissemination.complete"

    if marker_file.exists():
        logger.info(f"NPPES data already downloaded: {output_file}")
        return output_file

    logger.info(
        f"Downloading NPPES dissemination file (this is large, ~7GB)..."
    )
    logger.info(f"URL: {NPPES_DOWNLOAD_URL}")

    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        with client.stream("GET", NPPES_DOWNLOAD_URL) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(output_file, "wb") as f:
                with tqdm(
                    total=total, unit="B", unit_scale=True, desc="NPPES"
                ) as pbar:
                    for chunk in resp.iter_bytes(DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)
                        pbar.update(len(chunk))

    marker_file.write_text("complete")
    logger.info(f"NPPES download complete -> {output_file}")
    return output_file
