"""
Data source definitions for CMS FWA ingestion.

Each source specifies where to download from, what columns to expect,
and how to load into DuckDB. URLs are constructed dynamically based on
the configured year/state in settings.

Data sources (all public, no credentials required):
  - CMS Part B by Provider & Service: JSON API via data.cms.gov
  - LEIE (OIG exclusions): Direct CSV download
  - NPPES NPI Registry: Zipped CSV dissemination file
"""

from dataclasses import dataclass, field

from cms_fwa.config import settings

# ---------------------------------------------------------------------------
# CMS Part B — Provider & Service level billing data
# ---------------------------------------------------------------------------
# The data.cms.gov API serves datasets by a UUID that changes each year.
# These UUIDs map to "Medicare Physician & Other Practitioners by Provider
# and Service" for each release year.

PARTB_DATASET_UUIDS: dict[int, str] = {
    2021: "e650987d-01b7-4f09-b75e-b0b075afbf98",
    2022: "0e9f2f2b-7bf9-451a-912c-e02e654dd725",
    2023: "92396110-2aed-4d63-a6a2-5d6207d46a29",
}

PARTB_API_BASE = "https://data.cms.gov/data-api/v1/dataset"
PARTB_PAGE_SIZE = 5000  # max rows per API call


def get_partb_api_url(year: int | None = None) -> str:
    """Build the CMS Part B API endpoint for the configured year."""
    year = year or settings.cms_data_year
    uuid = PARTB_DATASET_UUIDS.get(year)
    if uuid is None:
        available = ", ".join(str(y) for y in sorted(PARTB_DATASET_UUIDS))
        raise ValueError(
            f"No dataset UUID configured for year {year}. Available: {available}"
        )
    return f"{PARTB_API_BASE}/{uuid}/data"


# Columns we keep from the Part B dataset (raw names from the API).
# The API returns ALL columns; we select these to reduce storage.
PARTB_COLUMNS: list[str] = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Last_Org_Name",
    "Rndrng_Prvdr_First_Name",
    "Rndrng_Prvdr_MI",
    "Rndrng_Prvdr_Crdntls",
    "Rndrng_Prvdr_Gndr",
    "Rndrng_Prvdr_Ent_Cd",
    "Rndrng_Prvdr_St1",
    "Rndrng_Prvdr_City",
    "Rndrng_Prvdr_State_Abrvtn",
    "Rndrng_Prvdr_State_FIPS",
    "Rndrng_Prvdr_Zip5",
    "Rndrng_Prvdr_RUCA",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind",
    "HCPCS_Cd",
    "HCPCS_Desc",
    "HCPCS_Drug_Ind",
    "Place_Of_Srvc",
    "Tot_Benes",
    "Tot_Srvcs",
    "Tot_Bene_Day_Srvcs",
    "Avg_Sbmtd_Chrg",
    "Avg_Mdcr_Alowd_Amt",
    "Avg_Mdcr_Pymt_Amt",
    "Avg_Mdcr_Stdzd_Amt",
]

# ---------------------------------------------------------------------------
# LEIE — OIG List of Excluded Individuals/Entities
# ---------------------------------------------------------------------------

LEIE_CSV_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"

LEIE_COLUMNS: list[str] = [
    "LASTNAME",
    "FIRSTNAME",
    "MIDNAME",
    "NPI",
    "DOB",
    "ADDRESS",
    "CITY",
    "STATE",
    "ZIP",
    "SPECIALTY",
    "EXCLTYPE",
    "EXCLDATE",
    "REINDATE",
    "WAIVERDATE",
    "WVRSTATE",
]

# ---------------------------------------------------------------------------
# NPPES — National Plan & Provider Enumeration System
# ---------------------------------------------------------------------------
# Full dissemination file is ~7GB zipped. We download it once and use
# DuckDB's lazy CSV reader to extract only the columns we need.

NPPES_DOWNLOAD_URL = (
    "https://download.cms.gov/nppes/NPPES_Data_Dissemination_January_2024.zip"
)

# We only need a handful of columns from the 300+ in the full file
NPPES_COLUMNS: list[str] = [
    "NPI",
    "Entity Type Code",
    "Provider Last Name (Legal Name)",
    "Provider First Name",
    "Provider Organization Name (Legal Business Name)",
    "Provider Business Practice Location Address State Name",
    "Provider Business Practice Location Address Postal Code",
    "Healthcare Provider Taxonomy Code_1",
    "Provider Enumeration Date",
    "Provider Gender Code",
]

# Rename mapping for cleaner column names in DuckDB
NPPES_COLUMN_RENAMES: dict[str, str] = {
    "NPI": "npi",
    "Entity Type Code": "entity_type_code",
    "Provider Last Name (Legal Name)": "last_name",
    "Provider First Name": "first_name",
    "Provider Organization Name (Legal Business Name)": "organization_name",
    "Provider Business Practice Location Address State Name": "practice_state",
    "Provider Business Practice Location Address Postal Code": "practice_zip",
    "Healthcare Provider Taxonomy Code_1": "taxonomy_code",
    "Provider Enumeration Date": "enumeration_date",
    "Provider Gender Code": "gender_code",
}


# ---------------------------------------------------------------------------
# DuckDB raw table names
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RawTableNames:
    """Centralized raw-schema table names to avoid magic strings."""

    partb: str = "raw.partb_by_provider_service"
    leie: str = "raw.leie_exclusions"
    nppes: str = "raw.nppes_providers"


RAW_TABLES = RawTableNames()
