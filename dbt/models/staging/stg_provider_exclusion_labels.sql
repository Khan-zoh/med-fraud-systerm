/*
    Staging: Provider Exclusion Labels (matched)

    One row per NPI labeled as excluded, produced by the Python LEIE matcher
    (src/cms_fwa/ingestion/leie_matcher.py) using a 4-tier fuzzy match strategy.
    Replaces the old NPI-only LEIE join in stg_leie_exclusions, which yielded
    only 26 labels.

    See docs/labeling-methodology.md for the matching strategy, audit results,
    and FP-rate estimates.

    This table is loaded into DuckDB by the matcher's materialize_to_duckdb()
    function; dbt picks it up via the 'matched_labels' source.
*/

with source as (
    select * from {{ source('matched_labels', 'provider_exclusion_labels') }}
),

cleaned as (
    select
        cast(npi as varchar)                as npi,
        cast(is_excluded as boolean)        as is_excluded,
        cast(match_tier as integer)         as match_tier,
        cast(similarity_score as double)    as similarity_score,
        exclusion_type                      as exclusion_type,
        try_cast(exclusion_date_parsed as date) as exclusion_date,
        -- Match is "currently active" unless we have a reinstatement signal —
        -- since the matcher doesn't track reinstatements yet, we treat every
        -- matched label as currently active. This is conservative for ML
        -- (more positive labels) and consistent with stg_leie_exclusions.
        true                                as is_currently_excluded,
        leie_specialty                      as leie_specialty,
        leie_row_n_candidates               as leie_row_n_candidates,
        provider_n_leie_rows                as provider_n_leie_rows
    from source
)

select * from cleaned
