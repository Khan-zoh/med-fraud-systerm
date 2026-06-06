/*
    Intermediate: Enriched Provider Records

    Joins provider claims summary with NPPES registry data and the curated
    multi-tier exclusion label set produced by the Python LEIE matcher.

    Join logic:
    - Part B ← LEFT JOIN → NPPES:           enrich with taxonomy/specialty
    - Part B ← LEFT JOIN → matched labels:  attach is_excluded flag and
                                            tier provenance for filtering

    The previous NPI-only LEIE join (stg_leie_exclusions) yielded just 26
    positive labels — too few for supervised evaluation. The new label
    source (stg_provider_exclusion_labels) uses tiered name+state matching
    and yields ~286 labels. See docs/labeling-methodology.md.
*/

with claims_summary as (
    select * from {{ ref('int_provider_claims_summary') }}
),

nppes as (
    select * from {{ ref('stg_nppes_providers') }}
),

labels as (
    select * from {{ ref('stg_provider_exclusion_labels') }}
),

enriched as (
    select
        cs.*,

        -- NPPES enrichment
        np.primary_taxonomy_code,
        np.enumeration_date,
        np.practice_state                as nppes_practice_state,
        np.practice_zip5                 as nppes_practice_zip5,

        -- Exclusion label (supervised target). Source: leie_matcher Python pipeline.
        coalesce(lb.is_excluded, false)  as is_excluded,
        lb.match_tier                    as exclusion_match_tier,
        lb.similarity_score              as exclusion_match_score,
        lb.exclusion_type                as exclusion_type,
        lb.exclusion_date                as exclusion_date,
        coalesce(lb.is_currently_excluded, false) as is_currently_excluded,

        -- Coalesce provider type from CMS data and NPPES taxonomy
        coalesce(cs.provider_type, 'Unknown') as provider_specialty

    from claims_summary cs
    left join nppes np
        on cs.npi = np.npi
    left join labels lb
        on cs.npi = lb.npi
)

select * from enriched
