/*
    Intermediate: Enriched Provider Records

    Joins provider claims summary with NPPES registry data and LEIE
    exclusion status. This is the single enriched provider table that
    feeds the marts layer.

    Join logic:
    - Part B ← LEFT JOIN → NPPES: enrich with taxonomy/specialty
    - Part B ← LEFT JOIN → LEIE: add exclusion flag (our label)

    LEFT JOINs because not all Part B NPIs will be in our NPPES extract
    (API fallback may miss some), and most providers are NOT in LEIE.
*/

with claims_summary as (
    select * from {{ ref('int_provider_claims_summary') }}
),

nppes as (
    select * from {{ ref('stg_nppes_providers') }}
),

leie as (
    select * from {{ ref('stg_leie_exclusions') }}
),

enriched as (
    select
        cs.*,

        -- NPPES enrichment
        np.primary_taxonomy_code,
        np.enumeration_date,
        np.practice_state                as nppes_practice_state,
        np.practice_zip5                 as nppes_practice_zip5,

        -- LEIE exclusion flag (our supervised learning label)
        case
            when le.npi is not null then true
            else false
        end                              as is_excluded,
        le.exclusion_type,
        le.exclusion_date,
        le.is_currently_excluded,

        -- Coalesce provider type from CMS data and NPPES taxonomy
        coalesce(cs.provider_type, 'Unknown') as provider_specialty

    from claims_summary cs
    left join nppes np
        on cs.npi = np.npi
    left join leie le
        on cs.npi = le.npi
)

select * from enriched
