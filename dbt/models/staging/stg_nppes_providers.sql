/*
    Staging: NPPES Provider Registry

    Cleans NPI registry data:
    - Standardizes column names (already snake_case from loader)
    - Classifies entity type (individual vs. organization)
    - Extracts clean zip5 from full postal code
*/

with source as (
    select * from {{ source('raw', 'nppes_providers') }}
),

cleaned as (
    select
        cast(npi as varchar)                                    as npi,

        case
            when cast(entity_type_code as varchar) = '1' then 'individual'
            when cast(entity_type_code as varchar) = '2' then 'organization'
            else 'unknown'
        end                                                     as entity_type,

        trim(last_name)                                         as last_name,
        trim(first_name)                                        as first_name,
        trim(organization_name)                                 as organization_name,
        trim(practice_state)                                    as practice_state,
        left(trim(practice_zip), 5)                             as practice_zip5,
        trim(taxonomy_code)                                     as primary_taxonomy_code,
        try_cast(enumeration_date as date)                      as enumeration_date,
        trim(gender_code)                                       as gender_code

    from source
    where npi is not null
      and trim(cast(npi as varchar)) != ''
)

select * from cleaned
