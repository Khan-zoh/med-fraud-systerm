/*
    Staging: LEIE Exclusions

    Cleans the OIG List of Excluded Individuals/Entities:
    - Standardizes column names
    - Parses exclusion dates
    - Filters to records with valid NPIs (many older records lack NPIs)

    Important context: an LEIE exclusion is NOT the same as a fraud conviction.
    Exclusions cover a range of offenses. We use this as a *proxy label* for
    supervised learning, with full awareness of its limitations.
*/

with source as (
    select * from {{ source('raw', 'leie_exclusions') }}
),

cleaned as (
    select
        -- Provider identifiers
        cast("NPI" as varchar)              as npi,
        trim("LASTNAME")                    as last_name,
        trim("FIRSTNAME")                   as first_name,
        trim("MIDNAME")                     as middle_name,

        -- Location
        trim("ADDRESS")                     as address,
        trim("CITY")                        as city,
        trim("STATE")                       as state,
        trim("ZIP")                         as zip_code,

        -- Exclusion details
        trim("SPECIALTY")                   as specialty,
        trim("EXCLTYPE")                    as exclusion_type,
        try_cast("EXCLDATE" as date)        as exclusion_date,
        try_cast("REINDATE" as date)        as reinstatement_date,
        try_cast("WAIVERDATE" as date)      as waiver_date,
        trim("WVRSTATE")                    as waiver_state,

        -- Derived: is the exclusion currently active?
        case
            when try_cast("REINDATE" as date) is not null then false
            else true
        end                                 as is_currently_excluded

    from source
    where "NPI" is not null
      and trim("NPI") != ''
      and length(trim("NPI")) = 10
)

select * from cleaned
