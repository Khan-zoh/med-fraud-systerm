/*
    Staging: Part B Claims (Provider × Service level)

    Cleans raw CMS Part B data:
    - Standardizes column names to snake_case
    - Casts string numerics to proper types
    - Trims whitespace from text fields
    - Filters out rows with null NPI (unusable for our analysis)
*/

with source as (
    select * from {{ source('raw', 'partb_by_provider_service') }}
),

cleaned as (
    select
        -- Provider identifiers
        cast("Rndrng_NPI" as varchar)                   as npi,
        trim("Rndrng_Prvdr_Last_Org_Name")              as provider_last_name,
        trim("Rndrng_Prvdr_First_Name")                 as provider_first_name,
        trim("Rndrng_Prvdr_Crdntls")                    as provider_credentials,
        trim("Rndrng_Prvdr_Gndr")                       as provider_gender,
        trim("Rndrng_Prvdr_Ent_Cd")                     as entity_type,

        -- Provider location
        trim("Rndrng_Prvdr_St1")                        as provider_street,
        trim("Rndrng_Prvdr_City")                       as provider_city,
        trim("Rndrng_Prvdr_State_Abrvtn")               as provider_state,
        trim("Rndrng_Prvdr_State_FIPS")                 as provider_state_fips,
        trim("Rndrng_Prvdr_Zip5")                       as provider_zip5,
        trim("Rndrng_Prvdr_RUCA")                       as provider_ruca,

        -- Provider specialty
        trim("Rndrng_Prvdr_Type")                       as provider_type,
        trim("Rndrng_Prvdr_Mdcr_Prtcptg_Ind")          as medicare_participating,

        -- Service identifiers
        trim("HCPCS_Cd")                                as hcpcs_code,
        trim("HCPCS_Desc")                              as hcpcs_description,
        trim("HCPCS_Drug_Ind")                          as hcpcs_drug_indicator,
        trim("Place_Of_Srvc")                           as place_of_service,

        -- Billing metrics (cast from string to numeric)
        cast("Tot_Benes" as integer)                    as total_beneficiaries,
        cast("Tot_Srvcs" as double)                     as total_services,
        cast("Tot_Bene_Day_Srvcs" as double)            as total_beneficiary_day_services,
        cast("Avg_Sbmtd_Chrg" as double)                as avg_submitted_charge,
        cast("Avg_Mdcr_Alowd_Amt" as double)            as avg_medicare_allowed_amt,
        cast("Avg_Mdcr_Pymt_Amt" as double)             as avg_medicare_payment_amt,
        cast("Avg_Mdcr_Stdzd_Amt" as double)            as avg_medicare_standardized_amt

    from source
    where "Rndrng_NPI" is not null
      and trim("Rndrng_NPI") != ''
)

select * from cleaned
