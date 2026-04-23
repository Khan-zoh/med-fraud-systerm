/*
    Intermediate: Provider Claims Summary

    Aggregates Part B claim lines to provider level — one row per NPI.
    This is the foundation for feature engineering: total volume, revenue,
    procedure diversity, and billing intensity metrics.
*/

with claims as (
    select * from {{ ref('stg_partb_claims') }}
),

provider_summary as (
    select
        npi,

        -- Provider demographics (take the first non-null from claim lines)
        min(provider_last_name)                         as provider_last_name,
        min(provider_first_name)                        as provider_first_name,
        min(provider_type)                              as provider_type,
        min(entity_type)                                as entity_type,
        min(provider_state)                             as provider_state,
        min(provider_zip5)                              as provider_zip5,
        min(provider_gender)                            as provider_gender,

        -- Volume metrics
        count(distinct hcpcs_code)                      as unique_hcpcs_codes,
        sum(total_services)                             as total_services,
        sum(total_beneficiaries)                        as total_beneficiaries,
        sum(total_beneficiary_day_services)             as total_beneficiary_day_services,

        -- Revenue metrics
        sum(total_services * avg_submitted_charge)      as total_submitted_charges,
        sum(total_services * avg_medicare_payment_amt)  as total_medicare_payments,
        sum(total_services * avg_medicare_allowed_amt)  as total_medicare_allowed,
        sum(total_services * avg_medicare_standardized_amt) as total_standardized_payments,

        -- Averages (weighted by service volume)
        sum(total_services * avg_submitted_charge)
            / nullif(sum(total_services), 0)            as wavg_submitted_charge,
        sum(total_services * avg_medicare_payment_amt)
            / nullif(sum(total_services), 0)            as wavg_medicare_payment,

        -- Billing intensity
        sum(total_services)
            / nullif(sum(total_beneficiaries), 0)       as services_per_beneficiary,

        -- Place of service mix
        sum(case when place_of_service = 'F' then total_services else 0 end)
            / nullif(sum(total_services), 0)            as facility_service_ratio,

        -- Drug indicator mix
        sum(case when hcpcs_drug_indicator = 'Y' then total_services else 0 end)
            / nullif(sum(total_services), 0)            as drug_service_ratio

    from claims
    group by npi
)

select * from provider_summary
