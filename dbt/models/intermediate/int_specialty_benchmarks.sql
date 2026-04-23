/*
    Intermediate: Specialty Peer Group Benchmarks

    Computes aggregate statistics per provider_type (specialty).
    These become the "control limits" in our SPC-inspired feature
    engineering — a provider is anomalous when they deviate significantly
    from their specialty peer group.

    Industrial Engineering connection: this is the same logic as computing
    control chart parameters (X-bar, sigma) for a process, where each
    specialty is a distinct process.
*/

with providers as (
    select * from {{ ref('int_provider_claims_summary') }}
),

specialty_stats as (
    select
        provider_type                                       as specialty,

        -- Peer group size
        count(*)                                            as peer_count,

        -- Volume benchmarks
        avg(total_services)                                 as avg_total_services,
        stddev_pop(total_services)                          as std_total_services,
        median(total_services)                              as median_total_services,

        -- Revenue benchmarks
        avg(total_medicare_payments)                        as avg_total_payments,
        stddev_pop(total_medicare_payments)                 as std_total_payments,
        median(total_medicare_payments)                     as median_total_payments,

        -- Beneficiary benchmarks
        avg(total_beneficiaries)                            as avg_total_beneficiaries,
        stddev_pop(total_beneficiaries)                     as std_total_beneficiaries,

        -- Billing intensity benchmarks
        avg(services_per_beneficiary)                       as avg_services_per_bene,
        stddev_pop(services_per_beneficiary)                as std_services_per_bene,
        median(services_per_beneficiary)                    as median_services_per_bene,

        -- Procedure diversity benchmarks
        avg(unique_hcpcs_codes)                             as avg_unique_hcpcs,
        stddev_pop(unique_hcpcs_codes)                      as std_unique_hcpcs,

        -- Charge benchmarks
        avg(wavg_submitted_charge)                          as avg_wavg_charge,
        stddev_pop(wavg_submitted_charge)                   as std_wavg_charge,

        -- Percentiles for robust outlier detection
        percentile_cont(0.25) within group (order by total_services)
                                                            as p25_total_services,
        percentile_cont(0.75) within group (order by total_services)
                                                            as p75_total_services,
        percentile_cont(0.95) within group (order by total_services)
                                                            as p95_total_services,
        percentile_cont(0.99) within group (order by total_services)
                                                            as p99_total_services

    from providers
    where provider_type is not null
    group by provider_type
    having count(*) >= 10  -- need enough peers for meaningful statistics
)

select * from specialty_stats
