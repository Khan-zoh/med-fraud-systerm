/*
    Mart: Provider Feature Table (ML-Ready)

    One row per provider with all features needed for modeling.
    This is the primary input to the ML pipeline (Phase 5).

    Features are organized by the IE principle they embody:
    1. SPC z-scores (control chart logic)
    2. Peer benchmarking ratios
    3. Billing intensity / throughput metrics
    4. Upcoding indicators
    5. Label (LEIE exclusion flag)

    All z-scores use the provider's specialty peer group as the baseline —
    a cardiologist is compared to other cardiologists, not to podiatrists.
*/

with providers as (
    select * from {{ ref('int_provider_enriched') }}
),

benchmarks as (
    select * from {{ ref('int_specialty_benchmarks') }}
),

hcpcs_agg as (
    -- Compute upcoding ratio per provider
    select
        npi,
        sum(case when is_high_complexity then hcpcs_services else 0 end)
            / nullif(sum(hcpcs_services), 0)    as high_complexity_ratio
    from {{ ref('int_provider_hcpcs_mix') }}
    group by npi
),

specialty_hcpcs_agg as (
    -- Compute peer-group upcoding ratio per specialty
    select
        provider_type                           as specialty,
        sum(case when is_high_complexity then hcpcs_services else 0 end)
            / nullif(sum(hcpcs_services), 0)    as peer_high_complexity_ratio
    from {{ ref('int_provider_hcpcs_mix') }}
    group by provider_type
),

features as (
    select
        p.npi,
        p.provider_last_name,
        p.provider_first_name,
        p.provider_specialty,
        p.entity_type,
        p.provider_state,
        p.provider_zip5,
        p.primary_taxonomy_code,

        -- =====================================================
        -- RAW METRICS
        -- =====================================================
        p.total_services,
        p.total_beneficiaries,
        p.total_medicare_payments,
        p.total_submitted_charges,
        p.unique_hcpcs_codes,
        p.services_per_beneficiary,
        p.facility_service_ratio,
        p.drug_service_ratio,
        p.wavg_submitted_charge,
        p.wavg_medicare_payment,

        -- =====================================================
        -- FEATURE 1: SPC Z-SCORES (Statistical Process Control)
        -- =====================================================
        -- How many standard deviations is this provider from their
        -- specialty mean? Same logic as a control chart: values beyond
        -- ±3σ are "out of control."

        (p.total_services - b.avg_total_services)
            / nullif(b.std_total_services, 0)
            as zscore_total_services,

        (p.total_medicare_payments - b.avg_total_payments)
            / nullif(b.std_total_payments, 0)
            as zscore_total_payments,

        (p.total_beneficiaries - b.avg_total_beneficiaries)
            / nullif(b.std_total_beneficiaries, 0)
            as zscore_total_beneficiaries,

        (p.services_per_beneficiary - b.avg_services_per_bene)
            / nullif(b.std_services_per_bene, 0)
            as zscore_services_per_bene,

        (p.unique_hcpcs_codes - b.avg_unique_hcpcs)
            / nullif(b.std_unique_hcpcs, 0)
            as zscore_unique_hcpcs,

        (p.wavg_submitted_charge - b.avg_wavg_charge)
            / nullif(b.std_wavg_charge, 0)
            as zscore_avg_charge,

        -- =====================================================
        -- FEATURE 2: PEER BENCHMARKING RATIOS
        -- =====================================================
        -- How does this provider compare to their specialty median?
        -- Ratios > 1 mean above-median, >> 1 means far above peers.

        p.total_services
            / nullif(b.median_total_services, 0)
            as ratio_services_vs_median,

        p.total_medicare_payments
            / nullif(b.median_total_payments, 0)
            as ratio_payments_vs_median,

        p.services_per_beneficiary
            / nullif(b.median_services_per_bene, 0)
            as ratio_svc_per_bene_vs_median,

        -- =====================================================
        -- FEATURE 3: BILLING VELOCITY / THROUGHPUT
        -- =====================================================
        -- Services per beneficiary per day — a provider seeing 100
        -- patients/day and billing 20 procedures each is physically
        -- suspect. (Queueing theory: even at 100% utilization, there
        -- are throughput limits.)

        p.total_beneficiary_day_services
            / nullif(p.total_beneficiaries, 0)
            as beneficiary_day_intensity,

        -- =====================================================
        -- FEATURE 4: UPCODING INDICATORS
        -- =====================================================
        -- Ratio of high-complexity codes relative to peers.

        ha.high_complexity_ratio,

        ha.high_complexity_ratio
            / nullif(sha.peer_high_complexity_ratio, 0)
            as upcoding_ratio_vs_peers,

        -- =====================================================
        -- FEATURE 5: PEER GROUP CONTEXT
        -- =====================================================
        b.peer_count                        as specialty_peer_count,
        b.avg_total_services                as specialty_avg_services,
        b.p95_total_services                as specialty_p95_services,
        b.p99_total_services                as specialty_p99_services,

        -- Is this provider above the 95th percentile for their specialty?
        case
            when p.total_services > b.p95_total_services then true
            else false
        end                                 as is_above_p95_volume,

        case
            when p.total_services > b.p99_total_services then true
            else false
        end                                 as is_above_p99_volume,

        -- =====================================================
        -- LABEL (for supervised learning)
        -- =====================================================
        p.is_excluded,
        p.exclusion_type,
        p.exclusion_date,
        p.is_currently_excluded

    from providers p
    left join benchmarks b
        on p.provider_specialty = b.specialty
    left join hcpcs_agg ha
        on p.npi = ha.npi
    left join specialty_hcpcs_agg sha
        on p.provider_specialty = sha.specialty
)

select * from features
