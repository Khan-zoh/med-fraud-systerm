/*
    Intermediate: Provider HCPCS Code Mix

    Captures each provider's procedure distribution — the fraction of their
    total services devoted to each HCPCS code. This feeds the Jensen-Shannon
    divergence calculation in the marts layer (peer benchmarking feature).

    Also flags high-complexity E&M codes for upcoding detection.
*/

with claims as (
    select * from {{ ref('stg_partb_claims') }}
),

provider_totals as (
    select
        npi,
        sum(total_services) as provider_total_services
    from claims
    group by npi
),

hcpcs_mix as (
    select
        c.npi,
        c.hcpcs_code,
        c.provider_type,
        sum(c.total_services)                               as hcpcs_services,
        sum(c.total_services) / pt.provider_total_services  as hcpcs_fraction,

        -- High-complexity E&M flag (99214-99215, 99244-99245, 99284-99285)
        -- These are the codes most commonly associated with upcoding
        case
            when c.hcpcs_code in (
                '99215', '99214',           -- Office visits high complexity
                '99245', '99244',           -- Consultations high complexity
                '99285', '99284',           -- ED visits high complexity
                '99223', '99233',           -- Inpatient high complexity
                '99291'                     -- Critical care
            ) then true
            else false
        end                                                 as is_high_complexity

    from claims c
    inner join provider_totals pt
        on c.npi = pt.npi
    group by
        c.npi,
        c.hcpcs_code,
        c.provider_type,
        pt.provider_total_services
)

select * from hcpcs_mix
