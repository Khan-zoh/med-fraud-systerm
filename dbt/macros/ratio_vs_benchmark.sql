/*
    Macro: ratio_vs_benchmark

    Computes the ratio of a provider metric to a benchmark (typically
    the specialty median). Returns null when benchmark is zero.

    Usage:
        {{ ratio_vs_benchmark('total_services', 'median_total_services') }}
*/

{% macro ratio_vs_benchmark(value_col, benchmark_col) %}
    {{ value_col }} / nullif({{ benchmark_col }}, 0)
{% endmacro %}
