/*
    Macro: safe_zscore

    Computes a z-score (value - mean) / stddev with null-safety.
    Returns null when stddev is zero (all peers have the same value)
    rather than dividing by zero.

    Usage:
        {{ safe_zscore('total_services', 'avg_total_services', 'std_total_services') }}

    IE context: this is the same standardization used in SPC (Statistical
    Process Control) to determine if a measurement falls within control limits.
*/

{% macro safe_zscore(value_col, mean_col, std_col) %}
    ({{ value_col }} - {{ mean_col }}) / nullif({{ std_col }}, 0)
{% endmacro %}
