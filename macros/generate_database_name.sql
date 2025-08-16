{% macro generate_database_name(custom_database_name=none, node=none) -%}

    {%- set default_database = target.database|trim -%}
    {%- if target.name != "prd" -%}

        {{ target.database|trim }}

    {%- else -%}
        {%- if custom_database_name is not none and custom_database_name|trim != '' -%}
            {{ custom_database_name|trim }}
        {%- else -%}
            {{ default_database }}
        {%- endif -%}
    {%- endif -%}

{%- endmacro %}