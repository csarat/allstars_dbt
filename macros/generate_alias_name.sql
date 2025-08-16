{% macro generate_alias_name(custom_alias_name=none, node=none) -%}
    {%- set custom_database_name = node.config.get('database', '') -%}
    {%- set custom_schema_name = node.config.get('schema', '') -%}

    {%- if target.name != "prd" and custom_database_name is not none and custom_schema_name is not none -%}

        {%- set custom_prefix = custom_database_name ~ "__" ~ custom_schema_name ~ "__" -%}

    {%- else -%}

        {%- set custom_prefix = '' -%}

    {%- endif -%}

    {%- if custom_alias_name -%}

        {{ custom_prefix }}{{ custom_alias_name | trim }}

    {%- elif none.version -%}

        {{ custom_prefix }}{{ return(node.name ~ "-v" ~ (node.version | replace(".", "_"))) }}

    {%- else -%}

        {{custom_prefix}}{{ node.name }}

    {%- endif -%}

{%- endmacro %}