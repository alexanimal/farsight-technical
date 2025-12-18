SELECT
    c.column_name,
    c.data_type,
    c.is_nullable,
    c.column_default,
    c.character_maximum_length,
    c.numeric_precision,
    c.numeric_scale,
    tc.constraint_type AS column_key,
    pgd.description AS column_comment
FROM
    information_schema.columns c
LEFT JOIN
    information_schema.key_column_usage kcu
    ON c.table_name = kcu.table_name
    AND c.table_schema = kcu.table_schema
    AND c.column_name = kcu.column_name
LEFT JOIN
    information_schema.table_constraints tc
    ON kcu.constraint_name = tc.constraint_name
    AND kcu.table_schema = tc.table_schema
LEFT JOIN
    pg_catalog.pg_statio_user_tables AS st
    ON c.table_name = st.relname
    AND c.table_schema = st.schemaname
LEFT JOIN
    pg_catalog.pg_description pgd
    ON pgd.objoid = st.relid
    AND pgd.objsubid = c.ordinal_position
WHERE
    c.table_name = 'organizations'  -- Replace with your table name
    AND c.table_schema = 'public'     -- Replace if using a different schema
ORDER BY
    c.ordinal_position;