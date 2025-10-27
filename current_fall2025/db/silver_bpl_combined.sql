DROP TABLE IF EXISTS silver.bpl_combined;

CREATE TABLE silver.bpl_combined AS
WITH parsed_dates AS (
    SELECT
        id,
        -- Extract all 4-digit years from date_tsim
        regexp_matches(
            flatten_jsonb_array(data->'date_tsim'), 
            '\d{4}', 
            'g'
        ) AS year_matches,
        -- Check if it's a circa date
        flatten_jsonb_array(data->'date_tsim') LIKE '%ca.%' AS is_circa
    FROM bronze.bpl_metadata
),
date_ranges AS (
    SELECT
        id,
        MIN((year_matches)[1]::int) AS raw_start,
        MAX((year_matches)[1]::int) AS raw_end,
        BOOL_OR(is_circa) AS has_circa  -- TRUE if any date has "ca."
    FROM parsed_dates
    GROUP BY id
)
SELECT
    b.id AS document_id,
    CONCAT_WS(
      ' ',
      'Title:',      b.data->>'title_info_primary_tsi',
      'Subtitle:',   b.data->>'title_info_primary_subtitle_tsi',
      'Abstract:',   b.data->>'abstract_tsi',
      'Notes:',      flatten_jsonb_array(b.data->'note_tsim'),
      'Subjects:',   flatten_jsonb_array(b.data->'subject_topic_tsim'),
      'People:',     flatten_jsonb_array(b.data->'subject_name_tsim'),
      'Locations:',  flatten_jsonb_array(b.data->'subject_geographic_sim'),
      'Date:',       flatten_jsonb_array(b.data->'date_tsim'),
      'Type:',       flatten_jsonb_array(b.data->'type_of_resource_ssim'),
      'Collection:', flatten_jsonb_array(b.data->'collection_name_ssim')
    ) AS summary_text,
    b.data AS metadata,
    -- Handle circa in one go: if circa, subtract 5, else use raw value
    CASE 
        WHEN dr.has_circa THEN dr.raw_start - 5 
        ELSE dr.raw_start 
    END AS date_start,
    CASE 
        WHEN dr.has_circa THEN dr.raw_end + 5 
        ELSE dr.raw_end 
    END AS date_end
FROM bronze.bpl_metadata b
LEFT JOIN date_ranges dr ON b.id = dr.id;

CREATE INDEX idx_bpl_combined_dates ON silver.bpl_combined(date_start, date_end);


