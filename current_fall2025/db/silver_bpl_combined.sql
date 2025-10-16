CREATE TABLE silver.bpl_combined AS
SELECT
    id AS document_id,
    CONCAT_WS(
      ' ',
      'Title:',      data->>'title_info_primary_tsi',
      'Subtitle:',   data->>'title_info_primary_subtitle_tsi',
      'Abstract:',   data->>'abstract_tsi',
      'Notes:',      flatten_jsonb_array(data->'note_tsim'),
      'Subjects:',   flatten_jsonb_array(data->'subject_topic_tsim'),
      'People:',     flatten_jsonb_array(data->'subject_name_tsim'),
      'Locations:',  flatten_jsonb_array(data->'subject_geographic_sim'),
      'Date:',       flatten_jsonb_array(data->'date_tsim'),
      'Type:',       flatten_jsonb_array(data->'type_of_resource_ssim'),
      'Collection:', flatten_jsonb_array(data->'collection_name_ssim')
    ) AS summary_text
FROM bronze.bpl_metadata;
