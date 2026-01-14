CREATE OR REPLACE FUNCTION flatten_jsonb_array(j JSONB)
RETURNS TEXT LANGUAGE SQL IMMUTABLE AS $$
  SELECT CASE
    WHEN j IS NULL THEN ''
    WHEN jsonb_typeof(j) = 'array'
      THEN array_to_string(ARRAY(SELECT jsonb_array_elements_text(j)), ' ')
    ELSE j::text
  END;
$$;
