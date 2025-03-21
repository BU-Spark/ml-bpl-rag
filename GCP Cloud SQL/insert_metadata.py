import ijson
import psycopg2
import os
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg2 import pool

# Load environment variables
load_dotenv()

# Database connection details
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")  # Default to local proxy
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = "bpl_metadata"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")

# File path
FILE_PATH = "bpl_data.json"

# Number of worker threads
NUM_THREADS = 8  # Try increasing to 8-10 for better performance

# Connection Pool (reuse connections instead of creating new ones)
DB_POOL = psycopg2.pool.ThreadedConnectionPool(
    minconn=NUM_THREADS, 
    maxconn=NUM_THREADS * 2,  # Allow more connections
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

# SQL Insert Query
INSERT_QUERY = """
INSERT INTO metadata (
    id, title, abstract, subjects, collection, date, institution, metadata_url, image_url, 
    physical_location, identifier_local_other, identifier_uri, identifier_uri_preview, note, rights,
    license, reuse_allowed, digital_origin, extent, type_of_resource, lang_term, publishing_state,
    processing_state, destination_site, hosting_status, harvesting_status, oai_header_id, exemplary_image,
    exemplary_image_key_base, admin_set_name, admin_set_ark_id, institution_ark_id, collection_ark_id,
    filenames, subject_geographic, subject_coordinates, subject_point, subject_geojson, subject_hiergeo, timestamp
) VALUES (
    %(id)s, %(title)s, %(abstract)s, %(subjects)s, %(collection)s, %(date)s, %(institution)s, %(metadata_url)s, %(image_url)s, 
    %(physical_location)s, %(identifier_local_other)s, %(identifier_uri)s, %(identifier_uri_preview)s, %(note)s, %(rights)s,
    %(license)s, %(reuse_allowed)s, %(digital_origin)s, %(extent)s, %(type_of_resource)s, %(lang_term)s, %(publishing_state)s,
    %(processing_state)s, %(destination_site)s, %(hosting_status)s, %(harvesting_status)s, %(oai_header_id)s, %(exemplary_image)s,
    %(exemplary_image_key_base)s, %(admin_set_name)s, %(admin_set_ark_id)s, %(institution_ark_id)s, %(collection_ark_id)s,
    %(filenames)s, %(subject_geographic)s, %(subject_coordinates)s, %(subject_point)s, %(subject_geojson)s, %(subject_hiergeo)s, %(timestamp)s
) ON CONFLICT (id) DO NOTHING;
"""

def process_batch(batch):
    """ Inserts a batch into PostgreSQL using connection pool """
    if not batch:
        return

    conn = None
    try:
        conn = DB_POOL.getconn()  # Get connection from pool
        cur = conn.cursor()
        cur.executemany(INSERT_QUERY, batch)
        conn.commit()
        cur.close()
        print(f"Inserted {len(batch)} records")
    except Exception as e:
        print(f"Error inserting batch: {e}")
    finally:
        if conn:
            DB_POOL.putconn(conn)  # Return connection to pool

def stream_json(file_path):
    """ Generator to stream JSON data from the file efficiently """
    with open(file_path, "r") as f:
        parser = ijson.items(f, "Data.item.data.item")  # Stream JSON
        for item in parser:
            attributes = item.get("attributes", {})

            yield {
                "id": item["id"],
                "title": attributes.get("title_info_primary_tsi", ""),
                "abstract": attributes.get("abstract_tsi", ""),
                "subjects": attributes.get("subject_facet_ssim", []),
                "collection": attributes.get("collection_name_ssim", [""])[0],
                "date": attributes.get("date_edtf_ssm", [""])[0],
                "institution": attributes.get("institution_name_ssi", ""),
                "metadata_url": attributes.get("identifier_uri_ss", ""),
                "image_url": attributes.get("identifier_uri_preview_ss", ""),
                "physical_location": attributes.get("physical_location_ssim", []),
                "identifier_local_other": attributes.get("identifier_local_other_tsim", []),
                "identifier_uri": attributes.get("identifier_uri_ss", ""),
                "identifier_uri_preview": attributes.get("identifier_uri_preview_ss", ""),
                "note": attributes.get("note_tsim", []),
                "rights": attributes.get("rights_ss", ""),
                "license": attributes.get("license_ss", ""),
                "reuse_allowed": attributes.get("reuse_allowed_ssi", ""),
                "digital_origin": attributes.get("digital_origin_ssi", ""),
                "extent": attributes.get("extent_tsi", ""),
                "type_of_resource": attributes.get("type_of_resource_ssim", []),
                "lang_term": attributes.get("lang_term_ssim", []),
                "publishing_state": attributes.get("publishing_state_ssi", ""),
                "processing_state": attributes.get("processing_state_ssi", ""),
                "destination_site": attributes.get("destination_site_ssim", []),
                "hosting_status": attributes.get("hosting_status_ssi", ""),
                "harvesting_status": attributes.get("harvesting_status_bsi", False),
                "oai_header_id": attributes.get("oai_header_id_ssi", ""),
                "exemplary_image": attributes.get("exemplary_image_ssi", ""),
                "exemplary_image_key_base": attributes.get("exemplary_image_key_base_ss", ""),
                "admin_set_name": attributes.get("admin_set_name_ssi", ""),
                "admin_set_ark_id": attributes.get("admin_set_ark_id_ssi", ""),
                "institution_ark_id": attributes.get("institution_ark_id_ssi", ""),
                "collection_ark_id": attributes.get("collection_ark_id_ssim", []),
                "filenames": attributes.get("filenames_ssim", []),
                "subject_geographic": attributes.get("subject_geographic_sim", []),
                "subject_coordinates": attributes.get("subject_coordinates_geospatial", []),
                "subject_point": attributes.get("subject_point_geospatial", []),
                "subject_geojson": json.dumps(attributes.get("subject_geojson_facet_ssim", {})),
                "subject_hiergeo": json.dumps(attributes.get("subject_hiergeo_geojson_ssm", {})),
                "timestamp": attributes.get("timestamp", "")
            }

def multi_threaded_insert():
    """ Uses ThreadPoolExecutor to insert data in parallel """
    batch_size = 1000  # Adjust batch size for better performance
    batch = []
    futures = []

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for record in stream_json(FILE_PATH):
            batch.append(record)

            if len(batch) >= batch_size:
                futures.append(executor.submit(process_batch, batch))
                batch = []  # Reset batch
        
        # Insert remaining records
        if batch:
            futures.append(executor.submit(process_batch, batch))

        # Wait for all threads to complete
        for future in as_completed(futures):
            future.result()

# Start the multi-threaded insertion
multi_threaded_insert()

# Close the connection pool
DB_POOL.closeall()