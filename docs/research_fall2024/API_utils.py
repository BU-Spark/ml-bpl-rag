import requests
import json
import pandas as pd
import time
import os

def get_records(query=None, filter_queries=None, per_page=20, page=1):
    """
    Retrieves records from the Digital Commonwealth API with enhanced filtering options.
    
    Args:
        query (str, optional): The main search query
        filter_queries (dict, optional): Dictionary of additional filters
        per_page (int): Number of results per page
        page (int): Page number
        
    Returns:
        dict: JSON response from the API
    """
    base_url = "https://www.digitalcommonwealth.org/search.json"
    
    # Base parameters
    params = {
        "per_page": per_page,
        "page": page
    }
    
    # Add main search query if provided
    if query:
        params["q"] = query
        params["search_field"] = "all_fields"
    
    # Add any filter queries
    if filter_queries:
        for key, value in filter_queries.items():
            params[key] = value
    
    try:
        print(f"Making API request with params: {params}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        json_data = response.json()
        
        # Check the structure of the response
        if "response" in json_data:
            records_count = len(json_data["response"].get("docs", []))
            total_count = json_data["response"].get("numFound", 0)
            print(f"Page {page}: Retrieved {records_count} of {total_count} total records")
        else:
            print(f"Page {page}: Retrieved data with unexpected structure")
            
        return json_data
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None

def get_records_with_diverse_queries(max_records_per_query=100):
    """
    Fetches records using a diverse set of queries to get a representative sample.
    
    Args:
        max_records_per_query (int): Maximum number of records to fetch per query
        
    Returns:
        list: Combined list of unique records
    """
    # List of diverse queries covering different topics, collections, and formats
    query_groups = [
        # General geographic/historical queries
        ["Boston", "Massachusetts", "New England"],
        
        # Specific collections
        ["Stephen Lewis poster collection", "anti-war posters", "immigration posters"],
        
        # Topics and subjects
        ["Civil War", "Women's rights", "Abolition", "Labor movement"],
        
        # Time periods
        ["19th century", "Great Depression", "World War"],
        
        # Formats
        ["photographs", "manuscripts", "maps", "newspapers"]
    ]
    
    # Special filter queries for specific collections or institutions
    filter_groups = [
        # Example: Specific institutions
        {"institution_name_ssi": "University of Massachusetts Boston, Joseph P. Healey Library"},
        {"institution_name_ssi": "Boston Public Library"},
        
        # Example: Specific formats
        {"genre_basic_ssim": "Posters"},
        {"genre_basic_ssim": "Photographs"}
    ]
    
    all_records = []
    unique_ids = set()
    
    # Process standard queries
    for query_group in query_groups:
        for query in query_group:
            print(f"\nFetching records for query: {query}")
            records = get_all_records(query, max_records=max_records_per_query)
            
            if records:
                # Filter out duplicates
                new_records = [r for r in records if r.get('id') not in unique_ids]
                unique_ids.update([r.get('id') for r in new_records])
                
                all_records.extend(new_records)
                print(f"Added {len(new_records)} new unique records from query '{query}'")
                print(f"Total unique records so far: {len(unique_ids)}")
    
    # Process filter queries
    for filters in filter_groups:
        print(f"\nFetching records with filters: {filters}")
        records = get_all_records(query=None, filter_queries=filters, max_records=max_records_per_query)
        
        if records:
            # Filter out duplicates
            new_records = [r for r in records if r.get('id') not in unique_ids]
            unique_ids.update([r.get('id') for r in new_records])
            
            all_records.extend(new_records)
            print(f"Added {len(new_records)} new unique records from filters {filters}")
            print(f"Total unique records so far: {len(unique_ids)}")
    
    print(f"\nCompleted fetching data. Total unique records: {len(all_records)}")
    return all_records

def get_all_records(query=None, filter_queries=None, max_records=None):
    """
    Retrieves all records for given criteria, handling pagination.
    
    Args:
        query (str, optional): The main search query
        filter_queries (dict, optional): Dictionary of additional filters
        max_records (int, optional): Maximum number of records to retrieve
        
    Returns:
        list: Combined list of records
    """
    all_records = []
    page = 1
    total_retrieved = 0
    per_page = 20
    
    while True:
        data = get_records(query, filter_queries, per_page, page)
        if not data:
            break
        
        # Handle the Digital Commonwealth API response structure
        if "response" in data:
            records = data["response"].get("docs", [])
            num_found = data["response"].get("numFound", 0)
        else:
            records = data.get("data", [])
            num_found = data.get("meta", {}).get("total_count", 0)
        
        if not records:
            print("No records found in this page")
            break
            
        print(f"Found {len(records)} records on page {page}")
        
        # Process records to ensure correct structure
        processed_records = []
        for record in records:
            # Ensure record is a dictionary
            if isinstance(record, dict):
                # If the record has 'attributes', extract those to top level
                if 'attributes' in record:
                    processed_record = record['attributes']
                    processed_record['id'] = record.get('id', '')
                    if 'links' in record:
                        processed_record['links'] = record['links']
                    processed_records.append(processed_record)
                else:
                    processed_records.append(record)
            else:
                print(f"Skipping invalid record format: {type(record)}")
        
        all_records.extend(processed_records)
        total_retrieved += len(processed_records)
        
        # Print progress
        print(f"Progress: {total_retrieved}/{num_found} records retrieved")
        
        # Check if we've reached the maximum number of records
        if max_records and total_retrieved >= max_records:
            all_records = all_records[:max_records]  # Trim the list if needed
            print(f"Reached maximum requested records: {max_records}")
            break
            
        # Check if we've retrieved all available records
        if total_retrieved >= num_found:
            print("Retrieved all available records")
            break
            
        # Move to next page
        page += 1
        
        # Add a small delay to be nice to the API
        time.sleep(0.5)
    
    print(f"Total records retrieved: {len(all_records)}")
    return all_records

def fetch_full_metadata(record_id):
    """
    Fetches the full metadata record for an item.
    
    Args:
        record_id (str): The ID of the record
        
    Returns:
        dict: Full metadata record
    """
    url = f"https://www.digitalcommonwealth.org/search/{record_id}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error fetching metadata: {e}")
        return None

def save_to_json(data, filename="bpl_data.json"):
    """
    Saves data to a JSON file
    
    Args:
        data: The data to save
        filename (str): The filename to save to
        
    Returns:
        bool: Success status
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved data to {filename}")
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def load_from_json(filename="bpl_data.json"):
    """
    Loads data from a JSON file
    
    Args:
        filename (str): The filename to load from
        
    Returns:
        The loaded data or None if error
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {filename}")
        return data
    except Exception as e:
        print(f"Error loading from JSON: {e}")
        return None

def save_records_to_csv(records, filename="digital_commonwealth_data.csv"):
    """
    Saves records to a CSV file, handling complex nested structures
    
    Args:
        records (list): List of record dictionaries
        filename (str): Output filename
        
    Returns:
        bool: Success status
    """
    if not records:
        print("No records to save")
        return False
        
    try:
        # Convert records to DataFrame
        df = pd.DataFrame(records)
        
        # Handle list fields for CSV storage
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
        
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Successfully saved {len(records)} records to {filename}")
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

def load_records_from_csv(filename="digital_commonwealth_data.csv"):
    """
    Loads records from a CSV file and restores list fields
    
    Args:
        filename (str): The CSV file to load
        
    Returns:
        pandas.DataFrame: The loaded data
    """
    try:
        df = pd.read_csv(filename, encoding='utf-8')
        
        # Restore list fields from JSON strings
        for col in df.columns:
            # Check if column contains JSON strings representing lists
            try:
                first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(first_valid, str) and first_valid.startswith('[') and first_valid.endswith(']'):
                    df[col] = df[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x
                    )
            except (ValueError, IndexError):
                continue
                
        print(f"Successfully loaded {len(df)} records from {filename}")
        print(f"Columns in the loaded data: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        return df
    except Exception as e:
        print(f"Error loading from CSV: {e}")
        return None

def create_bpl_data():
    """
    Creates the bpl_data.json file by fetching diverse records
    
    Returns:
        str: Path to the created file
    """
    print("Creating bpl_data.json file with diverse queries...")
    
    # Get diverse records
    records = get_records_with_diverse_queries(max_records_per_query=50)
    
    if records:
        # Save to JSON
        save_to_json(records, "bpl_data.json")
        return os.path.abspath("bpl_data.json")
    else:
        print("Failed to retrieve records")
        return None

# If run directly, create the bpl_data.json file
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch and save data from Digital Commonwealth")
    parser.add_argument("--output", type=str, default="bpl_data.json",
                      help="Output JSON file (default: bpl_data.json)")
    parser.add_argument("--queries", type=str, nargs="+", 
                      help="Custom search queries to use instead of default diverse set")
    parser.add_argument("--max-per-query", type=int, default=100,
                      help="Maximum records per query (default: 100)")
    
    args = parser.parse_args()
    
    if args.queries:
        print(f"Using custom queries: {args.queries}")
        all_records = []
        for query in args.queries:
            records = get_all_records(query, max_records=args.max_per_query)
            if records:
                all_records.extend(records)
        
        # Remove duplicates
        unique_records = {r.get('id'): r for r in all_records if r.get('id')}.values()
        save_to_json(list(unique_records), args.output)
    else:
        print("Using diverse set of queries")
        records = get_records_with_diverse_queries(max_records_per_query=args.max_per_query)
        save_to_json(records, args.output)

