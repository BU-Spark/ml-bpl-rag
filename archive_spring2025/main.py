import os
import argparse
import pandas as pd
import json

# Import our custom modules
from API_utils import (
    get_all_records, get_records_with_diverse_queries, 
    save_records_to_csv, load_records_from_csv,
    save_to_json, load_from_json, create_bpl_data
)
from EDA import run_eda

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Digital Commonwealth Data Analysis')
    
    # Data source options
    parser.add_argument('--create-json', action='store_true',
                        help='Create bpl_data.json using diverse queries')
    parser.add_argument('--json-file', type=str, default='bpl_data.json',
                        help='JSON file to load or create (default: bpl_data.json)')
    parser.add_argument('--csv-file', type=str, default='digital_commonwealth_data.csv',
                        help='CSV file to save/load data (default: digital_commonwealth_data.csv)')
    
    # Query options
    parser.add_argument('--queries', type=str, nargs='+',
                        help='Custom search queries (default: uses diverse set)')
    parser.add_argument('--max-per-query', type=int, default=50,
                        help='Maximum records per query (default: 50)')
    parser.add_argument('--max-records', type=int, default=500,
                        help='Maximum total records to retrieve (default: 500)')
    
    # Processing options
    parser.add_argument('--skip-api', action='store_true',
                        help='Skip API call and use existing data file')
    parser.add_argument('--skip-eda', action='store_true',
                        help='Skip EDA analysis')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of displaying')
    
    return parser.parse_args()

def main():
    """Main function to run the data pipeline"""
    args = parse_arguments()
    
    # Step 1: Get data (either from API or from file)
    if args.skip_api:
        # Try to load existing data
        if os.path.exists(args.json_file):
            print(f"Loading JSON data from: {args.json_file}")
            data = load_from_json(args.json_file)
            if data is None:
                print(f"Failed to load JSON data. Exiting.")
                return None
        elif os.path.exists(args.csv_file):
            print(f"Loading CSV data from: {args.csv_file}")
            df = load_records_from_csv(args.csv_file)
            if df is None:
                print(f"Failed to load CSV data. Exiting.")
                return None
            return df
        else:
            print(f"No data files found ({args.json_file} or {args.csv_file}). Please run without --skip-api first.")
            return None
    elif args.create_json or not os.path.exists(args.json_file):
        # Fetch new data from API
        if args.queries:
            print(f"Retrieving data from API for custom queries: {args.queries}")
            all_records = []
            for query in args.queries:
                records = get_all_records(query, max_records=args.max_per_query)
                if records:
                    all_records.extend(records)
            
            # Remove duplicates
            data = list({r.get('id'): r for r in all_records if r.get('id')}.values())
        else:
            print("Retrieving data using diverse query set")
            data = get_records_with_diverse_queries(max_records_per_query=args.max_per_query)
        
        if data is None or len(data) == 0:
            print("No records found. Exiting.")
            return None
            
        print(f"Retrieved {len(data)} records")
        
        # Save to JSON
        save_to_json(data, args.json_file)
    else:
        # Load existing JSON file
        print(f"Loading existing JSON data from: {args.json_file}")
        data = load_from_json(args.json_file)
        if data is None:
            print(f"Failed to load JSON data. Exiting.")
            return None
    
    # Convert to DataFrame
    try:
        df = pd.DataFrame(data)
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error converting data to DataFrame: {e}")
        return None
    
    # Save to CSV for future use
    save_records_to_csv(data, args.csv_file)
    
    # Exit if we couldn't load data
    if df is None or len(df) == 0:
        print("No data available for analysis. Exiting.")
        return None
    
    # Step 2: Run EDA if not skipped
    if not args.skip_eda:
        print("\nRunning Exploratory Data Analysis:")
        
        # Configure matplotlib for saving plots if requested
        if args.save_plots:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Override plt.show to save files instead
            original_show = plt.show
            def save_figure(filename=None):
                if filename is None:
                    # Generate a filename based on the current figure number
                    filename = f"plots/figure_{plt.gcf().number}.png"
                plt.savefig(filename)
                plt.close()
            plt.show = save_figure
        
        # Run the EDA
        df = run_eda(df)
        
        # Restore original plt.show if we modified it
        if args.save_plots:
            plt.show = original_show
            print(f"Plots saved to 'plots/' directory")
    
    print("\nAnalysis complete!")
    return df

if __name__ == "__main__":
    df = main()
    # The dataframe is returned in case you want to use it in an interactive session
    print("\nYou can access the data through the 'df' variable")
