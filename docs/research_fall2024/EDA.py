import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    from wordcloud import WordCloud
    wordcloud_available = True
except ImportError:
    wordcloud_available = False
    print("WordCloud not installed. Word cloud visualization will be skipped.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk_available = True
    # Download NLTK resources if available
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except ImportError:
    nltk_available = False
    print("NLTK not installed. Advanced text analysis will be limited.")

def check_font_availability():
    """
    Check if proper fonts are available for WordCloud
    """
    if not wordcloud_available:
        return False
        
    try:
        # Try to create a small wordcloud with a simple word
        # This will fail if proper fonts aren't available
        test_cloud = WordCloud(width=100, height=100, background_color='white')
        test_cloud.generate("test")
        return True
    except Exception as e:
        print(f"WordCloud font error: {e}")
        print("WordCloud visualization will be skipped due to font issues.")
        return False

# Check font availability at import time
if wordcloud_available:
    wordcloud_fonts_available = check_font_availability()
else:
    wordcloud_fonts_available = False

def save_or_show_plot(plt, title="plot", save_plots=False, plots_dir="plots"):
    """
    Helper function to either save or show the plot
    """
    if save_plots:
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        filename = f"{plots_dir}/{title.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")
    else:
        plt.show()

def perform_eda(df, save_plots=False, plots_dir="plots"):
    """
    Perform exploratory data analysis on Digital Commonwealth data
    similar to LibRAG_EDA.ipynb
    
    Args:
        df: DataFrame containing the data
        save_plots: If True, save plots to files instead of displaying
        plots_dir: Directory to save plots if save_plots is True
    """
    print("==== LIBRAG EDA ====")
    print(f"Dataset contains {len(df)} records with {len(df.columns)} columns")

    # Basic information about the dataset
    print("\n--- Basic Dataset Information ---")
    print(f"Dataset shape: {df.shape}")
    
    # Display columns
    print("\n--- Column Information ---")
    print(df.dtypes)
    
    # Display sample data
    print("\n--- Sample Data ---")
    print(df.head(2))
    
    # Calculate missing values
    print("\n--- Missing Values Analysis ---")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    }).sort_values('Percentage', ascending=False)
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    print(missing_df)
    
    # Visualize missing data
    plt.figure(figsize=(14, 8))
    missing_df_plot = missing_df.head(20)  # Top 20 missing fields
    ax = missing_df_plot['Percentage'].plot(kind='bar')
    plt.title('Top 20 Fields with Missing Data')
    plt.ylabel('Percentage Missing')
    plt.xlabel('Fields')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    for i, v in enumerate(missing_df_plot['Percentage']):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center')
    save_or_show_plot(plt, "Missing_Data_Analysis", save_plots, plots_dir)
    
    # Top-level EDA plots
    
    # 1. Record count by institution
    if 'institution_name_ssi' in df.columns:
        print("\n--- Records by Institution ---")
        inst_counts = df['institution_name_ssi'].value_counts().head(10)
        print(inst_counts)
        
        plt.figure(figsize=(14, 8))
        ax = inst_counts.plot(kind='bar')
        plt.title('Records by Institution')
        plt.ylabel('Number of Records')
        plt.xlabel('Institution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Add value labels to the bars
        for i, v in enumerate(inst_counts):
            ax.text(i, v + 1, str(v), ha='center')
        save_or_show_plot(plt, "Records_by_Institution", save_plots, plots_dir)
    elif 'institution' in df.columns:
        print("\n--- Records by Institution ---")
        inst_counts = df['institution'].value_counts().head(10)
        print(inst_counts)
        
        plt.figure(figsize=(14, 8))
        ax = inst_counts.plot(kind='bar')
        plt.title('Records by Institution')
        plt.ylabel('Number of Records')
        plt.xlabel('Institution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Add value labels to the bars
        for i, v in enumerate(inst_counts):
            ax.text(i, v + 1, str(v), ha='center')
        save_or_show_plot(plt, "Records_by_Institution", save_plots, plots_dir)
    
    # 2. Record count by format
    format_columns = [col for col in df.columns if 'format' in col.lower()]
    if format_columns:
        format_col = format_columns[0]
        print(f"\n--- Records by Format ({format_col}) ---")
        
        # Convert list fields to strings
        if df[format_col].apply(lambda x: isinstance(x, list)).any():
            # Flatten lists
            all_formats = []
            for formats in df[format_col]:
                if isinstance(formats, list):
                    all_formats.extend(formats)
                elif formats is not None and not pd.isna(formats):
                    all_formats.append(formats)
            
            # Count formats
            format_counts = pd.Series(all_formats).value_counts().head(10)
        else:
            format_counts = df[format_col].value_counts().head(10)
        
        print(format_counts)
        
        plt.figure(figsize=(14, 8))
        ax = format_counts.plot(kind='bar')
        plt.title('Records by Format')
        plt.ylabel('Number of Records')
        plt.xlabel('Format')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Add value labels to the bars
        for i, v in enumerate(format_counts):
            ax.text(i, v + 1, str(v), ha='center')
        save_or_show_plot(plt, "Records_by_Format", save_plots, plots_dir)
    
    # 3. Record count by collection
    collection_columns = [col for col in df.columns if 'collection' in col.lower()]
    if collection_columns:
        collection_col = collection_columns[0]
        print(f"\n--- Records by Collection ({collection_col}) ---")
        
        # Convert list fields to strings
        if df[collection_col].apply(lambda x: isinstance(x, list)).any():
            # Flatten lists
            all_collections = []
            for collections in df[collection_col]:
                if isinstance(collections, list):
                    all_collections.extend(collections)
                elif collections is not None and not pd.isna(collections):
                    all_collections.append(collections)
            
            # Count collections
            collection_counts = pd.Series(all_collections).value_counts().head(10)
        else:
            collection_counts = df[collection_col].value_counts().head(10)
        
        print(collection_counts)
        
        plt.figure(figsize=(14, 8))
        ax = collection_counts.plot(kind='bar')
        plt.title('Records by Collection')
        plt.ylabel('Number of Records')
        plt.xlabel('Collection')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Add value labels to the bars
        for i, v in enumerate(collection_counts):
            ax.text(i, v + 1, str(v), ha='center')
        save_or_show_plot(plt, "Records_by_Collection", save_plots, plots_dir)
    
    # 4. Analyze years/dates if available
    date_columns = [col for col in df.columns if any(term in col.lower() for term in ['date', 'year', 'time'])]
    date_analyzed = False
    for date_col in date_columns:
        if date_col in df.columns and not date_analyzed:
            print(f"\n--- Analyzing Date Information: {date_col} ---")
            
            # Try to extract years using regex
            try:
                # Convert to string first to handle potential list values
                if df[date_col].apply(lambda x: isinstance(x, list)).any():
                    # Just use the first date in each list
                    date_values = df[date_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                else:
                    date_values = df[date_col]
                
                # Convert to string for regex extraction
                date_values = date_values.astype(str)
                years = date_values.str.extract(r'(\d{4})', expand=False)
                years = pd.to_numeric(years, errors='coerce')
                years = years.dropna()
                
                if len(years) > 0:
                    print(f"Year range: {years.min()} - {years.max()}")
                    
                    # Create decade bins for better visualization
                    decades = ((years // 10) * 10).value_counts().sort_index()
                    
                    plt.figure(figsize=(14, 8))
                    ax = decades.plot(kind='bar')
                    plt.title('Records by Decade')
                    plt.ylabel('Number of Records')
                    plt.xlabel('Decade')
                    plt.tight_layout()
                    # Add value labels to the bars
                    for i, v in enumerate(decades):
                        ax.text(i, v + 1, str(v), ha='center')
                    save_or_show_plot(plt, "Records_by_Decade", save_plots, plots_dir)
                    
                    # Also create a pie chart of centuries
                    centuries = ((years // 100) * 100).value_counts().sort_index()
                    plt.figure(figsize=(10, 10))
                    centuries.plot(kind='pie', autopct='%1.1f%%')
                    plt.title('Records by Century')
                    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    plt.tight_layout()
                    save_or_show_plot(plt, "Records_by_Century_Pie", save_plots, plots_dir)
                    
                    # Timeline plot
                    plt.figure(figsize=(14, 6))
                    plt.hist(years, bins=20, color='skyblue', edgecolor='black')
                    plt.title('Timeline of Records')
                    plt.xlabel('Year')
                    plt.ylabel('Number of Records')
                    plt.grid(axis='y', alpha=0.75)
                    plt.tight_layout()
                    save_or_show_plot(plt, "Records_Timeline", save_plots, plots_dir)
                    
                    date_analyzed = True  # Only analyze the first valid date column
            except Exception as e:
                print(f"Error analyzing date column {date_col}: {e}")
    
    # 5. Analyze subjects (key metadata for librarians)
    subject_columns = [col for col in df.columns if 'subject' in col.lower()]
    if subject_columns:
        subject_col = subject_columns[0]
        print(f"\n--- Subject Analysis ({subject_col}) ---")
        
        # Extract and flatten subjects
        all_subjects = []
        for subjects in df[subject_col]:
            if isinstance(subjects, list):
                all_subjects.extend(subjects)
            elif subjects is not None and not pd.isna(subjects):
                all_subjects.append(subjects)
        
        # Count subjects
        subject_counts = pd.Series(all_subjects).value_counts().head(20)
        print(subject_counts)
        
        # Visualize top subjects as a horizontal bar chart
        plt.figure(figsize=(14, 10))
        ax = subject_counts.head(15).plot(kind='barh')
        plt.title('Top 15 Subjects')
        plt.xlabel('Number of Records')
        plt.ylabel('Subject')
        plt.tight_layout()
        # Add value labels to the bars
        for i, v in enumerate(subject_counts.head(15)):
            ax.text(v + 0.5, i, str(v), va='center')
        save_or_show_plot(plt, "Top_Subjects_Bar", save_plots, plots_dir)
        
        # Also create a pie chart of top 8 subjects
        plt.figure(figsize=(12, 12))
        subject_counts.head(8).plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribution of Top 8 Subjects')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()
        save_or_show_plot(plt, "Top_Subjects_Pie", save_plots, plots_dir)
    
    # 6. Look for text content for potential retrieval
    print("\n--- Text Content Analysis ---")
    
    # Common text field names in Digital Commonwealth
    text_fields = [
        'description_tsim', 'abstract_tsim', 'full_text_tesim',
        'text_tesi', 'description', 'abstract', 'full_text',
        'title_info_primary_tsi', 'primary_title_tesim'  # Adding title fields
    ]
    
    # Find available text fields
    available_text_fields = [field for field in text_fields if field in df.columns]
    
    if available_text_fields:
        for field in available_text_fields:
            # Count non-null values
            non_null = df[field].notna().sum()
            print(f"Field '{field}' has {non_null} non-null values ({non_null/len(df)*100:.1f}%)")
            
            # Sample text content
            if non_null > 0:
                print("\nSample text content:")
                sample = df[df[field].notna()].sample(1).iloc[0][field]
                
                # Format sample for display (handle lists)
                if isinstance(sample, list):
                    sample_text = "\n".join(sample)
                else:
                    sample_text = str(sample)
                
                # Print truncated sample
                if len(sample_text) > 500:
                    print(f"{sample_text[:500]}...")
                else:
                    print(sample_text)
                
                # Perform keyword analysis if NLTK is available
                if nltk_available and non_null > 10:
                    print("\nKeyword Analysis:")
                    
                    # Combine all text
                    all_text = []
                    for text in df[field].dropna():
                        if isinstance(text, list):
                            all_text.extend(text)
                        else:
                            all_text.append(str(text))
                    
                    all_text_str = ' '.join(all_text)
                    
                    # Tokenize and remove stopwords
                    stop_words = set(stopwords.words('english'))
                    tokens = word_tokenize(all_text_str.lower())
                    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
                    
                    # Get word frequency
                    word_freq = Counter(filtered_tokens)
                    
                    print("\nTop 20 Keywords:")
                    for word, count in word_freq.most_common(20):
                        print(f"{word}: {count}")
                    
                    # Create a bar chart of top keywords if WordCloud isn't available
                    word_freq_df = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Count'])
                    plt.figure(figsize=(14, 10))
                    ax = sns.barplot(x='Count', y='Word', data=word_freq_df)
                    plt.title(f'Top 20 Keywords in {field}')
                    plt.tight_layout()
                    # Add value labels to the bars
                    for i, v in enumerate(word_freq_df['Count']):
                        ax.text(v + 0.5, i, str(v), va='center')
                    save_or_show_plot(plt, f"Keywords_{field}", save_plots, plots_dir)
                    
                    # Word cloud visualization
                    if wordcloud_available and wordcloud_fonts_available:
                        try:
                            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_tokens))
                            
                            plt.figure(figsize=(12, 8))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            plt.title(f'Word Cloud of {field}')
                            save_or_show_plot(plt, f"WordCloud_{field}", save_plots, plots_dir)
                        except Exception as e:
                            print(f"Error creating word cloud: {e}")
                            print("Skipping word cloud visualization.")
    else:
        print("No text content fields found in the dataset")
        
        # Create a simple text analysis from titles if available
        if 'title' in df.columns or 'title_info_primary_tsi' in df.columns:
            title_field = 'title' if 'title' in df.columns else 'title_info_primary_tsi'
            
            print(f"\nFalling back to title analysis using {title_field}")
            
            # Get all non-null titles
            titles = df[title_field].dropna()
            
            # Combine all titles into a single string
            all_titles = ' '.join(titles.astype(str))
            
            # Simple word frequency without NLTK
            # Remove special characters and split by whitespace
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
            
            # Count word frequency
            word_counts = Counter(words)
            
            # Remove common stop words manually
            common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from'}
            for word in common_words:
                if word in word_counts:
                    del word_counts[word]
            
            print("\nTop 20 words in titles:")
            for word, count in word_counts.most_common(20):
                print(f"{word}: {count}")
            
            # Create a bar chart of top words
            word_freq_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
            plt.figure(figsize=(14, 10))
            ax = sns.barplot(x='Count', y='Word', data=word_freq_df)
            plt.title('Top 20 Words in Titles')
            plt.tight_layout()
            # Add value labels to the bars
            for i, v in enumerate(word_freq_df['Count']):
                ax.text(v + 0.5, i, str(v), va='center')
            save_or_show_plot(plt, "Words_in_Titles", save_plots, plots_dir)
    
    # 7. Analyze URLs (for images and resources)
    print("\n--- URL Analysis ---")
    url_columns = [col for col in df.columns if any(term in col.lower() for term in ['url', 'link', 'image', 'thumbnail'])]
    
    if url_columns:
        # Count number of records with at least one URL
        records_with_urls = 0
        for idx, row in df.iterrows():
            has_url = False
            for col in url_columns:
                if pd.notna(row[col]):
                    has_url = True
                    break
            if has_url:
                records_with_urls += 1
                
        # Create pie chart of records with/without URLs
        plt.figure(figsize=(10, 10))
        plt.pie([records_with_urls, len(df) - records_with_urls], 
                labels=['Has URL', 'No URL'], 
                autopct='%1.1f%%', 
                colors=['#66b3ff', '#ff9999'])
        plt.title('Records with URLs')
        plt.axis('equal')
        save_or_show_plot(plt, "Records_with_URLs_Pie", save_plots, plots_dir)
        
        # Individual URL field analysis
        for url_col in url_columns:
            # Count non-null URLs
            non_null = df[url_col].notna().sum()
            print(f"Field '{url_col}' has {non_null} non-null values ({non_null/len(df)*100:.1f}%)")
            
            # Sample URLs
            if non_null > 0:
                print("\nSample URLs:")
                
                if df[url_col].apply(lambda x: isinstance(x, list)).any():
                    # Handle list values
                    samples = df[df[url_col].notna()].sample(min(3, non_null))[url_col]
                    for i, sample in enumerate(samples):
                        if isinstance(sample, list):
                            if len(sample) > 0:
                                print(f"Sample {i+1}: {sample[0]}")
                            else:
                                print(f"Sample {i+1}: Empty list")
                        else:
                            print(f"Sample {i+1}: {sample}")
                else:
                    # Regular values
                    samples = df[df[url_col].notna()].sample(min(3, non_null))[url_col]
                    for i, sample in enumerate(samples):
                        print(f"Sample {i+1}: {sample}")
    else:
        print("No URL fields found in the dataset")
    
    # 8. Create a summary visualization
    # This will show the breakdown of record counts by institution in a pie chart
    if 'institution_name_ssi' in df.columns:
        plt.figure(figsize=(12, 12))
        inst_counts = df['institution_name_ssi'].value_counts()
        # Combine all institutions with fewer than 5 records into 'Other'
        threshold = 5
        small_inst = inst_counts[inst_counts < threshold]
        if not small_inst.empty:
            inst_counts = inst_counts[inst_counts >= threshold]
            inst_counts['Other'] = small_inst.sum()
        
        inst_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Records by Institution')
        plt.axis('equal')
        save_or_show_plot(plt, "Records_by_Institution_Pie", save_plots, plots_dir)
    
    print("\n==== EDA COMPLETE ====")
    return df

def run_eda(df, save_plots=False, plots_dir="plots"):
    """
    Wrapper function to perform EDA on a DataFrame.
    This matches the function signature expected in main.py.
    
    Args:
        df: DataFrame containing the data
        save_plots: If True, save plots to files instead of displaying
        plots_dir: Directory to save plots if save_plots is True
    """
    return perform_eda(df, save_plots, plots_dir)

# If run as a standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform EDA on Digital Commonwealth data")
    parser.add_argument("--file", type=str, default="digital_commonwealth_data.csv", 
                        help="CSV file with Digital Commonwealth data")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save plots to files instead of displaying")
    parser.add_argument("--plots-dir", type=str, default="plots",
                        help="Directory to save plots if --save-plots is enabled")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.file}")
    try:
        # Load data (handles JSON encoded list fields)
        df = pd.read_csv(args.file)
        
        # Try to convert JSON-encoded lists back to Python lists
        for col in df.columns:
            try:
                # Check if column might contain JSON lists
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(sample, str) and sample.startswith('[') and sample.endswith(']'):
                    df[col] = df[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x
                    )
            except:
                # Skip if conversion fails
                continue
        
        # Run EDA
        perform_eda(df, args.save_plots, args.plots_dir)
    except Exception as e:
        print(f"Error loading or analyzing data: {e}")