import json
import requests
import sys
import logging

# Initialize logging
logging.basicConfig(filename='text_load.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

i_begin = int(sys.argv[1])  # Convert to integer
i_end = int(sys.argv[2])

def load_full_text(i_begin,i_end):

    file_path_write = '/projectnb/sparkgrp/ml-bpl-rag-data/text/'
    counter = 0
    text_counter = 0
    # Function to get the full text
    def get_text(url):
        url = url + '/text'
        try:
            response = requests.get(url, timeout=10)  # Timeout to prevent hanging
            if response.status_code == 200:
                return response.text
            else:
                logging.warning(f"Error {response.status_code} for URL {url}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for URL {url}: {e}")
            return None
    
    full_text_locate = {}
    
    # Load metadata
    with open("/projectnb/sparkgrp/ml-bpl-rag-data/bpl_data.json", 'r') as f:
        bpl_meta = json.load(f)
    
    # Ensure valid range
    if i_begin < 0:
        i_begin = 0
    if (i_end > (len(bpl_meta['Data']) - 1)) or (i_end < i_begin):
        i_end = len(bpl_meta['Data']) - 1    
    try:
        # Process the items within the specified range
        for item in list(bpl_meta['Data'])[i_begin:i_end]:
            if ('has_transcription_bsi' in item['attributes']) and ('identifier_uri_ss' in item['attributes']):
                full_text_locate[item['id']] = {
                    'text': get_text(item['attributes']['identifier_uri_ss'])
                }
                text_counter += 1
            counter += 1  # Increment counter for every processed item

            # Save checkpoint every 3000 items
            if counter % 50000 == 0:
                with open(f'{file_path_write}ft_{i_end//100000}_checkpoint_{str(counter // 50000)}_{text_counter}.json', 'w') as check:
                    json.dump(full_text_locate, check)
                full_text_locate.clear()  # Clear the dictionary to free memory
                text_counter = 0
    
    except Exception as e:
        with open(f'{file_path_write}checkpoint_interrupted.json', 'w') as check:
            json.dump(full_text_locate, check)
        logging.error(f"Process interrupted: {e}")
    
    # Save final checkpoint
    with open(f'{file_path_write}ft_{i_end//100000}_checkpoint_end_{text_counter}.json', 'w') as check:
        json.dump(full_text_locate, check)
    
    print(f"Checked in {counter} texts")

if __name__ == "__main__":
    load_full_text(i_begin,i_end)

