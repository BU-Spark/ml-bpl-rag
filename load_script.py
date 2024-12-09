

import json

import time

import os

import sys

import requests


def fetch_digital_commonwealth():

    start = time.time()

    BASE_URL = "https://www.digitalcommonwealth.org/search.json?search_field=all_fields&per_page=100&q="

    PAGE = sys.argv[1]

    END_PAGE = sys.argv[2]

    file_name = f"out{PAGE}_{END_PAGE}.json"

    FINAL_PAGE = 13038

    output = []

    file_path = f"./{file_name}"

    # file_path = './output.json'

    if os.path.exists(file_path):

        with open(file_path,'r') as file:

            output = json.load(file)

            if int(PAGE) < (len(output) + 1):

                PAGE = len(output) + 1

    

    if int(PAGE) >= int(END_PAGE):

        return None

    print(f'Reading page {PAGE} up to page {END_PAGE}')

    retries = 0

    while True:

        try:

            response = requests.get(f"{BASE_URL}&page={PAGE}")

            response.raise_for_status()

            data = response.json()

            

            # Append current page data to the output list

            output.append(data)

            

            # Save the entire output to a JSON file after each iteration

            with open(file_path, 'w') as f:

                json.dump(output, f)





            # check if theres a next page

            # print(len(response))

            if data['meta']['pages']['next_page']:

                if data['meta']['pages']['next_page'] == int(END_PAGE):

                    print(f"Processed and saved page {PAGE}. Total pages saved: {len(output)}")

                    break

                elif data['meta']['pages']['next_page'] == FINAL_PAGE:

                    print(f"finished page {PAGE}")

                    PAGE = FINAL_PAGE

                else:

                    print(f"finished page {PAGE}")

                    PAGE = data['meta']['pages']['next_page']

            else:

                print(f"Processed and saved page {PAGE}. Total pages saved: {len(output)}")

                break

            

            retries = 0

            # Optional: Add a small delay to avoid overwhelming the API

            # time.sleep(0.5)

        except requests.exceptions.RequestException as e:

            print(f"An error occurred: {e}")

            print(f"Processed and saved page {PAGE}. Total pages saved: {len(output)}")

            retries += 1

            if retries >= 5:

                break

    end = time.time()

    print(f"Timer: {end - start}")

    print(f"Finished processing all pages. Total pages saved: {len(output)}")

if __name__ == "__main__":

    fetch_digital_commonwealth()
