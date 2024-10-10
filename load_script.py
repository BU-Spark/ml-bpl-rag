import requests

import json

import time

import sys



def fetch_digital_commonwealth():

    BASE_URL = "https://www.digitalcommonwealth.org/search.json?search_field=all_fields&per_page=100&q="

    PAGE = sys.argv[1]

    END_PAGE = sys.argv[2]

    output = []

    print(f'Reading page {PAGE} up to page {END_PAGE}')

    while True:

        try:

            response = requests.get(f"{BASE_URL}&page={PAGE}")

            response.raise_for_status()

            data = response.json()

            

            # Append current page data to the output list

            output.append(data)

            

            # Save the entire output to a JSON file after each iteration

            with open('output.json', 'w') as f:

                json.dump(output, f)

            

            # print(f"Processed and saved page {PAGE}. Total pages saved: {len(output)}")



            # Check if there is a next page

            # print(f"next page {data['meta']['pages']['next_page']} of type {type(data['meta']['pages']['next_page'])}")

            if data['meta']['pages']['next_page']:

                if data['meta']['pages']['next_page'] == int(END_PAGE):

                    print(f"Processed and saved page {PAGE}. Total pages saved: {len(output)}")

                    break

                else:

                    PAGE = data['meta']['pages']['next_page']

            else:

                print(f"Processed and saved page {PAGE}. Total pages saved: {len(output)}")

                break



            # Optional: Add a small delay to avoid overwhelming the API

            # time.sleep(0.5)



        except requests.exceptions.RequestException as e:

            print(f"An error occurred: {e}")

            print(f"Processed and saved page {PAGE}. Total pages saved: {len(output)}")

            break



    print(f"Finished processing all pages. Total pages saved: {len(output)}")



if __name__ == "__main__":

    fetch_digital_commonwealth()
