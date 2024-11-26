import requests
import json
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

class GoogleAPI:
    """
        A class to interact with the Google Custom Search JSON API.

        Attributes:
            api_key (str): The API key for accessing the Google Custom Search API.
            cse_id (str): The Custom Search Engine ID (CSE ID) for the specific search engine.
    """
    def __init__(self, api_key, cse_id):
        """
        Initializes the GoogleAPI instance with an API key and a Custom Search Engine ID (CSE ID).

        Args:
            api_key (str): The API key for accessing the Google Custom Search API.
            cse_id (str): The Custom Search Engine ID (CSE ID) for the specific search engine.
        """
        self.api_key = api_key
        self.cse_id = cse_id
        # print(api_key == None,cse_id == None)
    
    def search(self, query, num_results=10):
        """
        Performs a search query using the Google Custom Search JSON API.

        Args:
            query (str): The search query string.
            num_results (int, optional): The maximum number of results to retrieve (default is 10).
                                         Note: The API allows a maximum of 10 results per request.

        Returns:
            list: A list of search result items (if any), or an empty list if no results are found or an error occurs.
                  Each item is a dictionary containing details about the search result.
        """
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'num': min(num_results, 10),  # Ensure num_results is no more than 10
        }
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            try:
                results = response.json()
                if 'items' in results:
                    return results['items']
                else:
                    print("No search results found.")
                    return []
            except json.JSONDecodeError:
                print("Error: Response is not valid JSON.")
                return []
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print(response.text)
            return []
        
if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    api_key = OPENAI_API_KEY# Replace with your actual API key
    cx_key =  GOOGLE_CSE_ID     # Replace with your actual CSE ID
    search_eng = GoogleAPI(api_key, cx_key)
    results = search_eng.search("Best games to play right now", num_results=10)
    
    for result in results:
        print(result['link'])  # Prints the links of the search results
