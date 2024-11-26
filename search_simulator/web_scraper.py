# web_scraper.py

import requests
from bs4 import BeautifulSoup

class WebScraper:
    """
    A simple web scraper to fetch and clean the textual content of a webpage.

    Methods:
        fetch_content(url): Fetches and processes the content from the specified URL.
    """
    def fetch_content(self, url):
        """
        Fetches and extracts the main textual content from a webpage.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            dict: A dictionary containing:
                  - 'content' (str): The cleaned textual content of the webpage.
                  - 'url' (str): The URL of the scraped webpage.
            None: Returns None if an error occurs during the scraping process.

        Raises:
            requests.RequestException: If an HTTP request-related error occurs.
            Exception: For any other errors during content extraction.
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script_or_style in soup(['script', 'style', 'noscript']):
                script_or_style.decompose()

            # Extract text
            text = soup.get_text(separator=' ', strip=True)

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)

            return {'content': text, 'url': url}
        except Exception as e:
            print(f"Failed to fetch content from {url}: {e}")
            return None
