# web_scraper.py
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests

# Configure logging
logging.basicConfig(
    filename="web_scraper.log",  # Log file name
    level=logging.ERROR,         # Log level for errors
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class WebScraper:
    """
    A web scraper that uses BeautifulSoup primarily and falls back to Selenium
    when restricted URLs prevent access via requests.

    Methods:
        fetch_content(url): Fetches and processes the content from the specified URL.
    """

    def __init__(self):
        """
        Initializes the Selenium WebDriver in headless mode.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--user-agent=Mozilla/5.0")

        self.driver = webdriver.Chrome(service=ChromeService(), options=chrome_options)
        
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

        """
        # First attempt: Use requests + BeautifulSoup
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
            logging.error(f"Requests failed via BS4 for {url}: {e}")


        try:
            self.driver.get(url)

            # Wait for the page to load completely
            self.driver.implicitly_wait(10)

            # Get the page source and parse it with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

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
            logging.error(f"Selenium failed for {url}: {e}")
            return None

    def close(self):
        """
        Closes the Selenium WebDriver.
        """
        self.driver.quit()


# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    url = "https://www.chelseafertilitynyc.com/"  # Replace with the URL you want to scrape

    result = scraper.fetch_content(url)
    if result:
        print(f"Content from {result['url']}:\n{result['content'][:500]}...")  # Print the first 500 characters
    else:
        print("Failed to fetch content. Check the log file for details.")

    scraper.close()
