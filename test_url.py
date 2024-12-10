from urllib.parse import unquote, urlparse, urlunparse

def are_urls_equal(url1, url2):
    def normalize_url(url):
        parsed_url = urlparse(url)
        normalized_path = unquote(parsed_url.path)
        normalized_query = unquote(parsed_url.query)
        return urlunparse(
            (parsed_url.scheme, parsed_url.netloc, normalized_path, parsed_url.params, normalized_query, parsed_url.fragment)
        )

    print(normalize_url(url2))
    return normalize_url(url1) == normalize_url(url2)
    
def in_urls(url, urls):
    for u in urls:
        if are_urls_equal(url, u):
            return True
    return False

urls = ['https://weillcornell.org/news/weill-cornell-medicine%E2%80%99s-center-for-reproductive-medicine-has-been-ranked-by-newsweek-as-the-1', 'https://weillcornell.org/news/weill-cornell-medicine%E2%80%99s-center-for-reproductive-medicine-has-been-ranked-by-newsweek-as-the-1', 'https://weillcornell.org/news/newsweek-ranks-center-for-reproductive-medicine-nation%E2%80%99s-1-fertility-clinic']
url = "https://weillcornell.org/news/newsweek-ranks-center-for-reproductive-medicine-nation’s-1-fertility-clinic"

print(in_urls(url, urls)) # True