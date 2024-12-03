import argparse
import numpy as np
import re
from transformers import pipeline
from search_simulator.rag_system import RAGSystem
from content_optimization.content_optimization import optimize_text, url_to_text
from search_simulator.search_simulator import SearchSimulator 
from urllib.parse import unquote, urlparse, urlunparse

def evaluate(query, url, response):
    # Similarity Score
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    rag = searchSimulator.rag_system
    similarity_score = cosine_similarity(rag.embed(query), rag.embed(response))
    print("Similarity Score [0,1]:", similarity_score)

    # Website Score
    def extract_urls(response):
        return re.findall(r'(https?://[^\s\)]+)', response)
    
    def are_urls_equal(url1, url2):
        def normalize_url(url):
            parsed_url = urlparse(url)
            normalized_path = unquote(parsed_url.path)
            normalized_query = unquote(parsed_url.query)
            return urlunparse(
                (parsed_url.scheme, parsed_url.netloc, normalized_path, parsed_url.params, normalized_query, parsed_url.fragment)
            )

        return normalize_url(url1) == normalize_url(url2)
    
    def in_urls(url, urls):
        for u in urls:
            if are_urls_equal(url, u):
                return True
        return False
    
    urls = extract_urls(response)
    print("Extracted URLs:", urls)
    website_score = 1 if in_urls(url, urls) else 0
    print("Website Score {0,1}:", website_score)

    # Sentiment Score
    def analyze_sentiment(text):
        sentiment_pipeline = pipeline("sentiment-analysis")
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        if label == "NEGATIVE":
            return 1 - score
        elif label == "POSITIVE":
            return score
        else:
            raise ValueError("Unexpected sentiment label received")
        
    sentiment_score = analyze_sentiment(response) if website_score else 0
    print("Sentiment Score [0,1]:", sentiment_score)

    return (similarity_score + website_score + sentiment_score) / 3

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Search Lab")
    parser.add_argument(
        "--func_name",
        type=str,
        default=None,
        help=(
            "The name of the optimization function to apply. Available functions include: "
            "fluent_optimization_gpt, unique_words_optimization_gpt, authoritative_optimization_mine, "
            "more_quotes_mine, citing_credible_sources_mine, simple_language_mine, "
            "technical_terms_mine, seo_optimize_mine2."
        ),
    )
    parser.add_argument(
        "--prompt_injection",
        type=str,
        default=None,
        help="Optional text to prepend to the website content for prompt injection.",
    )
    args = parser.parse_args()

    # accept query and URL from user input - TODO: eventually change to iterate over csv file and write score back to csv file
    # query = input("Enter the query for evaluation: ")
    # url = input("Enter the website URL to optimize: ")

    query = "What is the best center for reproductive medicine in New York City?"
    url = "https://weillcornell.org/news/newsweek-ranks-center-for-reproductive-medicine-nation’s-1-fertility-clinic"

    text = url_to_text(url)

    # apply experiment/website fix
    if args.func_name:
        text = optimize_text(args.func_name, text)

    # TODO: @Jackson add in prompt injection
    
    # get LLM search response and evaluate
    searchSimulator = SearchSimulator()
    _, _, response = searchSimulator.generate_search_result(query, url, text)
    print("Search Response:", response)
    score = evaluate(url, query, response)
    print("Score [0,1]:", score)