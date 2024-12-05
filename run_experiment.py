import argparse
from turtle import position
import numpy as np
import re
from search_simulator.rag_system import RAGSystem
from content_optimization.content_optimization import optimize_text, url_to_text
from search_simulator.search_simulator import SearchSimulator 
from urllib.parse import unquote, urlparse, urlunparse
from prompt_injection.prompt_injection import prompt_injection
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def evaluate(query, url, response):
    # Ranking score
    def evaluate_position(response, url):
        def normalize_url(url):
            parsed_url = urlparse(url)
            normalized_path = unquote(parsed_url.path)
            normalized_query = unquote(parsed_url.query)
            return urlunparse(
                (parsed_url.scheme, parsed_url.netloc, normalized_path, parsed_url.params, normalized_query, parsed_url.fragment)
            )
        
        normalized_target = normalize_url(url)
        results = [r for r in response.split('\n\n') if r.strip()]
        
        for i, result in enumerate(results, 1):
            result_urls = re.findall(r'(https?://[^\s\)]+)', result)
            normalized_result_urls = [normalize_url(u) for u in result_urls]
            
            if normalized_target in normalized_result_urls:
                position_score = 1.0 if i == 1 else 1.0 / i
                # print(f"Position in results: {i}")
                # print(f"Position Score [0,1]: {position_score}")
                return position_score
                
        print("URL not found in results")
        return 0.0
    position_score = evaluate_position(response, url)

    
    # Similarity Score
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    rag = searchSimulator.rag_system
    similarity_score = cosine_similarity(rag.embed(query), rag.embed(response))

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
    website_score = 1 if in_urls(url, urls) else 0

    # Sentiment Score
    def analyze_sentiment(text):
        sentiment = SentimentIntensityAnalyzer()
        return sentiment.polarity_scores(text)['pos']
    
    sentiment_score = analyze_sentiment(response) if website_score else 0
    

    final_score = (position_score + similarity_score + website_score + sentiment_score) / 4

    return position_score, similarity_score, website_score, sentiment_score, final_score

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

    # TODO: eventually change to iterate over csv file and write score back to csv file
    query = "What is the best center for reproductive medicine in New York City?"
    url = "https://weillcornell.org/news/newsweek-ranks-center-for-reproductive-medicine-nation’s-1-fertility-clinic"
    title = "Weill Cornell Medicine's Center for Reproductive Medicine"

    text = url_to_text(url)

    # apply experiment/website fix
    if args.func_name:
        text = optimize_text(args.func_name, text)

    if args.prompt_injection:
        best_prompt = prompt_injection(query, url, title)
        text = text + "\n" + best_prompt
    
    # get LLM search response and evaluate
    searchSimulator = SearchSimulator()
    _, _, response = searchSimulator.generate_search_result(query, url, text)
    position_score, similarity_score, website_score, sentiment_score, final_score = evaluate(query, url, response)

    # print out results!
    print("Query: \n", query)
    print("-" * 50)
    print("Search Response: \n", response)
    print("-" * 50)
    print("Position Score [0,1]:", position_score)
    print("Similarity Score [0,1]:", similarity_score)
    print("Website Score [0,1]:", website_score)
    print("Sentiment Score [0,1]:", sentiment_score)
    print("-" * 50)
    print("Averaged Score [0,1]:", final_score)