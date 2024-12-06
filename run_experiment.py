import argparse
import numpy as np
import re
from search_simulator.rag_system import RAGSystem
from content_optimization.content_optimization import optimize_text, url_to_text
from content_optimization.evaluate_seo import get_seo_score
from search_simulator.search_simulator import SearchSimulator 
from urllib.parse import unquote, urlparse, urlunparse
from prompt_injection.prompt_injection import prompt_injection
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

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
                
        # URL not found in results
        return 0.0

    position_score = evaluate_position(response, url)

    
    # Similarity Score
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    searchSimulator = SearchSimulator()
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
    
    return position_score, similarity_score, website_score, sentiment_score

def main(input_csv, output_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Prepare SearchSimulator
    search_simulator = SearchSimulator()

    results = []

    for row_i, row in df.iterrows():
        print("Processing row ", row_i, "/", len(df))
        query = row['Query']
        url = row['URL'] 
        title = row['Use Case']

        # call main_helper
        text = url_to_text(url)

        # Get LLM search response and evaluate
        _, _, response = search_simulator.generate_search_result(query, url, text)
        position_score, similarity_score, website_score, sentiment_score = evaluate(query, url, response)
        seo_score = get_seo_score(text)
        final_score = (seo_score + position_score + similarity_score + website_score + sentiment_score) / 5

        if args.func_name or args.prompt_injection:
            # apply experiment/website fix
            if args.func_name:
                optimized_text = optimize_text(args.func_name, text)
            elif args.prompt_injection:
                best_prompt = prompt_injection(query, url, title)
                optimized_text = text + "\n" + best_prompt
            else:
                optimized_text = text

            _, _, response_after = search_simulator.generate_search_result(query, url, optimized_text)
            position_score_after, similarity_score_after, website_score_after, sentiment_score_after = evaluate(query, url, response_after)
            seo_score_after = get_seo_score(optimized_text)
            final_score_after = (seo_score_after + position_score_after + similarity_score_after + website_score_after + sentiment_score_after) / 5
        else:
            position_score_after, similarity_score_after, website_score_after, sentiment_score_after, seo_score_after, final_score_after = 0, 0, 0, 0, 0, 0
        
        # write the results
        results.append({
        "Use Case": row['Use Case'],
        "Query": query,
        "URL": url,
        "SEO Score": seo_score,
        "Position Score": position_score,
        "Similarity Score": similarity_score,
        "Website Score": website_score,
        "Sentiment Score": sentiment_score,
        "Final Score": final_score,
        "SEO Score After": seo_score_after,
        "Position Score After": position_score_after,
        "Similarity Score After": similarity_score_after,
        "Website Score After": website_score_after,
        "Sentiment Score After": sentiment_score_after,
        "Final Score After": final_score_after
        })

    # Save results to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Lab")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
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

    main(args.input_csv, args.output_csv)