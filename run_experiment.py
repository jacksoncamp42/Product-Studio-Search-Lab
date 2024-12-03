import argparse
import numpy as np
import re
from transformers import pipeline
from RAGSystem import embed
from content_optimization import optimize_text, url_to_text

def run_experiment(url, func_name, prompt_injection=None):
    # Fetch the website text
    text = url_to_text(url)

    # Apply the specified optimization function if provided
    if func_name:
        text = optimize_text(func_name, text)

    # Apply prompt injection if provided
    if prompt_injection:
        text = f"{prompt_injection}\n\n{text}"

    return text

def evaluate(query, url, response):
    # Similarity Score
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    similarity_score = cosine_similarity(embed(query), embed(response))
    print("Similarity Score [0,1]:", similarity_score)

    # Website Score
    def extract_urls(response):
        return re.findall(r'(https?://\S+)', response)
    
    urls = extract_urls(response)
    website_score = 1 if url in urls else 0
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
        "--content_func",
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
    query = input("Enter the query for evaluation: ")
    url = input("Enter the website URL to optimize: ")

    text = run_experiment(url, args.func_name, args.prompt_injection)
    response = # TODO: @Sid, inject the text into the llm response and call llm engine
    score = evaluate(url, query, response)
    print("Score [0,1]:", score)