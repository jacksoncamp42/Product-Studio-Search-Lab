import streamlit as st

# Assume the following functions are available from your previously defined code:
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

import re
import numpy as np
from urllib.parse import unquote, urlparse, urlunparse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Helper functions
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


def run_initial_evaluation(query, url):
    text = url_to_text(url)
    search_simulator = SearchSimulator()

    _, _, response = search_simulator.generate_search_result(query, url, text)
    position_score, similarity_score, website_score, sentiment_score = evaluate(query, url, response)
    seo_score = get_seo_score(text)
    final_score = (seo_score + position_score + similarity_score + website_score + sentiment_score) / 5

    data = {
        "query": query,
        "url": url,
        "original_text": text,
        "original_response": response,
        "original_scores": {
            "seo_score": seo_score,
            "position_score": position_score,
            "similarity_score": similarity_score,
            "website_score": website_score,
            "sentiment_score": sentiment_score,
            "final_score": final_score
        }
    }
    return data

def run_optimization(data, method=None, injection=False):
    query = data["query"]
    url = data["url"]
    text = data["original_text"]

    if injection:
        # Assume title extraction is optional; if needed, you can parse from HTML or skip.
        title = "N/A"
        best_prompt = prompt_injection(query, url, title)
        optimized_text = text + "\n" + best_prompt
    elif method:
        optimized_text = optimize_text(method, text)
    else:
        optimized_text = text

    simulator = SearchSimulator()
    _, _, response = simulator.generate_search_result(query, url, optimized_text)

    position_score, similarity_score, website_score, sentiment_score = evaluate(query, url, response)
    seo_score = get_seo_score(optimized_text)
    final_score = (seo_score + position_score + similarity_score + website_score + sentiment_score) / 5

    data["optimized_text"] = optimized_text
    data["optimized_response"] = response
    data["optimized_scores"] = {
        "seo_score": seo_score,
        "position_score": position_score,
        "similarity_score": similarity_score,
        "website_score": website_score,
        "sentiment_score": sentiment_score,
        "final_score": final_score
    }

    return data

# Streamlit UI
st.set_page_config(page_title="GEO Demo", layout="wide")

if 'experiment_data' not in st.session_state:
    st.session_state.experiment_data = {}

if 'page' not in st.session_state:
    st.session_state.page = 0

if st.session_state.page == 0:
    st.title("Generative Engine Optimization (GEO) Demo")

    query = st.text_input("Enter your Query:", value="What is the best shoe for running?")
    url = st.text_input("Enter the Website URL:", value="https://www.example.com")

    content_optimization_methods = {
        "Make text more fluent": "fluent_optimization_gpt",
        "Add more unique keywords": "unique_words_optimization_gpt",
        "Increase authoritative tone": "authoritative_optimization_mine",
        "Add more quotes": "more_quotes_mine",
        "Cite credible sources": "citing_credible_sources_mine",
        "Simplify language": "simple_language_mine",
        "Add technical terms": "technical_terms_mine",
        "Add new SEO keywords": "seo_optimize_mine2"
    }

    # First UI step where user chooses optimization type
    optimization_choice = st.selectbox(
        "Do you want to use prompt injection, content optimization, or neither?",
        ["None", "Prompt Injection", "Content Optimization"]
    )

    # If they choose content optimization, show the list of human-readable options
    chosen_method = None
    use_prompt_injection = False

    if optimization_choice == "Prompt Injection":
        use_prompt_injection = True
    elif optimization_choice == "Content Optimization":
        human_readable_method = st.selectbox(
            "Select a content optimization method:",
            list(content_optimization_methods.keys())
        )
        chosen_method = content_optimization_methods[human_readable_method]
    elif optimization_choice == "None":
        # Neither prompt injection nor content optimization
        pass

    if st.button("Confirm Website"):
        st.session_state.experiment_data = run_initial_evaluation(query, url)
        st.session_state.experiment_data["use_prompt_injection"] = use_prompt_injection
        st.session_state.experiment_data["chosen_method"] = chosen_method
        st.session_state.page = 1
        st.rerun()

elif st.session_state.page == 1:
    st.title("Original Website and Results")

    data = st.session_state.experiment_data
    original_text = data["original_text"]
    original_scores = data["original_scores"]
    original_response = data["original_response"]

    st.subheader("Original Website Content")
    st.text_area("Website Content:", original_text, height=200)

    st.subheader("Original Scores")
    st.write(original_scores)

    st.subheader("Search Engine Result (Before Optimization)")
    st.text(original_response)

    chosen_method = data["chosen_method"]
    run_opt = st.button("Run Optimization")

    if run_opt:
        with st.spinner("Running optimization..."):
            data = run_optimization(
                data, 
                method=data["chosen_method"], 
                injection=data["use_prompt_injection"]
            )
            st.session_state.experiment_data = data
        st.success("Optimization completed!")
        st.session_state.page = 2
        st.rerun()

elif st.session_state.page == 2:
    st.title("Optimized Website and Results")

    data = st.session_state.experiment_data
    optimized_text = data.get("optimized_text", data["original_text"])
    optimized_scores = data.get("optimized_scores", data["original_scores"])
    optimized_response = data.get("optimized_response", data["original_response"])

    st.subheader("Optimized Website Content")
    st.text_area("Optimized Content:", optimized_text, height=200)

    st.subheader("Optimized Scores")
    st.write(optimized_scores)

    st.subheader("Search Engine Result (After Optimization)")
    st.text(optimized_response)

    view_mode = st.radio("View Mode:", ["Original", "Optimized"])
    if view_mode == "Original":
        st.write("**Original Content**")
        st.text_area("Content", data["original_text"], height=200)
        st.write("**Original Scores**")
        st.write(data["original_scores"])
        st.write("**Original Response**")
        st.text(data["original_response"])
    else:
        st.write("**Optimized Content**")
        st.text_area("Content", optimized_text, height=200)
        st.write("**Optimized Scores**")
        st.write(optimized_scores)
        st.write("**Optimized Response**")
        st.text(optimized_response)

    st.subheader("Navigation")
    if st.button("Go Back"):
        st.session_state.page = 1
        st.rerun()
