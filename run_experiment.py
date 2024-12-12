import argparse
import os
import re
import time
from urllib.parse import unquote, urlparse, urlunparse

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from content_optimization.content_optimization import optimize_text, url_to_text
from content_optimization.evaluate_seo import get_seo_score
from prompt_injection.prompt_injection import prompt_injection
from search_simulator.search_simulator import SearchSimulator


# copied over from prompt_injection.py
def get_citation_prompt() -> str:
    return """
    Additional Instructions:

    When mentioning any fact or referencing information from a website, ensure you explicitly cite the source using the format [source: URL]. This includes any statistics, claims, or referenced material. If no reliable information is available, state that clearly without fabricating details. If referencing a website make sure it’s in this format [source: URL] and also make sure that it is a link and no titles. 
    What I mean by this is that any time you reference any website (document) provided to you, at the end of the reference make sure to cite the website similar to how it is done in research papers. What this means is that you should have a list of citations at the bottom of your response as an ordered list. Finally, in your response, any time you reference information from one of these documents, you should add the url and title of the document in brackets next to the reference.

    For example: a response would be like:

    Search Results:
    --------------------------------------------------
    The best center for reproductive medicine in New York City, according to Newsweek's America’s Best Fertility Clinics 2023 survey, is the [Center for Reproductive Medicine](https://weillcornell.org/news/newsweek-ranks-center-for-reproductive-medicine-nation’s-1-fertility-clinic) affiliated with Weill Cornell Medicine. This center has been recognized as the nation's top fertility clinic and has a long-standing history of success, with over 31,500 babies born through in vitro fertilization (IVF) since 1988. Led by Dr. Zev Rosenwaks, the center is known for its comprehensive care, advanced fertility tests, and procedures. It specializes in assisting patients with complex medical histories in achieving parenthood, even after unsuccessful attempts elsewhere. The dedication of the center to helping aspiring parents fulfill their dreams, coupled with its high success rates, positions it as a leading choice for reproductive medicine in New York City.

    While the Center for Reproductive Medicine is ranked at the top, another renowned establishment in New York City providing exceptional fertility care is the [NYU Langone Fertility Center](https://www.fertilityny.org/). Although it may not have the specific top ranking of the Center for Reproductive Medicine, the NYU Langone Fertility Center is known for providing quality services in the field of reproductive medicine.

    For individuals seeking more information about top reproductive endocrinologists and fertility centers in New York City, Castle Connolly provides a comprehensive list of top doctors and centers of excellence in reproductive endocrinology and infertility in the area.

    In summary, while the Center for Reproductive Medicine at Weill Cornell Medicine stands out as the best center for reproductive medicine in New York City based on recent rankings, there are multiple reputable options available in the city, including the NYU Langone Fertility Center.

    Sources:
    1. [Newsweek Ranks Center for Reproductive Medicine Nation’s #1 Fertility Clinic](https://weillcornell.org/news/newsweek-ranks-center-for-reproductive-medicine-nation’s-1-fertility-clinic)
    2. [NYU Langone Fertility Center | Fertility Care for NYC Families](https://www.fertilityny.org/)
    3. [Top Reproductive Endocrinologists near New York, NY - Castle Connolly](https://www.castleconnolly.com/specialty/reproductive-endocrinology-infertility/new-york-ny)

    """

def evaluate_position(response, url):
    def normalize_url(url):
        parsed_url = urlparse(url)
        normalized_path = unquote(parsed_url.path)
        normalized_query = unquote(parsed_url.query)
        return urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                normalized_path,
                parsed_url.params,
                normalized_query,
                parsed_url.fragment,
            )
        )

    normalized_target = normalize_url(url)
    results = [r for r in response.split("\n\n") if r.strip()]

    for i, result in enumerate(results, 1):
        result_urls = re.findall(r"(https?://[^\s\)]+)", result)
        normalized_result_urls = [normalize_url(u) for u in result_urls]

        if normalized_target in normalized_result_urls:
            position_score = 1.0 if i == 1 else 1.0 / i
            return position_score

    return 0.0


def evaluate_similarity(query, response):
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    searchSimulator = SearchSimulator()
    rag = searchSimulator.rag_system
    similarity_score = cosine_similarity(rag.embed(query), rag.embed(response))
    return similarity_score


def evaluate_website(url, response):
    def extract_urls(response):
        return re.findall(r"(https?://[^\s\)]+)", response)

    def are_urls_equal(url1, url2):
        def normalize_url(url):
            parsed_url = urlparse(url)
            normalized_path = unquote(parsed_url.path)
            normalized_query = unquote(parsed_url.query)
            return urlunparse(
                (
                    parsed_url.scheme,
                    parsed_url.netloc,
                    normalized_path,
                    parsed_url.params,
                    normalized_query,
                    parsed_url.fragment,
                )
            )

        return normalize_url(url1) == normalize_url(url2)

    def in_urls(url, urls):
        for u in urls:
            if are_urls_equal(url, u):
                return True
        return False

    urls = extract_urls(response)
    return 1 if in_urls(url, urls) else 0


def evaluate_sentiment(url, response):
    def analyze_sentiment(text):
        sentiment = SentimentIntensityAnalyzer()
        return sentiment.polarity_scores(text)["pos"]

    def get_relevant_text(text, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        org_name = domain.split(".")[-2]

        paragraphs = text.split("\n\n")
        relevant_paragraphs = []
        for paragraph in paragraphs:
            if (
                url in paragraph
                or domain in paragraph
                or org_name.lower() in paragraph.lower()
            ):
                relevant_paragraphs.append(paragraph)

        return " ".join(relevant_paragraphs) if relevant_paragraphs else text

    relevant_text = get_relevant_text(response, url)
    sentiment_score = analyze_sentiment(relevant_text)
    return sentiment_score

def evaluate(query, url, response):
    position_score = evaluate_position(response, url)
    similarity_score = evaluate_similarity(query, response)
    website_score = evaluate_website(url, response)
    sentiment_score = evaluate_sentiment(url, response)
    seo_score = get_seo_score(response)

    final_score = (
        seo_score + position_score + similarity_score + website_score + sentiment_score
    ) / 5

    return (
        position_score,
        similarity_score,
        website_score,
        sentiment_score,
        seo_score,
        final_score,
    )

def main(input_csv, output_csv, start_row, end_row, print_flag):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Filter rows based on start_row and end_row
    if start_row is not None or end_row is not None:
        df = df.iloc[start_row:end_row]
    elif start_row is not None:
        df = df.iloc[start_row:]

    # Prepare SearchSimulator
    search_simulator = SearchSimulator(llm_generation_instructions=get_citation_prompt())

    results = []

    for row_i, row in df.iterrows():
        try:
            start_time = time.time()
            print(f"Processing row {row_i}")

            query = row["Query"]
            url = row["URL"]
            title = row["Use Case"]

            # Extract text and evaluate
            text = url_to_text(url)
            _, _, response = search_simulator.generate_search_result(query, url, text)
            (
                position_score,
                similarity_score,
                website_score,
                sentiment_score,
                seo_score,
                final_score,
            ) = evaluate(query, url, response)

            # Print response if flag is set
            if print_flag:
                print(f"Response before optimization:\n{response}")

            # Apply optimization if specified
            if args.func_name or args.prompt_injection:
                if args.func_name:
                    optimized_text = optimize_text(args.func_name, text)
                elif args.prompt_injection:
                    best_prompt = prompt_injection(
                        query,
                        url,
                        title,
                        depth=args.depth,
                        branching_factor=args.branching_factor,
                        num_runs=args.num_runs,
                    )
                    optimized_text = text + "\n" + best_prompt
                else:
                    optimized_text = text

                _, _, response_after = search_simulator.generate_search_result(
                    query, url, optimized_text
                )
                (
                    position_score_after,
                    similarity_score_after,
                    website_score_after,
                    sentiment_score_after,
                    seo_score_after,
                    final_score_after,
                ) = evaluate(query, url, response_after)

                # Print optimized response if flag is set
                if print_flag:
                    print(f"Response after optimization:\n{response_after}")
            else:
                (
                    position_score_after,
                    similarity_score_after,
                    website_score_after,
                    sentiment_score_after,
                    seo_score_after,
                    final_score_after,
                ) = (0, 0, 0, 0, 0, 0)

            # Add to results
            results.append(
                {
                    "Use Case": row["Use Case"],
                    "Query": query,
                    "URL": url,
                    "SEO Score": seo_score,
                    "SEO Score After": seo_score_after,
                    "Position Score": position_score,
                    "Position Score After": position_score_after,
                    "Similarity Score": similarity_score,
                    "Similarity Score After": similarity_score_after,
                    "Website Score": website_score,
                    "Website Score After": website_score_after,
                    "Sentiment Score": sentiment_score,
                    "Sentiment Score After": sentiment_score_after,
                    "Final Score": final_score,
                    "Final Score After": final_score_after,
                }
            )

            # Save responses to files
            if args.func_name == "fluent_optimization_gpt":
                folder_name = "content_optimization_fluent_results"
            elif args.func_name == "authoritative_optimization_mine":
                folder_name = "content_optimization_authoritative_results"
            elif args.func_name == "prompt_injection":
                folder_name = "prompt_injection_results"
            else:
                folder_name = "baseline_results"
            os.makedirs(folder_name, exist_ok=True)

            # Write scores before optimization
            with open(
                f"{folder_name}/row_{row_i}_before.txt", "w", encoding="utf-8"
            ) as f:
                f.write(f"Use Case: {row['Use Case']}\n")
                f.write(f"Query: {query}\n")
                f.write(f"URL: {url}\n\n")
                f.write(f"Text: {text}\n\n")
                f.write(f"Response: {response}\n\n")
                f.write(f"Position Score: {position_score}\n")
                f.write(f"Similarity Score: {similarity_score}\n")
                f.write(f"Website Score: {website_score}\n")
                f.write(f"Sentiment Score: {sentiment_score}\n")
                f.write(f"SEO Score: {seo_score}\n")
                f.write(f"Final Score: {final_score}\n")

            # Write scores after optimization if optimization was applied
            if args.func_name or args.prompt_injection:
                with open(
                    f"{folder_name}/row_{row_i}_after.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(f"Use Case: {row['Use Case']}\n")
                    f.write(f"Query: {query}\n")
                    f.write(f"URL: {url}\n\n")
                    f.write(f"Text: {optimized_text}\n\n")
                    f.write(f"Response_After: {response_after}\n\n")
                    f.write(f"Position Score_After: {position_score_after}\n")
                    f.write(f"Similarity Score_After: {similarity_score_after}\n")
                    f.write(f"Website Score_After: {website_score_after}\n")
                    f.write(f"Sentiment Score_After: {sentiment_score_after}\n")
                    f.write(f"SEO Score_After: {seo_score_after}\n")
                    f.write(f"Final Score_After: {final_score_after}\n")

            # Print output
            elapsed_time = time.time() - start_time
            print(
                f"Processed row {row_i} - Final Score: {final_score}, Final Score After: {final_score_after} - Time: {elapsed_time:.2f} seconds"
            )

        except Exception as e:
            print(f"Error processing row {row_i}: {e}")

    # Save results to CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Lab")
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Path to the output CSV file"
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=None,
        help="Row to start processing from (0-indexed)",
    )
    parser.add_argument(
        "--end_row",
        type=int,
        default=None,
        help="Row to stop processing (0-indexed, exclusive)",
    )
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
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Depth parameter for tree of attacks (default: 2)",
    )
    parser.add_argument(
        "--branching_factor",
        type=int,
        default=2,
        help="Branching factor for tree of attacks (default: 2)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=2,
        help="Number of runs for each evaluation (default: 2)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Flag to print the response before and after optimization, along with the final scores.",
    )
    args = parser.parse_args()

    main(args.input_csv, args.output_csv, args.start_row, args.end_row, args.print)
