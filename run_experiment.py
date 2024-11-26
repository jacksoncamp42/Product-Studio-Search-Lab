import argparse
from content_optimization import optimize_text, url_to_text

def run_experiment(url, func_name, prompt_injection=None):
    """
    Run the main experiment for text optimization on a website.

    Parameters:
    - url (str): The website URL to optimize.
    - func_name (str): The name of the optimization function to apply.
    - prompt_injection (str, optional): Additional prompt injection text to include.

    Returns:
    - str: The optimized text.
    """
    # Fetch the website text
    text = url_to_text(url)

    # Apply the specified optimization function if provided
    if func_name:
        text = optimize_text(func_name, text)

    # Apply prompt injection if provided
    if prompt_injection:
        text = f"{prompt_injection}\n\n{text}"

    return text

def optimize():
    """
    Main function to parse arguments and run the text optimization experiment.
    """
    parser = argparse.ArgumentParser(description="Search Lab")
    parser.add_argument("--url", type=str, help="The website URL to optimize.")
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

    # Run the experiment
    optimized_text = run_experiment(
        url=args.url, func_name=args.func_name, prompt_injection=args.prompt_injection
    )

    # Output the optimized text
    return optimized_text

def evaluate(text):
    # TODO: write the metric functions in llm simulator class, then import and call here
    pass

if __name__ == "__main__":
    text = optimize()
    score = evaluate(text)

    # TODO: print out the score OR change to iterate through a whole csv of urls and write score back out
