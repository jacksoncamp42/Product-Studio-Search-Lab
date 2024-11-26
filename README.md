# Product-Studio-Search-Lab

Install requirements.txts:
```
pip install -r requirements.txt
```

To run an experiment:
```
python run_experiment.py --url="" --prompt_injection --content_func=fluent_optimization_gpt
```

## Relevant Functions
evaluate_seo.py:

```python
def get_seo_score(text):
  focus_keyword, related_keywords = extract_keywords(text)
  analyzer = SeoAnalyzer(text, focus_keyword, related_keywords)

  # Output results
  print(f"SEO Score: {analyzer.get_seo_score()}")
  print(f"Keyword SEO Score: {analyzer.get_keyword_seo_score()}")
  print(f"Keyword Density: {analyzer.get_keyword_density()}")
  print("Sub Keyword Density:", ", ".join(
      f"({sub['keyword']} {sub['density']})" for sub in analyzer.get_sub_keywords_density()
  ))
  print(f"Keyword Frequency: {analyzer.get_keyword_frequency()}")
```

content_optimization.py:
```python
def url_to_text(url: str) ->  str:

def optimize_text(func_name, text):
    """
    Optimize text using the specified optimization function.

    Parameters:
    - func_name (str): Name of the optimization function to use.
    - text (str): The text to optimize.

    Available functions:
    - fluent_optimization_gpt
    - unique_words_optimization_gpt
    - authoritative_optimization_mine
    - more_quotes_mine
    - citing_credible_sources_mine
    - simple_language_mine
    - technical_terms_mine
    - seo_optimize_mine2
    """
    func = globals().get(func_name)
    if func is None:
        raise ValueError(f"Function '{func_name}' not found.")
    return func(text)
```

prompt_injection.py:

```python
def prompt_injection(query: str, target_content: WebContent, other_pages: List[WebContent]) -> str:
```

```python
class WebContent:
    url: str
    title: str
    content: str
```
