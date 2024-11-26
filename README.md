# Product-Studio-Search-Lab

Install requirements.txts:

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
