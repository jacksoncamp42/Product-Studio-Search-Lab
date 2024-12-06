# -*- coding: utf-8 -*-
"""Evaluate SEO.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1o8_9AxWEjodS8qgrfttkWty5o4j67iOm

## Evaluate SEO
"""

import yake

def extract_keywords(text, max_keywords=10):
    kw_extractor = yake.KeywordExtractor(top=max_keywords, stopwords=None)
    keywords = kw_extractor.extract_keywords(text)
    keywords = [kw for kw, score in keywords]
    focus_keyword = keywords[0]  # Use the most significant keyword as the focus
    related_keywords = keywords[1:6]  # Use the next significant keywords as related
    return focus_keyword, related_keywords

class SeoAnalyzer:
    MINIMUM_KEYWORD_DENSITY = 0.46
    MAXIMUM_KEYWORD_DENSITY = 1.1
    MAXIMUM_SUB_KEYWORD_DENSITY = 0.9
    MINIMUM_SUB_KEYWORD_DENSITY = 0.12

    def __init__(self, optimized_text: str, focus_keyword: str, related_keywords: list):
        self.optimized_text = optimized_text
        self.focus_keyword = focus_keyword
        self.related_keywords = related_keywords
        self.keyword_density = self.calculate_density(self.focus_keyword, self.optimized_text)
        self.sub_keywords_density = self.calculate_sub_keywords_density()

    def calculate_density(self, keyword: str, text: str) -> float:
        if not keyword or not text:
            return 0.0
        keyword_count = self.count_occurrences_in_string(keyword, text)
        word_count = len(text.split())
        return (keyword_count / word_count) * 100 if word_count > 0 else 0.0

    def count_occurrences_in_string(self, keyword: str, text: str) -> int:
        if not keyword or not text:
            return 0
        occurrences = text.lower().split(keyword.lower())
        return max(len(occurrences) - 1, 0)

    def calculate_sub_keywords_density(self):
        densities = []
        for sub_keyword in self.related_keywords:
            density = self.calculate_density(sub_keyword, self.optimized_text)
            densities.append({
                'keyword': sub_keyword,
                'density': density
            })
        return densities

    def get_seo_score(self) -> float:
        MAX_SCORE = 100
        good_points = 1
        warnings = 1 if self.keyword_density < self.MINIMUM_KEYWORD_DENSITY or self.keyword_density > self.MAXIMUM_KEYWORD_DENSITY else 0
        return min(((good_points / (warnings + good_points)) * 100), MAX_SCORE)

    def get_keyword_seo_score(self) -> float:
        MAX_SCORE = 100
        keyword_density_score = self.keyword_density * 10
        sub_keywords_density_score = sum(max(sub_keyword['density'] * 10, 0) for sub_keyword in self.sub_keywords_density)
        total_score = min(keyword_density_score + sub_keywords_density_score, MAX_SCORE)
        return total_score

    def get_keyword_density(self) -> float:
        return self.keyword_density

    def get_sub_keywords_density(self) -> list:
        return self.sub_keywords_density

    def get_keyword_frequency(self) -> int:
        return self.count_occurrences_in_string(self.focus_keyword, self.optimized_text)

def get_seo_score(text):
    focus_keyword, related_keywords = extract_keywords(text)
    analyzer = SeoAnalyzer(text, focus_keyword, related_keywords)

    # # Output results
    # print(f"SEO Score: {analyzer.get_seo_score()}")
    # print(f"Keyword SEO Score: {analyzer.get_keyword_seo_score()}")
    # print(f"Keyword Density: {analyzer.get_keyword_density()}")
    # print("Sub Keyword Density:", ", ".join(
    #     f"({sub['keyword']} {sub['density']})" for sub in analyzer.get_sub_keywords_density()
    # ))
    # print(f"Keyword Frequency: {analyzer.get_keyword_frequency()}")
    return float(analyzer.get_seo_score()) / float(100)