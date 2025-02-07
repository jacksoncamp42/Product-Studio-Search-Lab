# Introduction

In the evolving world of digital search, the rise of large language models (LLMs) is creating a shift in how users interact
with online content. Traditional search engines like Google and Bing rely on keyword-based algorithms to rank and deliver
results, and to rank higher website owners use Search Engine Optimization (SEO) tactics. However, LLM-driven search
offers conversational and contextual responses, redefining how information is accessed. For businesses, this means new
challenges in maintaining visibility, as LLMs prioritize semantic relevance over conventional SEO methods.

While the SEO industry, valued at $68 billion in 2022, is built on optimizing for traditional search engines, the field of
AI Optimization (AIO) for LLM-driven search is still in its early stages. Current solutions diagnose how LLMs perceive a
website but fall short of providing actionable fixes or strategies. Moreover, businesses lack tools to test real-time results of
LLM optimizations, making it difficult to validate their effectiveness.

This unmet need presents a critical gap in the market and a clear opportunity for businesses to rethink their approach
to visibility and optimization. Companies require a comprehensive solution that not only diagnoses their LLM visibility but
also provides concrete recommendations and a way to test these optimizations effectively. With this in mind, we developed
SearchLab—an end-to-end platform designed to meet the unique challenges of LLM-driven search.

# Write-Up

Please see 'How Might We Rethink SEO in the Context of LLM Driven Search?.pdf'.

# Set Up

Install requirements.txts:
```
pip install -r requirements.txt
```

Create a .env file:
```
OPENAI_API_KEY=
GOOGLE_API_KEY=
GOOGLE_CSE_ID=
```

To run an experiment:
```
python run_experiment.py --url="" --prompt_injection --content_func=fluent_optimization_gpt
```
