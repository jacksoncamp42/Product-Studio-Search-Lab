# Write-Up

Please see 'How Might We Rethink SEO in the Context of LLM Driven Search?.pdf'.

# Product-Studio-Search-Lab

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
