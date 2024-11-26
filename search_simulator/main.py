# main.py

from llm_engine import LLMEngine
from llm_prompt import LLMPrompt
from rag_system import RAGSystem
from google_api import GoogleAPI
from dotenv import load_dotenv, find_dotenv
from web_scraper import WebScraper
import os

load_dotenv(find_dotenv())


def main():
    # Initialize API keys and IDs
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    # print(OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID)
    
    # Initialize components
    llm_engine = LLMEngine(api_key=OPENAI_API_KEY)
    llm_prompt = LLMPrompt(llm_engine)
    rag_system = RAGSystem(api_key=OPENAI_API_KEY)
    google_api = GoogleAPI(api_key=GOOGLE_API_KEY, cse_id=GOOGLE_CSE_ID)
    web_scraper = WebScraper()

    # Conversation loop
    while True:
        # Step 1: Get user input
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Conversation ended.")
            break

        # Step 2: Generate sub-queries
        sub_queries = llm_prompt.generate_subqueries(user_query)
        if not sub_queries:
            print("No sub-queries were generated.")
            continue

        # Step 3: Fetch search results and content
        all_contents = []
        for sub_query in sub_queries:
            search_results = google_api.search(sub_query, num_results=25)
            for result in search_results:
                url = result.get('link')
                fetched = web_scraper.fetch_content(url)
                if fetched:
                    all_contents.append(fetched)
        if not all_contents:
            print("No content was fetched from the search results.")
            continue

        # Step 4: Add documents to RAG system
        rag_system.add_documents(all_contents)

        # Step 5: Retrieve relevant documents
        relevant_documents = rag_system.retrieve_relevant_documents(user_query)
        if not relevant_documents:
            print("No relevant documents were found in the RAG system.")
            continue

        # Step 6: Generate final answer
        final_answer = llm_prompt.generate_response(user_query, relevant_documents)

        print("\nAssistant:")
        print(final_answer)

if __name__ == "__main__":
    main()