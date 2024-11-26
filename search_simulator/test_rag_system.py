# test_rag_system.py
from dotenv import load_dotenv, find_dotenv
import os


load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if __name__ == "__main__":
    from rag_system import RAGSystem

    # Initialize RAGSystem
    rag_system = RAGSystem(api_key=OPENAI_API_KEY)

    # Sample documents with URLs
    texts_with_urls = [
        {'content': "The Eiffel Tower is located in Paris.", 'url': 'https://example.com/eiffel-tower'},
        {'content': "The Great Wall of China is visible from space.", 'url': 'https://example.com/great-wall'},
        {'content': "The tallest mountain in the world is Mount Everest.", 'url': 'https://example.com/everest'},
    ]

    # Add documents
    rag_system.add_documents(texts_with_urls)

    # Query
    query = "Where is the Eiffel Tower located?"

    # Retrieve relevant documents
    relevant_docs = rag_system.retrieve_relevant_documents(query, top_k=1)
    print("Relevant Documents:")
    for doc in relevant_docs:
        print(f"Content: {doc['content']}")
        print(f"URL: {doc['url']}")
