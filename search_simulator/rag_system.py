# rag_system.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

class RAGSystem:
    """
    A system for implementing Retrieval-Augmented Generation (RAG) using FAISS and OpenAI embeddings.

    Attributes:
        embedding_model (str): The embedding model to use for text vectorization (default is 'text-embedding-ada-002').
        api_key (str): The API key for accessing the embedding model.
        embeddings: The initialized embedding model instance.
        vector_store: The vector store to hold and retrieve documents for RAG (initially None).
    """
    def __init__(self, embedding_model='text-embedding-ada-002', api_key=None):
        """
        Initializes the RAGSystem with an embedding model and API key.

        Args:
            embedding_model (str, optional): The embedding model to use for text vectorization 
                                             (default is 'text-embedding-ada-002').
            api_key (str, optional): The API key for accessing the embedding model (default is None).
        """
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model, openai_api_key=self.api_key
        )
        self.vector_store = None  # Will initialize when adding documents


    def add_documents(self, texts_with_urls):
        """
        Adds documents to the vector store for retrieval.

        Args:
            texts_with_urls (list): A list of dictionaries, each containing:
                                    - 'content' (str): The document's text content.
                                    - 'url' (str): The URL associated with the document.

        Returns:
            None: Initializes the vector store or updates it with new documents.
        """
        if not texts_with_urls:
            return

        documents = []
        for item in texts_with_urls:
            text = item['content']
            url = item['url']
            doc = Document(page_content=text, metadata={'url': url})
            documents.append(doc)

        if self.vector_store is None:
            # Create the vector store from scratch using FAISS
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add new documents to the existing vector store
            try:
                self.vector_store.add_documents(documents)
            except:
                for document in documents:
                    try:
                        self.vector_store.add_documents([document])
                    except: 
                        pass


    def retrieve_relevant_documents(self, query, top_k=5):
        """
        Retrieves the top-k relevant documents for a given query.

        Args:
            query (str): The user's input query.
            top_k (int, optional): The number of relevant documents to retrieve (default is 5).

        Returns:
            list: A list of dictionaries, each containing:
                  - 'content' (str): The document's text content.
                  - 'url' (str): The URL associated with the document.
                  Returns an empty list if no vector store is initialized.
        """
        if self.vector_store is None:
            return []
        # Use similarity_search which handles embedding internally
        docs = self.vector_store.similarity_search_with_score(query, k=top_k)
        # Return documents with content and URL
        relevant_docs = [
            {'content': doc.page_content, 'url': doc.metadata.get('url', '')}
            for doc, _ in docs
        ]

        doc_scores = [(doc.metadata.get('url', ''),  score) for doc,score in docs]
        return relevant_docs, doc_scores
    
    def embed(self, text):
        return self.embeddings.embed_query(text)
