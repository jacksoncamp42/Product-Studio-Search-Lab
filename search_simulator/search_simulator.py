
from .llm_engine import LLMEngine
from .llm_prompt import LLMPrompt
from .rag_system import RAGSystem
from .google_api import GoogleAPI
from dotenv import load_dotenv, find_dotenv
from .web_scraper import WebScraper
import os

load_dotenv(find_dotenv())


class SearchSimulator: 
    """_summary_
    """
    def __init__(self, llm_query_instructions=None, llm_generation_instructions=None):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
        self.llm_engine = LLMEngine(api_key=self.OPENAI_API_KEY)
        self.llm_prompt = LLMPrompt(self.llm_engine)
        self.rag_system = RAGSystem(api_key=self.OPENAI_API_KEY)
        self.google_api = GoogleAPI(api_key=self.GOOGLE_API_KEY, cse_id=self.GOOGLE_CSE_ID)
        self.web_scraper = WebScraper()
        self.llm_query_instructions = llm_query_instructions
        self.llm_generation_instructions = llm_generation_instructions
    
    def generate_search_result(self, user_query, website_to_optimize, website_content=None):
        """ 
        Runs the search simulator on a query, 
        Adds the optimized website to the rag system if it doesnt appear in the search results 
        Updates the content of the website if it does appear in the search results and website_content is not None
        Returns:
            sub_query_results: Dictionary with the the following form: 
                { sub_query: "The text of the subquery", results: [List of URLs] }
            embedding_similarity_results: List of key value pairs of the form: 
                [(website_url, query_website_embedding_similarity)]
            search_result: String containing the search simulators output. 
            
        """

        sub_queries = self.llm_prompt.generate_subqueries(user_query, instructions=self.llm_query_instructions)
        if not sub_queries:
            print("No sub-queries were generated.")
            return None, None, None
        
        all_contents = []
        found=False
        sub_results = []
        for sub_query in sub_queries:
            search_results = self.google_api.search(sub_query, num_results=10)
            sub_results[sub_query] = []
            for result in search_results:
                url = result.get('link')
                if url == website_content:
                    fetched = website_to_optimize
                    found=True
                else:
                    fetched = self.web_scraper.fetch_content(url)
                if fetched:
                    all_contents.append(fetched)
                    sub_results[sub_query].append(url)
        if not found:
            all_contents.append({'url': website_to_optimize, 'content': website_content})


        
        if len(all_contents) == 1: 
            raise NotImplementedError
        
        self.rag_system.add_documents(all_contents)

        relevant_documents, scores = self.rag_system.retrieve_relevant_documents(user_query)
        
        final_answer = self.llm_prompt.generate_response(user_query, relevant_documents, instructions=self.llm_generation_instructions)

        return sub_results, scores, final_answer




