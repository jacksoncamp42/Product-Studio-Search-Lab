# llm_prompt.py
import json
from langchain_core.output_parsers import JsonOutputParser



class LLMPrompt:
    """
    A class to handle prompt generation and response parsing for interactions with a language model (LLM).

    Attributes:
        llm_engine: An instance of the LLM engine used to generate responses.
        subquery_template (str): Template for generating sub-queries based on user input.
        response_template (str): Template for generating detailed responses based on user input and documents.
        parser: An instance of JsonOutputParser for parsing JSON output from the LLM.
        conversation_history (list): A list to maintain the conversation history with roles and content.
    """
    def __init__(self, llm_engine, subquery_template=None, response_template=None):
        """
        Initializes the LLMPrompt instance with an LLM engine and optional templates.

        Args:
            llm_engine: The LLM engine used to generate responses.
            subquery_template (str, optional): A custom template for generating sub-queries (default is None).
            response_template (str, optional): A custom template for generating responses (default is None).
        """
        self.llm_engine = llm_engine
        self.subquery_template = subquery_template or self.default_subquery_template()
        self.response_template = response_template or self.default_response_template()
        self.parser = JsonOutputParser()
        self.conversation_history = []  # To maintain conversation history

    def default_subquery_template(self):
        """
        Provides the default template for generating sub-queries from a user query.

        Returns:
            str: A default sub-query generation template as a string.
        """
        return """
                Generate concise search queries based on the following user question. 
                Each of these queries should help you gain additional information about the question you are trying to answer. 
                Limit yourself to 5 queries. 

                User Query:
                "{user_query}"

                Output the search queries as a JSON-formatted list of strings. Ensure the JSON is valid and parsable.

                Example Output:
                ["search query 1", "search query 2", "search query 3"]
                """

    def default_response_template(self):
        """
        Provides the default template for generating responses from documents and conversation history.

        Returns:
            str: A default response generation template as a string.
        """
        return """
                You are a knowledgeable assistant. Use the information from the provided documents to answer the user's question in detail. 
                Include hyperlinked sources within your answer using the format [text](URL). At the end of your answer, include a list of the sources with their URLs.

                Instructions:
                - Provide a detailed and informative answer to the user's question.
                - When you mention information from a document, hyperlink the relevant text to the document's URL.
                - At the end, list all the sources with their corresponding URLs under a "Sources" heading.

                Conversation History:
                {conversation_history}

                Documents:
                {documents}
                """

    def generate_subqueries(self, user_query, instructions=None):
        """
        Generates structured sub-queries from the user's input query.

        Args:
            user_query (str): The user's input query.
            instructions (str, optional): Additional instructions for the sub-query generation prompt (default is None).

        Returns:
            list: A list of generated sub-queries as strings, or an empty list if parsing fails.
        """
        prompt = self.subquery_template.format(user_query=user_query)
        if instructions:
            prompt = prompt + "\n\n" + instructions

        llm_input = [
            ("system", prompt)
        ]

        response = self.llm_engine.generate(llm_input)
        response_content = response.content.strip()

        try:
            sub_queries = self.parser.parse(response_content)
            if not isinstance(sub_queries, list):
                raise ValueError("Parsed output is not a list.")
            return sub_queries
        except Exception as e:
            print(f"Failed to parse structured output: {e}")
            return []

    def generate_response(self, user_query, documents_with_urls, instructions=None):
        """
        Generates a detailed response to the user's query using provided documents.

        Args:
            user_query (str): The user's input query.
            documents_with_urls (list): A list of dictionaries containing document content and URLs.
            instructions (str, optional): Additional instructions for the response generation prompt (default is None).

        Returns:
            str: A detailed response generated by the LLM.
        """
        # Format documents with indices, content, and URLs
        documents_formatted = ""
        for idx, doc in enumerate(documents_with_urls, start=1):
            doc_excerpt = doc['content'][:1000]  # Truncate if necessary
            doc_url = doc['url']
            documents_formatted += f"{idx}. {doc_excerpt}\nURL: {doc_url}\n\n"

        # Prepare conversation history
        conversation_formatted = ""
        for turn in self.conversation_history:
            role = turn['role']
            content = turn['content']
            conversation_formatted += f"{role}: {content}\n"

        prompt = self.response_template.format(
            conversation_history=conversation_formatted,
            documents=documents_formatted
        )

        if instructions:
            prompt = prompt + "\n\n" + instructions

        llm_input = [
            ("system", prompt),
            ("user", user_query),
        ]

        response = self.llm_engine.generate(llm_input)
        response_content = response.content.strip()

        # Update conversation history
        self.conversation_history.append({'role': 'user', 'content': user_query})
        self.conversation_history.append({'role': 'assistant', 'content': response_content})

        return response_content
 

if __name__ == "__main__":
    from llm_engine import LLMEngine
    llm_engine = LLMEngine(api_key="")
    llm_prompt = LLMPrompt(llm_engine)

    user_query = "What is the impact of climate change on polar bears?"

    sub_queries = llm_prompt.generate_subqueries(user_query)
    print("Generated Sub-Queries:")
    print(sub_queries)
