import os
from typing import List, Optional, Tuple

import openai
from actionweaver import ActionHandlerMixin, RequireNext, SelectOne, action
from actionweaver.llms.openai.chat import OpenAIChatCompletion
from actionweaver.llms.openai.tokens import TokenUsageTracker
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    SimpleWebPageReader,
    VectorStoreIndex,
)

openai.api_key = os.getenv("OPENAI_API_KEY")


class RAGBot(ActionHandlerMixin):
    def __init__(self, logger, st):
        self.index = VectorStoreIndex.from_documents([])
        self.logger = logger
        self.st = st
        self.token_tracker = TokenUsageTracker(budget=3000, logger=logger)
        self.llm = OpenAIChatCompletion(
            "gpt-4", token_usage_tracker=self.token_tracker, logger=logger
        )

        system_str = "You are a helpful assistant. Choose one of the following actions to get started: 'AnswerQuestion' or 'Read.' Please do not try to answer the question directly."
        self.messages = [{"role": "system", "content": system_str}]

    def __call__(self, query):
        self.messages.append({"role": "user", "content": query})

        return self.llm.create(
            self.messages,
            stream=True,
            orch_expr=SelectOne(["AnswerQuestion", "Read"]),
        )

    @action(name="AnswerQuestion", stop=True)
    def answer_question(self, query: str):
        """
        Answer a question.

        Parameters
        ----------
        query : str
            The query to be used for answering a question.
        """

        context_query = self.llm.create(
            [
                {
                    "role": "system",
                    "content": "You are an assistant extracting key information from text",
                },
                {"role": "user", "content": f"Query : {query}"},
            ],
            orch_expr=RequireNext(["ExtractQueryForKB"]),
        )

        context_str = self.recall(context_query)
        context = (
            "Information from knowledge base:\n"
            "---\n"
            f"{context_str}\n"
            "---\n"
            f"User: {query}\n"
            "Only answer question based on information from knowledge base"
            "If you don't have information in the knowledge base, performs a Google search instead. Your Response:"
        )

        response = self.llm.create(
            [
                {"role": "user", "content": context},
            ],
            orch_expr=SelectOne(["GoogleSearch"]),
        )

        self.messages.append({"role": "assistant", "content": response})
        return response

    @action(name="ExtractQueryForKB", scope="kb", stop=True)
    def extract_query_for_knowledge_base(self, kb_query: str):
        """
        Extract a query for retrieving relevant information from the knowledge base.

        Parameters
        ----------
        kb_query : str
            The query to be used for retrieving information from the knowledge base.
        """
        return kb_query

    @action(name="GoogleSearch", stop=True, scope="search")
    def search(self, query: str):
        """
        Perform a Google search and return query results with titles and links.

        Parameters
        ----------
        query : str
            The search query to be used for the Google search.

        Returns
        -------
        str
            A formatted string containing Google search results with titles, snippets, and links.
        """

        with self.st.spinner(f"Searching '{query}'..."):
            from langchain.utilities import GoogleSearchAPIWrapper

            search = GoogleSearchAPIWrapper()
            res = search.results(query, 10)
            formatted_data = ""

            # Iterate through the data and append each item to the formatted_data string
            for idx, item in enumerate(res):
                formatted_data += f"({idx}) {item['title']}: {item['snippet']}\n"
                formatted_data += f"[Source]: {item['link']}\n\n"

        return f"Here are Google search results:\n\n{formatted_data}"

    def recall(self, text):
        """
        Recall from your knowledge base using the provided text and include sources.

        Parameters
        ----------
        text : str
            The query text used to search the agent's knowledge base.

        Returns
        -------
        str
            A response containing relevant information retrieved from the knowledge base along with sources.
            If no information is found, it returns "No information on that topic."
        """
        query_engine = self.index.as_query_engine()
        response = query_engine.query(text)

        sources = []
        for v in response.metadata.values():
            sources += v["source"]

        sources = list(set(sources))

        if response.response:
            return f"{response.response}\n\n[Source]: {sources}"
        else:
            return "No information on that topic."

    def contains_url(self, text):
        import re

        # Regular expression pattern for matching URLs
        url_pattern = r"https?://\S+|www\.\S+"

        # Search for URLs in the input text
        if re.search(url_pattern, text):
            return True
        else:
            return False

    @action("Read", stop=True)
    def read(self, sources: str):
        """
        Read content from various sources.

        Parameters
        ----------
        sources : str
            The source identifier, which can be a web link or a file path, e.g. "https://www.example.com", "/path/to/your/local/file.txt".
        """
        if self.contains_url(sources):
            return self.llm.create(
                [{"role": "user", "content": sources}],
                orch_expr=RequireNext(["ReadURL"]),
            )

        return self.llm.create(
            [{"role": "user", "content": sources}],
            orch_expr=RequireNext(["ReadFile"]),
        )

    @action("ReadURL", scope="read", stop=True)
    def read_url(self, urls: List[str]):
        """
        Read the content from the provided web links.

        Parameters
        ----------
        urls : List[str]
            List of URLs to scrape.

        Returns
        -------
        str
            A message indicating successful reading of content from the provided URLs.
        """

        with self.st.spinner(f"Learning the content in {urls}"):
            service_context = ServiceContext.from_defaults(chunk_size=512)
            documents = SimpleWebPageReader(html_to_text=True).load_data(urls)

            for doc in documents:
                doc.metadata = {"source": urls}
                self.index.insert(
                    doc,
                    service_context=service_context,
                )
        return f"Contents in URLs {urls} have been successfully learned."

    @action("ReadFile", scope="read", stop=True)
    def read_file(
        self,
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        *args,
        **kwargs,
    ):
        """
        Read the content from provided files or directories.

        Parameters
        ----------
        input_dir : str, optional
            Path to the directory (default is None).
        input_files : List, optional
            List of file paths to read (overrides input_dir if provided).

        Returns
        -------
        str
            A message indicating successful reading of content from the files.
        """
        reader = SimpleDirectoryReader(input_dir=input_dir, input_files=input_files)
        with self.st.spinner(
            f"Learning the content in {[str(file) for file in reader.input_files]}"
        ):
            service_context = ServiceContext.from_defaults(chunk_size=512)
            documents = reader.load_data()
            for doc in documents:
                doc.metadata = {
                    "source": {"input_dir": input_dir, "input_files": input_files}
                }
                self.index.insert(doc, service_context=service_context)
        return f"Contents in files {[str(file) for file in reader.input_files]} have been successfully learned."


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="bot.log",
        filemode="a",
        format="%(asctime)s.%(msecs)04d %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    agent = RAGBot(logger, None)
