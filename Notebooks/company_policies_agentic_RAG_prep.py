"""
Lookup Company Policies (RAG):

The assistant retrieve policy information to answer user questions. Note that enforcement of these policies still must be done within the tools/APIs themselves, since the LLM can always ignore this.
"""
import re
import requests
import numpy as np
from load_notebook_config import LoadDirectoriesConfig, LoadOpenAIConfig

CFG_OPENAI = LoadOpenAIConfig()
CFG_DIRECTORIES = LoadDirectoriesConfig()


def load_swiss_faq():
    """
    Fetches and processes the Swiss FAQ document from a remote URL.

    This function performs the following steps:
    1. Retrieves the FAQ text from a URL specified in the configuration.
    2. Splits the FAQ text into separate sections based on the section headers.
    3. Returns the processed FAQ sections as a list of dictionaries, where each dictionary
       contains the text of one section.

    The URL for the FAQ document is obtained from the `CFG.swiss_faq_url` configuration setting.

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a section of the FAQ. 
        Each dictionary has a single key:
            - "page_content": The text content of the FAQ section.

    Raises:
        requests.HTTPError: If the request to fetch the FAQ document fails.
        ValueError: If the FAQ content cannot be parsed correctly.
    """

    # Fetch the FAQ text from a remote URL
    response = requests.get(
        CFG_DIRECTORIES.swiss_faq_url
    )
    response.raise_for_status()
    faq_text = response.text

    # Split the FAQ text into separate documents based on section headers
    docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]
    return docs


class VectorStoreRetriever:
    """
    A class that retrieves documents based on similarity to a given query using vector embeddings.

    Attributes:
        docs (list): A list of documents, each represented as a dictionary with a "page_content" key.
        vectors (np.ndarray): An array of vector embeddings corresponding to the documents.
        oai_client (openai.Client): An instance of the OpenAI client used to generate embeddings.

    Methods:
        from_docs(docs, oai_client):
            Class method to create an instance of VectorStoreRetriever from a list of documents and an OpenAI client.
        query(query, k=5):
            Queries the retriever to find the top `k` most similar documents to the given query.
    """

    def __init__(self, docs: list, vectors: list, oai_client):
        """
        Initializes the VectorStoreRetriever with documents, their embeddings, and an OpenAI client.

        Parameters:
            docs (list): A list of documents, each represented as a dictionary with a "page_content" key.
            vectors (list): A list of vector embeddings corresponding to the documents.
            oai_client (openai.Client): An instance of the OpenAI client used to generate embeddings.
        """
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        """
        Creates an instance of VectorStoreRetriever from documents and an OpenAI client.

        Generates embeddings for the documents using the OpenAI client and initializes the retriever.

        Parameters:
            docs (list): A list of documents, each represented as a dictionary with a "page_content" key.
            oai_client (openai.Client): An instance of the OpenAI client used to generate embeddings.

        Returns:
            VectorStoreRetriever: An instance of the retriever with documents and their embeddings.
        """
        embeddings = oai_client.embeddings.create(
            model=CFG_OPENAI.embedding_model, input=[
                doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        """
        Queries the retriever to find the top `k` most similar documents to the given query.

        Uses the OpenAI client to generate an embedding for the query and computes similarity scores
        between the query embedding and the document embeddings.

        Parameters:
            query (str): The query string to be matched against the documents.
            k (int): The number of top similar documents to return. Default is 5.

        Returns:
            list[dict]: A list of dictionaries representing the top `k` most similar documents, each
                        including the document's content and its similarity score.
        """
        embed = self._client.embeddings.create(
            model=CFG_OPENAI.embedding_model, input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]
