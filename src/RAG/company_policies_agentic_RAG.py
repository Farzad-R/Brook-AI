"""
Lookup Company Policies (RAG):

The assistant retrieve policy information to answer user questions. Note that enforcement of these policies still must be done within the tools/APIs themselves, since the LLM can always ignore this.
"""

import numpy as np
import openai
from langchain_core.tools import tool


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
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
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
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


@tool
def lookup_policy(query: str, retriever) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])


"""
lookup_policy full docstring:
-----------------------------
    Consults company policies to check whether certain options are permitted.

    Uses the retriever to find and return the most relevant sections of the FAQ document that
    pertain to the given query. This is useful for checking policy details before making changes
    or performing write events.

    Parameters:
        query (str): The query string to look up in the company policies.
        retriever: An instance of VectorStoreRetriever class.

    Returns:
        str: A string containing the contents of the most relevant sections of the FAQ document.
"""
