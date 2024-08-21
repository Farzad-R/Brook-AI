
from company_policies_agentic_RAG_prep import load_swiss_faq, VectorStoreRetriever
import openai
from langchain_core.tools import tool
from load_notebook_config import LoadRAGConfig

RAG_CFG = LoadRAGConfig()

docs = load_swiss_faq()
retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


@tool
def lookup_policy(query: str) -> str:
    """
    Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events.

    Args:
        query (str): The query string to look up in the company policies.

    Returns:
        str: A string containing the contents of the most relevant sections of the FAQ document.
    """
    docs = retriever.query(query, k=RAG_CFG.k)
    return "\n\n".join([doc["page_content"] for doc in docs])
