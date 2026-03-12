from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


def get_embeddings():
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Anthropic doesn't have embeddings, use OpenAI or local HuggingFace
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model=model)
    else:
        # Fallback to local HuggingFace embeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
