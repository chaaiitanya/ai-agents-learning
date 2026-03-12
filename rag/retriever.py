from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLanguageModel

from .vector_store import VectorStore


class RAGRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def retrieve_as_text(self, query: str, k: int = 5) -> str:
        docs = self.retrieve(query, k=k)
        if not docs:
            return "No relevant documents found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def build_qa_chain(self, llm: BaseLanguageModel) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
        )
