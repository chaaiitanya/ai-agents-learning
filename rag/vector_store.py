import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

from .embeddings import get_embeddings

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")


class VectorStore:
    def __init__(self, collection_name: str = "main"):
        self.embeddings = get_embeddings()
        self.collection_name = collection_name
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

    def add_documents(self, docs: list[Document]) -> list[str]:
        chunks = self.splitter.split_documents(docs)
        ids = self.store.add_documents(chunks)
        return ids

    def ingest_file(self, file_path: str) -> list[str]:
        path = Path(file_path)
        if path.suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif path.suffix == ".csv":
            loader = CSVLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()
        return self.add_documents(docs)

    def ingest_text(self, text: str, metadata: dict = None) -> list[str]:
        doc = Document(page_content=text, metadata=metadata or {})
        return self.add_documents([doc])

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, k=k)

    def as_retriever(self, k: int = 5):
        return self.store.as_retriever(search_kwargs={"k": k})
