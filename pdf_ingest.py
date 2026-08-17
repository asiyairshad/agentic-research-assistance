import os
import uuid
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from dotenv import load_dotenv
load_dotenv()
class DocumentStore:
    """Loads, chunks, embeds and stores user-uploaded PDFs in an in-memory Chroma store. Docs persist for the life of the running app (across
    uploads/queries in a session) but clear on restart/reload — that's fine,
    since only chat memory needs to survive restarts here"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    def ingest(self, file_path:str, user_id:str, filename:str )-> str:
        """Loads a pdf, split vit, embed each chunk, and stores it. Returns doc_id so this file can be referenced/queried/deleted lated."""

        doc_id = str(uuid.uuid4())
        pages = PyPDFLoader(file_path).load()
        chunks = self.splitter.split_documents(pages)

        for chunk in chunks:
            chunk.metadata.update({"user_id": user_id, "doc_id":doc_id, "filename":filename})

        self.vectorstore.add_documents(chunks)
        return doc_id



    def list_user_docs(self, user_id: str) -> list[dict]:
        raw = self.vectorstore.get(where={"user_id": user_id})
        seen = {}
        for meta in raw["metadatas"]:
            seen[meta["doc_id"]] = meta["filename"]
        return [{"doc_id": k, "filename": v} for k, v in seen.items()]
    def delete_doc(self, user_id:str, doc_id:str) -> None:
        self.vectorstore.delete(where={"$and":[{"user_id":user_id}, {"doc_id": doc_id}]})

_store: Optional[DocumentStore] = None

def get_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store