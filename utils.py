# utils.py
import os
from typing import List
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_pdf(path: str) -> List[Document]:
    """
    Load PDF from path and return a list of LangChain Documents (one per page).
    Uses pypdf to avoid loader version mismatches.
    """
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        metadata = {"source": os.path.basename(path), "page": i + 1}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def split_docs(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split list of Documents into chunks useful for embeddings/search.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks: List[Document], persist: bool = False, persist_path: str = "faiss_index") -> FAISS:
    """
    Create FAISS vectorstore from chunks using OpenAIEmbeddings.
    If persist True, it will save index to persist_path.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    if persist:
        # create directory if needed then save
        os.makedirs(persist_path, exist_ok=True)
        vectorstore.save_local(persist_path)
    return vectorstore
    
