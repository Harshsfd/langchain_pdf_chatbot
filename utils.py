# utils.py
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader  # ✅ replaces pypdf

def load_pdf(path: str) -> List[Document]:
    """
    Load PDF from path and return list of Documents using LangChain's PyPDFLoader.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

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
        os.makedirs(persist_path, exist_ok=True)
        vectorstore.save_local(persist_path)
    return vectorstore
        
