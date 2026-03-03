"""
Loads documents into a vectorstore.
Provides reusable functions for embeddings, and vectorstore.
"""

import hashlib
import os
import requests
from typing import Optional
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

def setup_embedding_function(
    server_url: str = "http://trac-malenia.ern.nps.edu:8080/gpu5/embed",
    fallback_model: str = "all-MiniLM-L6-v2"
) -> HuggingFaceEndpointEmbeddings:
    """
    Initialize HuggingFace embedding function with server fallback.

    Args:
        server_url: URL of the embedding server
        fallback_model: Local model to use if server unavailable

    Returns:
        Configured embedding function
    """
    try:
        hfe = HuggingFaceEndpointEmbeddings(model=server_url)
        print("Embedding function set from server")
        return hfe
    except Exception as e:
        print(f"Server connection failed: {e}")
        print("Setting up local embedding function...")

        hfe = HuggingFaceEmbeddings(
            model_name=fallback_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        print(f"Embedding function set locally with {fallback_model}")
        return hfe
    

def load_vectorstore(
    persist_directory: str = "./database/chroma_recursive_db",
    embedding_function: Optional[HuggingFaceEmbeddingServer] = None
) -> Chroma:
    """
    Load an existing Chroma vectorstore.

    Args:
        persist_directory: Path to the persisted vectorstore
        embedding_function: Embedding function to use (if None, creates default)

    Returns:
        Loaded Chroma vectorstore
    """
    if embedding_function is None:
        embedding_function = setup_embedding_function()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    num_docs = len(vectorstore.get()['ids'])
    print(f"Loaded vectorstore with {num_docs} documents")

    return vectorstore

def load_db_from_dir(directory="./docs", chromadb_path="./databases/characters"):
    """
    Loads all pdfs into a chromadb vectorstore
    """
    doc_paths = [os.path.join(directory, filename)
                 for filename in os.listdir(directory)
                 if filename.endswith('.pdf')]

    # Extract text into documents
    docs = [PyMuPDFLoader(doc_path).load() for doc_path in doc_paths]
    docs_flat = [item for sublist in docs for item in sublist]
    print(f'Total Pages: {len(docs_flat)}')

    # Split documents into chunks
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    recursive_docs = recursive_splitter.transform_documents(docs_flat)
    print(f'Total Chunks: {len(recursive_docs)}')

    # Add unique chunk IDs for vectorstore
    for i, chunk in enumerate(recursive_docs, start=1):
        source = chunk.metadata.get('source', 'unknown')
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        chunk.metadata['id'] = f"{source}_{content_hash}_{i}"

    print(f"Added IDs to {len(recursive_docs)} chunks")
    emb_function = setup_embedding_function()

    # Create Chroma vectorstore and add documents
    vectorstore = Chroma(
        persist_directory=chromadb_path,
        embedding_function=emb_function
    )

    # Add documents in batches to avoid memory issues
    batch_size = 32
    print("Adding documents to vectorstore...")

    for i in tqdm(range(0, len(recursive_docs), batch_size)):
        batch = recursive_docs[i:i + batch_size]
        batch_ids = [doc.metadata["id"] for doc in batch]
        vectorstore.add_documents(documents=batch, ids=batch_ids)

    print(f"\nTotal documents processed: {len(recursive_docs)}")
    print(f"Documents in vectorstore: {len(vectorstore.get()['ids'])}")

if __name__ == "__main__":
    doc_dir = "./docs"
    chroma_db_persist_path = "./databases/characters"
    load_db_from_dir(directory=doc_dir, chromadb_path=chroma_db_persist_path)
