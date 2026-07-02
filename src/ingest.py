"""Build the FAISS index the chatbot retrieves from.

Downloads the reference textbook, splits it into overlapping chunks, embeds
each chunk, and saves a local FAISS index. Run this once before starting the
chatbot:

    python -m src.ingest
"""
from __future__ import annotations

import os
import urllib.request

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXTBOOK_DIR = "textbooks"
INDEX_DIR = "faiss_index"

# Officially free, author-distributed PDF — safe to script-download.
# Do not commit the PDF or the built index to the repo (see .gitignore).
SOURCES = {
    "MML.pdf": "https://mml-book.github.io/book/mml-book.pdf",
}


def _has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def download_textbooks() -> None:
    os.makedirs(TEXTBOOK_DIR, exist_ok=True)
    for filename, url in SOURCES.items():
        path = os.path.join(TEXTBOOK_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"{filename} already present, skipping download.")


def build_index() -> FAISS:
    download_textbooks()

    all_docs = []
    for filename in SOURCES:
        print(f"Loading {filename}...")
        loader = PyPDFLoader(os.path.join(TEXTBOOK_DIR, filename))
        docs = loader.load()
        for d in docs:
            d.metadata["source_book"] = filename.replace(".pdf", "")
        all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} pages total.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if _has_cuda() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/")
    return vectorstore


if __name__ == "__main__":
    build_index()
