

from __future__ import annotations

import sys
from uuid import uuid4
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()

CHUNK_SIZE = 1_000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_estate"
VECTOR_DB_DIR = Path("resources") / "vectorstore"

# Make sure the persistence folder exists
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

_llm: ChatGroq | None = None
_vector_store: Chroma | None = None


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def get_llm() -> ChatGroq:
    """Lazily initialise the LLM so it is only created once."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3-70b-versatile",
            temperature=0.0,
            max_tokens=512,
        )
    return _llm


def get_vector_store() -> Chroma:
    """Lazily load (or create) the local Chroma collection."""
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"trust_remote_code": True}
        )
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTOR_DB_DIR),
        )
    return _vector_store


def ingest(urls: List[str]) -> None:
    """Fetch, chunk, and embed the content from *urls*."""
    vs = get_vector_store()
    print("⏳ Resetting collection …", flush=True)
    vs.reset_collection()

    print("⏳ Fetching URLs …", flush=True)
    loader = UnstructuredURLLoader(urls=urls)
    raw_docs = loader.load()

    print("⏳ Chunking …", flush=True)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "], chunk_size=CHUNK_SIZE
    )
    docs = splitter.split_documents(raw_docs)

    print(f"⏳ Adding {len(docs)} chunks to vector store …", flush=True)
    vs.add_documents(docs, ids=[str(uuid4()) for _ in docs])
    vs.persist()
    print("✅ Ingestion complete.")


def ask(question: str) -> None:
    """Run a retrieval‑augmented generation query and print answer + sources."""
    vs = get_vector_store()
    if vs._collection.count() == 0:  # pylint: disable=protected-access
        sys.exit("Vector store is empty. Ingest documents first.")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=get_llm(), retriever=vs.as_retriever()
    )
    print("⏳ Thinking …", flush=True)
    result = chain({"question": question})
    print(f"\nAnswer:\n{result['answer']}\n")
    if sources := result.get("sources"):
        print("Sources:")
        for src in sources.split("\n"):
            print(f"- {src}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            "Usage:\n  python real_estate_qa.py ingest <url1> <url2> …\n  python real_estate_qa.py ask \"<your question>\""
        )

    command, *args = sys.argv[1:]
    match command:
        case "ingest":
            if not args:
                sys.exit("Please provide at least one URL to ingest.")
            ingest(args)
        case "ask":
            ask(" ".join(args))
        case _:
            sys.exit(f"Unknown command: {command}")
