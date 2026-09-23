"""
ingest.py
---------
Module 3 - Support Assistant (/support_assistant)

Loads the 8 policy documents from docs/, chunks them (one chunk per document
- each is a single short paragraph, so no further splitting is needed),
embeds each chunk locally with sentence-transformers' all-MiniLM-L6-v2 (no
API key, no cost, runs on-device), and stores the embeddings in a persistent
ChromaDB collection.

This module is imported by graph.py, and can also be run standalone to
(re)build the index:
    python ingest.py
"""

from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_client = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def load_chunks() -> List[Dict]:
    """
    One chunk per document, since each policy doc is already a single short
    paragraph (well under a typical chunk-size threshold). chunk_id doubles
    as the document id, e.g. 'doc_01'.
    """
    chunks = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        doc_id = path.stem  # e.g. "doc_01"
        text = path.read_text(encoding="utf-8").strip()
        chunks.append({"id": doc_id, "text": text})
    return chunks


def get_collection(rebuild: bool = False):
    """
    Returns a ChromaDB collection populated with embeddings for every chunk.
    Uses a persistent client so the index survives across FastAPI restarts;
    pass rebuild=True to force re-embedding (e.g. after editing docs/).
    """
    global _client, _collection
    if _collection is not None and not rebuild:
        return _collection

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if rebuild:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = _client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0 or rebuild:
        chunks = load_chunks()
        model = get_model()
        embeddings = model.encode([c["text"] for c in chunks]).tolist()
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
        )
        print(f"Indexed {len(chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}'")

    _collection = collection
    return _collection


def retrieve_top_k(query: str, k: int = 3):
    """
    Embeds the query and retrieves the top-k most similar chunks by cosine
    similarity. Returns a list of dicts: {id, text, distance}.
    """
    collection = get_collection()
    model = get_model()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=k)

    hits = []
    for doc_id, doc_text, dist in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        hits.append({"id": doc_id, "text": doc_text, "distance": dist})
    return hits


if __name__ == "__main__":
    get_collection(rebuild=True)
    print("\nSample query test: 'Can I cancel my order after it's packed?'")
    for hit in retrieve_top_k("Can I cancel my order after it's packed?", k=3):
        print(f"  [{hit['id']}] (distance={hit['distance']:.4f}) {hit['text'][:100]}...")
