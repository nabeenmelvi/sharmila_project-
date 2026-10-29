"""Retrieval module for VeriMed-RAG.

Supports dense vector search with FAISS, BM25 keyword search, and reciprocal rank fusion hybrid search.
"""

import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "index"


class DenseRetriever:
    """Vector search retriever using FAISS and SentenceTransformers."""

    def __init__(self, prefix: str = "verimed", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index_path = INDEX_DIR / f"{prefix}.index"
        self.meta_path = INDEX_DIR / f"{prefix}_metadata.pkl"

        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Index or metadata not found at {self.index_path}")

        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(f"Loading embedding model {model_name}")
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search top-k most similar passages for the given query."""
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            item["rank"] = rank + 1
            item["retriever"] = "dense"
            results.append(item)
        return results


class BM25Retriever:
    """Lexical keyword retriever using Rank-BM25."""

    def __init__(self, corpus: List[str], metadata: List[Dict[str, Any]]):
        from rank_bm25 import BM25Okapi

        self.metadata = metadata
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search top-k passages matching query terms."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            item = dict(self.metadata[idx])
            item["score"] = float(scores[idx])
            item["rank"] = rank + 1
            item["retriever"] = "bm25"
            results.append(item)
        return results


def hybrid_search(dense_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], top_k: int = 5, k: int = 60) -> List[Dict[str, Any]]:
    """Combine dense and sparse search results using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for doc in dense_results:
        cid = doc["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + doc["rank"]))
        doc_map[cid] = doc

    for doc in bm25_results:
        cid = doc["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + doc["rank"]))
        if cid not in doc_map:
            doc_map[cid] = doc

    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    fused_results = []
    for rank, (cid, score) in enumerate(sorted_chunks):
        item = dict(doc_map[cid])
        item["rrf_score"] = score
        item["rank"] = rank + 1
        item["retriever"] = "hybrid_rrf"
        fused_results.append(item)

    return fused_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query the index.")
    parser.add_argument("--query", type=str, default="What are the cardiovascular effects of GLP-1?", help="Query string")
    parser.add_argument("--top_k", type=int, default=3, help="Number of passages to retrieve")
    args = parser.parse_args()

    try:
        retriever = DenseRetriever()
        results = retriever.search(args.query, top_k=args.top_k)
        print(f"\nResults for query: '{args.query}':\n")
        for res in results:
            print(f"Rank {res['rank']} [Score: {res['score']:.4f}] PMID: {res['pmid']}")
            print(f"Title: {res['title']}")
            print(f"Chunk: {res['chunk_text']}\n")
    except Exception as e:
        logger.error(f"Error querying index: {e}")
