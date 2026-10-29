"""Chunk, embed, and index biomedical abstracts using SentenceTransformers and FAISS.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """Simple word-based chunker with sliding window overlap.

    Args:
        text: Input document text.
        chunk_size: Approximate words per chunk.
        overlap: Word overlap between chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += (chunk_size - overlap)
    return chunks


def prepare_chunks(records: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Break records into chunks with tracked metadata.

    Args:
        records: List of document dicts with pmid, title, abstract.

    Returns:
        (chunks, metadata_list)
    """
    all_chunks = []
    metadata = []

    for doc_idx, rec in enumerate(records):
        pmid = rec.get("pmid", str(doc_idx))
        title = rec.get("title", "")
        abstract = rec.get("abstract", "")
        full_text = f"{title}. {abstract}".strip()

        chunks = chunk_text(full_text)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "chunk_id": f"{pmid}_c{chunk_idx}",
                "pmid": pmid,
                "title": title,
                "chunk_text": chunk,
                "doc_idx": doc_idx
            })

    logger.info(f"Generated {len(all_chunks)} chunks from {len(records)} documents.")
    return all_chunks, metadata


def build_faiss_index(
    chunks: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32
):
    """Compute dense embeddings and index them with FAISS.

    Args:
        chunks: List of chunk text strings.
        model_name: HuggingFace model identifier.
        batch_size: Embedding batch size.

    Returns:
        (faiss_index, embeddings)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}")

    # Use Inner Product (cosine similarity because embeddings are normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info(f"Built FAISS index with {index.ntotal} vectors.")

    return index, embeddings


def save_index_and_metadata(index, metadata: List[Dict[str, Any]], prefix: str = "verimed"):
    """Persist FAISS index and metadata to disk."""
    import faiss
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / f"{prefix}.index"
    meta_path = INDEX_DIR / f"{prefix}_metadata.pkl"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"FAISS index saved to: {index_path}")
    logger.info(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chunk documents and build vector index.")
    parser.add_argument("--data_file", type=str, default="pubmed_abstracts.json", help="Filename in data/ directory")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    args = parser.parse_args()

    data_path = DATA_DIR / args.data_file
    if not data_path.exists():
        logger.warning(f"Data file {data_path} not found. Creating mock data for indexing demonstration...")
        mock_data = [
            {"pmid": "1001", "title": "Metformin therapy in type 2 diabetes", "abstract": "Metformin reduces hepatic glucose production and improves insulin sensitivity."},
            {"pmid": "1002", "title": "GLP-1 receptor agonists and heart disease", "abstract": "GLP-1 receptor agonists reduce major adverse cardiovascular events in high-risk patients."},
            {"pmid": "1003", "title": "Dexamethasone efficacy in respiratory failure", "abstract": "Dexamethasone demonstrates mortality benefit in patients hospitalized with severe respiratory viral infection requiring mechanical ventilation."}
        ]
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(mock_data, f, indent=2)

    with open(data_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    chunks, metadata = prepare_chunks(docs)
    index, embeddings = build_faiss_index(chunks, model_name=args.model)
    save_index_and_metadata(index, metadata)
