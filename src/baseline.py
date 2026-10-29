"""Baseline models (M1 and M2) for VeriMed-RAG benchmark.

- M1: Lexical baseline (Sparse BM25 matching).
- M2: Dense Embedding baseline (Bi-encoder Semantic Search with FAISS).
"""

import json
import logging
from typing import List, Dict, Any
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.retrieve import DenseRetriever, BM25Retriever
except ImportError:
    from retrieve import DenseRetriever, BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


class M1LexicalBaseline:
    """M1 Baseline: BM25 Lexical Matching."""

    def __init__(self, data_file: str = "pubmed_abstracts.json"):
        path = DATA_DIR / data_file
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        corpus_texts = [f"{d.get('title', '')} {d.get('abstract', '')}" for d in docs]
        metadata = [{"pmid": d.get("pmid", str(i)), "title": d.get("title", ""), "chunk_text": corpus_texts[i]} for i, d in enumerate(docs)]
        self.retriever = BM25Retriever(corpus_texts, metadata)

    def run_query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.retriever.search(query, top_k=top_k)
        for r in results:
            r["baseline_model"] = "M1_BM25"
        return results


class M2DenseBaseline:
    """M2 Baseline: Dense Vector Retrieval with FAISS."""

    def __init__(self, prefix: str = "verimed"):
        self.retriever = DenseRetriever(prefix=prefix)

    def run_query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.retriever.search(query, top_k=top_k)
        for r in results:
            r["baseline_model"] = "M2_Dense_FAISS"
        return results


def run_baselines_on_benchmark(questions_file: str = "benchmark_questions.json", top_k: int = 5):
    """Run both M1 and M2 baselines against benchmark questions and record outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    qpath = DATA_DIR / questions_file

    if not qpath.exists():
        logger.warning(f"No benchmark questions found at {qpath}")
        return

    with open(qpath, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} evaluation queries.")

    m1 = None
    try:
        m1 = M1LexicalBaseline()
    except Exception as e:
        logger.warning(f"Could not initialize M1: {e}")

    m2 = None
    try:
        m2 = M2DenseBaseline()
    except Exception as e:
        logger.warning(f"Could not initialize M2: {e}")

    outputs = []
    for q in questions:
        query_text = q.get("question", "")
        qid = q.get("id", "")
        gt = q.get("ground_truth_pmids", [])

        m1_res = m1.run_query(query_text, top_k=top_k) if m1 else []
        m2_res = m2.run_query(query_text, top_k=top_k) if m2 else []

        outputs.append({
            "question_id": qid,
            "question": query_text,
            "ground_truth_pmids": gt,
            "m1_predictions": [r.get("pmid") for r in m1_res],
            "m2_predictions": [r.get("pmid") for r in m2_res],
            "m1_results": m1_res,
            "m2_results": m2_res
        })

    out_file = RESULTS_DIR / "baseline_runs.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    logger.info(f"Baseline benchmark outputs saved to {out_file}")


if __name__ == "__main__":
    run_baselines_on_benchmark()
