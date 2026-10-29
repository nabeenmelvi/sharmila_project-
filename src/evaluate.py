"""Evaluation metrics module for VeriMed-RAG.

Calculates standard Information Retrieval (IR) and QA metrics:
- Precision@K
- Recall@K
- Hit Rate@K (Success Rate)
- MRR@K (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
"""

import json
import math
import logging
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Compute Precision@K."""
    if not retrieved_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    hits = sum(1 for item in top_k if item in gt_set)
    return hits / k


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Compute Recall@K."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    hits = sum(1 for item in top_k if item in gt_set)
    return hits / len(gt_set)


def hit_rate_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Compute HitRate@K (1 if at least one relevant document in top-K, else 0)."""
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    return 1.0 if any(item in gt_set for item in top_k) else 0.0


def reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Compute Reciprocal Rank at K (1 / rank of first relevant item)."""
    gt_set = set(ground_truth_ids)
    for rank, item in enumerate(retrieved_ids[:k], start=1):
        if item in gt_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)

    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        rel = 1.0 if item in gt_set else 0.0
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gt_set), k) + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_predictions(runs: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
    """Evaluate predictions for each baseline model across all metrics."""
    models = ["m1", "m2"]
    metrics_summary = {}

    for model_key in models:
        pred_key = f"{model_key}_predictions"
        metrics_summary[model_key] = {}

        for k in k_values:
            precisions = []
            recalls = []
            hit_rates = []
            mrrs = []
            ndcgs = []

            for row in runs:
                preds = row.get(pred_key, [])
                gts = row.get("ground_truth_pmids", [])
                if not gts:
                    continue

                precisions.append(precision_at_k(preds, gts, k))
                recalls.append(recall_at_k(preds, gts, k))
                hit_rates.append(hit_rate_at_k(preds, gts, k))
                mrrs.append(reciprocal_rank(preds, gts, k))
                ndcgs.append(ndcg_at_k(preds, gts, k))

            num_queries = len(precisions) if precisions else 1
            metrics_summary[model_key][f"P@{k}"] = round(sum(precisions) / num_queries, 4)
            metrics_summary[model_key][f"R@{k}"] = round(sum(recalls) / num_queries, 4)
            metrics_summary[model_key][f"HitRate@{k}"] = round(sum(hit_rates) / num_queries, 4)
            metrics_summary[model_key][f"MRR@{k}"] = round(sum(mrrs) / num_queries, 4)
            metrics_summary[model_key][f"NDCG@{k}"] = round(sum(ndcgs) / num_queries, 4)

    return metrics_summary


def run_evaluation_pipeline(run_file: str = "baseline_runs.json"):
    """Run full evaluation and save reports in CSV and JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    in_path = RESULTS_DIR / run_file

    if not in_path.exists():
        logger.warning(f"Run file {in_path} not found. Please run baseline.py first.")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        runs = json.load(f)

    logger.info(f"Loaded {len(runs)} benchmark query runs.")
    summary = evaluate_predictions(runs)

    df = pd.DataFrame(summary).T
    csv_path = RESULTS_DIR / "evaluation_summary.csv"
    json_path = RESULTS_DIR / "evaluation_summary.json"

    df.to_csv(csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation complete. Saved to {csv_path} and {json_path}")
    print("\n=== Evaluation Results ===")
    print(df.to_string())


if __name__ == "__main__":
    run_evaluation_pipeline()
