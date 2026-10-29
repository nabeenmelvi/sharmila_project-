"""Ingest module for VeriMed-RAG.

Downloads and cleans PubMed abstracts and biomedical question datasets.
Uses BioPython Entrez API or local PubMed XML/JSON sources.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def fetch_pubmed_abstracts(
    query: str,
    max_results: int = 20,
    email: str = "user@example.com"
) -> List[Dict[str, Any]]:
    """Fetch PubMed abstracts using BioPython Entrez API.

    Args:
        query: Search term or MeSH query.
        max_results: Maximum number of abstracts to download.
        email: Email for NCBI Entrez API requirement.

    Returns:
        List of dicts containing pmid, title, abstract, journal, and year.
    """
    try:
        from Bio import Entrez, Medline
    except ImportError:
        logger.error("BioPython is not installed. Please install biopython.")
        return []

    Entrez.email = email
    logger.info(f"Searching PubMed for: '{query}' (max_results={max_results})")
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        id_list = record.get("IdList", [])
        logger.info(f"Found {len(id_list)} PubMed IDs.")

        if not id_list:
            return []

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        
        abstracts = []
        for rec in records:
            pmid = rec.get("PMID", "")
            title = rec.get("TI", "")
            abstract = rec.get("AB", "")
            journal = rec.get("JT", "")
            year = rec.get("DP", "")[:4] if rec.get("DP") else ""

            if abstract:
                clean_record = {
                    "pmid": pmid,
                    "title": clean_text(title),
                    "abstract": clean_text(abstract),
                    "journal": journal,
                    "year": year,
                    "source": "PubMed"
                }
                abstracts.append(clean_record)
        
        handle.close()
        logger.info(f"Successfully processed {len(abstracts)} valid abstracts.")
        return abstracts
    except Exception as e:
        logger.error(f"Error fetching from PubMed: {e}")
        return []


def clean_text(text: str) -> str:
    """Basic text cleaning for biomedical text.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    # Normalize whitespaces and linebreaks
    lines = text.splitlines()
    cleaned = " ".join(line.strip() for line in lines if line.strip())
    return cleaned


def save_dataset(records: List[Dict[str, Any]], filename: str = "pubmed_abstracts.json") -> Path:
    """Save processed records into the data directory in JSON and CSV format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DATA_DIR / filename
    out_csv = DATA_DIR / (Path(filename).stem + ".csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    
    logger.info(f"Saved {len(records)} records to {out_json} and {out_csv}")
    return out_json


def create_sample_benchmark_questions() -> Path:
    """Create sample biomedical verification / QA benchmark questions."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_questions = [
        {
            "id": "q1",
            "question": "What is the primary mechanism of action of metformin in type 2 diabetes?",
            "ground_truth_pmids": ["12345678"],
            "expected_answer": "Metformin primarily suppresses hepatic gluconeogenesis via activation of AMPK."
        },
        {
            "id": "q2",
            "question": "Are GLP-1 receptor agonists associated with reduced cardiovascular mortality?",
            "ground_truth_pmids": ["23456789"],
            "expected_answer": "Yes, large clinical trials have demonstrated cardiovascular risk reduction."
        },
        {
            "id": "q3",
            "question": "Does dexamethasone improve survival in severe COVID-19 patients requiring oxygen?",
            "ground_truth_pmids": ["34567890"],
            "expected_answer": "Yes, as demonstrated in the RECOVERY trial."
        }
    ]
    out_file = DATA_DIR / "benchmark_questions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, indent=2)
    logger.info(f"Sample benchmark questions created at {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PubMed articles and questions.")
    parser.add_argument("--query", type=str, default="diabetes metformin clinical trial", help="PubMed search query")
    parser.add_argument("--max_results", type=int, default=10, help="Max results to fetch")
    parser.add_argument("--email", type=str, default="user@example.com", help="NCBI email")
    args = parser.parse_args()

    logger.info("Starting ingestion workflow...")
    create_sample_benchmark_questions()
    abstracts = fetch_pubmed_abstracts(query=args.query, max_results=args.max_results, email=args.email)
    if abstracts:
        save_dataset(abstracts)
    else:
        logger.warning("No abstracts fetched. Sample benchmark questions saved.")
