"""Ingest module for VeriMed-RAG.

Downloads PubMed abstracts WITH the metadata the Evidence Agent needs
(publication type, year, MeSH terms, DOI), stratified by evidence level so
that guidelines, systematic reviews and RCTs are all represented.

Setup (optional but recommended):
    set NCBI_EMAIL=you@example.com        (Windows: setx NCBI_EMAIL "...")
    set NCBI_API_KEY=your_ncbi_key        (free at ncbi.nlm.nih.gov/account)

Examples:
    python src/ingest.py --per_stratum 20          # quick test run
    python src/ingest.py --per_stratum 500         # real corpus
    python src/ingest.py --topics hypertension --append
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Two topics only, as planned. Add more later if time allows.
TOPICS: Dict[str, str] = {
    "type2_diabetes": '"Diabetes Mellitus, Type 2"[mh]',
    "hypertension": '"Hypertension"[mh]',
}

# Ordered from highest to lowest evidence. A paper is assigned to the FIRST
# stratum it matches, so a guideline is never counted again as "other".
STRATA: Dict[str, str] = {
    "guideline": "(Guideline[pt] OR Practice Guideline[pt])",
    "systematic_review": "(Meta-Analysis[pt] OR Systematic Review[pt])",
    "rct": "Randomized Controlled Trial[pt]",
    "observational": "(Observational Study[pt] OR Cohort Studies[mh] OR Case-Control Studies[mh])",
    "review": "Review[pt]",
    "other": "Journal Article[pt]",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace and line breaks."""
    if not text:
        return ""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _setup_entrez(email: Optional[str]):
    """Configure Entrez and return (Entrez, Medline, pause_seconds)."""
    try:
        from Bio import Entrez, Medline
    except ImportError as e:
        raise SystemExit("BioPython is not installed. Run: pip install biopython") from e

    Entrez.email = email or os.getenv("NCBI_EMAIL") or "user@example.com"
    if Entrez.email == "user@example.com":
        logger.warning("Using placeholder email. Set NCBI_EMAIL or pass --email.")
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key
    # NCBI limits: 3 requests/s without a key, 10/s with a key.
    return Entrez, Medline, (0.11 if api_key else 0.35)


def _with_retry(fn, tries: int = 4, delay: float = 2.0):
    """Run fn(), retrying on network errors with increasing waits."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # network errors, HTTP 429/5xx, parse errors
            if attempt == tries:
                raise
            logger.warning(f"Attempt {attempt}/{tries} failed ({e}); retrying...")
            time.sleep(delay * attempt)


def build_query(topic_query: str, stratum_query: str, start_year: int) -> str:
    """Combine topic, evidence stratum and quality filters into one PubMed query."""
    return (
        f"({topic_query}) AND {stratum_query} "
        f"AND hasabstract AND English[lang] "
        f'AND ("{start_year}"[dp] : "3000"[dp]) '
        f"NOT Retracted Publication[pt]"
    )


def search_pmids(Entrez, query: str, max_results: int, pause: float) -> List[str]:
    """Return up to max_results PMIDs for a query (PubMed caps esearch at 10,000)."""
    def _do():
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        try:
            return Entrez.read(handle)
        finally:
            handle.close()

    record = _with_retry(_do)
    time.sleep(pause)
    return record.get("IdList", [])


def parse_medline_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one Medline record into our clean schema. Returns None to skip."""
    abstract = clean_text(rec.get("AB", ""))
    if not abstract:
        return None

    pub_types = list(rec.get("PT", []))
    if "Retracted Publication" in pub_types:
        return None

    dp = rec.get("DP", "") or ""
    year = dp[:4] if dp[:4].isdigit() else ""

    doi = ""
    for aid in rec.get("AID", []):
        if aid.endswith("[doi]"):
            doi = aid.replace("[doi]", "").strip()
            break

    return {
        "pmid": rec.get("PMID", ""),
        "title": clean_text(rec.get("TI", "")),
        "abstract": abstract,
        "journal": rec.get("JT", ""),
        "year": year,
        "pub_types": pub_types,                 # list, e.g. ["Randomized Controlled Trial", "Journal Article"]
        "mesh_terms": list(rec.get("MH", [])),
        "doi": doi,
        "source": "PubMed",
    }


def fetch_records(Entrez, Medline, pmids: List[str], pause: float,
                  batch_size: int = 200) -> List[Dict[str, Any]]:
    """Fetch and parse records in batches (large single requests fail)."""
    records: List[Dict[str, Any]] = []
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start:start + batch_size]

        def _do():
            handle = Entrez.efetch(db="pubmed", id=",".join(batch),
                                   rettype="medline", retmode="text")
            try:
                return list(Medline.parse(handle))
            finally:
                handle.close()

        raw = _with_retry(_do)
        time.sleep(pause)
        for rec in raw:
            parsed = parse_medline_record(rec)
            if parsed:
                records.append(parsed)
        logger.info(f"  fetched {min(start + batch_size, len(pmids))}/{len(pmids)}")
    return records


# --------------------------------------------------------------------------
# Corpus building
# --------------------------------------------------------------------------
def collect_corpus(topics: List[str], per_stratum: int, start_year: int,
                   email: Optional[str]) -> List[Dict[str, Any]]:
    """Download per_stratum abstracts for every (topic, evidence stratum) pair."""
    Entrez, Medline, pause = _setup_entrez(email)
    all_records: List[Dict[str, Any]] = []
    seen: set = set()

    for topic in topics:
        for stratum, stratum_query in STRATA.items():
            query = build_query(TOPICS[topic], stratum_query, start_year)
            logger.info(f"[{topic} / {stratum}] searching (max {per_stratum})")
            pmids = [p for p in search_pmids(Entrez, query, per_stratum, pause) if p not in seen]
            if not pmids:
                logger.warning(f"[{topic} / {stratum}] nothing new found")
                continue
            recs = fetch_records(Entrez, Medline, pmids, pause)
            for r in recs:
                seen.add(r["pmid"])
                r["topic"] = topic
                r["stratum"] = stratum
            all_records.extend(recs)
            logger.info(f"[{topic} / {stratum}] kept {len(recs)} abstracts")
    return all_records


def load_existing(filename: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_dataset(records: List[Dict[str, Any]], filename: str = "pubmed_abstracts.json") -> Path:
    """Save to JSON (lists preserved) and CSV (lists joined with '; ')."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DATA_DIR / filename
    out_csv = DATA_DIR / (Path(filename).stem + ".csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(records)
    for col in ("pub_types", "mesh_terms"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: "; ".join(v) if isinstance(v, list) else v)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    logger.info(f"Saved {len(records)} records to {out_json} and {out_csv}")
    return out_json


def report(records: List[Dict[str, Any]]) -> None:
    """Print a summary so you can verify the corpus before building the index."""
    if not records:
        return
    print("\n=== Corpus report ===")
    print(f"Total abstracts: {len(records)}")
    print("\nBy topic / stratum:")
    for (t, s), n in sorted(Counter((r["topic"], r["stratum"]) for r in records).items()):
        print(f"  {t:16s} {s:18s} {n}")
    pt_counts = Counter(pt for r in records for pt in r["pub_types"])
    print("\nTop publication types:")
    for pt, n in pt_counts.most_common(12):
        print(f"  {pt:40s} {n}")
    years = sorted(int(r["year"]) for r in records if r["year"])
    if years:
        print(f"\nYear range: {years[0]} to {years[-1]}")
    print(f"Missing year: {sum(1 for r in records if not r['year'])}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stratified PubMed abstracts with metadata.")
    parser.add_argument("--topics", nargs="+", default=list(TOPICS), choices=list(TOPICS),
                        help="Which topics to download (default: all)")
    parser.add_argument("--per_stratum", type=int, default=20,
                        help="Max abstracts per (topic, evidence level). Use ~500 for the real corpus.")
    parser.add_argument("--start_year", type=int, default=2010, help="Earliest publication year")
    parser.add_argument("--email", type=str, default=None, help="NCBI email (or set NCBI_EMAIL)")
    parser.add_argument("--out", type=str, default="pubmed_abstracts.json", help="Output file name")
    parser.add_argument("--append", action="store_true",
                        help="Merge with the existing file instead of replacing it")
    args = parser.parse_args()

    new_records = collect_corpus(args.topics, args.per_stratum, args.start_year, args.email)

    if args.append:
        existing = load_existing(args.out)
        have = {r["pmid"] for r in existing}
        new_records = existing + [r for r in new_records if r["pmid"] not in have]

    if new_records:
        save_dataset(new_records, args.out)
        report(new_records)
    else:
        logger.warning("No abstracts fetched. Existing data was NOT overwritten.")
