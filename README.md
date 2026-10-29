# VeriMed-RAG

Biomedical Retrieval-Augmented Generation (RAG) benchmarking system for PubMed literature verification and query answering.

## Project Structure

```text
verimed-rag/
  data/            # raw PubMed abstracts, questions
  index/           # vector index files (FAISS, metadata)
  src/
    ingest.py      # download + clean PubMed data
    chunk_embed.py # chunking + embeddings + vector index
    retrieve.py    # search function (dense, BM25, hybrid)
    baseline.py    # M1 and M2 baselines
    evaluate.py    # IR and QA evaluation metrics
  results/         # benchmark and evaluation outputs
  requirements.txt # package dependencies
```

## Quick Start

### 1. Activate Environment
In Windows PowerShell:
```powershell
.\sharmila\Scripts\Activate.ps1
```
Or in Command Prompt (cmd):
```cmd
sharmila\Scripts\activate.bat
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Pipeline Execution

1. **Ingest Data**:
   ```powershell
   python src/ingest.py --query "diabetes metformin" --max_results 20
   ```

2. **Chunk & Embed (Index Construction)**:
   ```powershell
   python src/chunk_embed.py
   ```

3. **Retrieve & Test Query**:
   ```powershell
   python src/retrieve.py --query "What is the primary mechanism of metformin?"
   ```

4. **Run Baselines (M1 & M2)**:
   ```powershell
   python src/baseline.py
   ```

5. **Evaluate Metrics**:
   ```powershell
   python src/evaluate.py
   ```
