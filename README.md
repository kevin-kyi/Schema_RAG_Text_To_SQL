# Schema RAG + TAPAS Pipeline (WTQ Baseline Reproduction)

This repository implements a schema-aware retrieval pipeline combined with the TAPAS table-question-answering model.  
The system retrieves relevant tables from the WikiTableQuestions (WTQ) dataset using dense schema embeddings and answers natural-language questions using TAPAS.

The WTQ-based system is the **main** implementation.  
A smaller legacy Spider-based version is included at the end for reference.

---

## Main Pipeline (WikiTableQuestions WTQ)



### Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Main WTQ Implementation Notebook

Open the notebook:

```
notebooks/wtq_final_pipeline.ipynb
```

Run all cells from top to bottom.

The notebook will:

- Load WTQ dataset  
- Load `schema.json`  
- Build dense schema embeddings  
- Retrieve tables for 200 sampled questions  
- Run TAPAS on the retrieved table  
- Compute:
  - Recall@1 / Recall@10 / Recall@50  
  - EM  
  - F1  
- Print detailed retrieval and reader error cases

An example `schema.json` is included and ready to use.

---

## Full WTQ Reproduction Workflow

To reproduce everything from scratch:

### **1. Download WTQ**

```bash
python src/wtq_pipeline/download_wtq.py
```

Downloads WTQ (train/validation/test) to:

```
data/wtq_hf/
```

### **2. Build a New schema.json**

```bash
python src/wtq_pipeline/build_schema.py
```

Generates a new `schema.json` used to build embeddings:

### **3. Run the Notebook**

Open:

```
notebooks/wtq_final_pipeline.ipynb
```

Run all cells.

---

## Running in Google Colab

The notebook was originally developed and tested in Google Colab with GPU runtime.

To run in Colab:

1. Open Colab notebook: https://colab.research.google.com/drive/1mQBQevVgebh-ZXrZKUKL-iUbDft8BzTA?usp=sharing  
2. Upload `schema.json`  
3. Install requirements inside Colab  
4. Run all cells  

The pipeline will produce the same evaluation results.

---

## Experimental Spider Dataset Pipeline

A previous version of the project implemented the same schema-embedding + retrieval pipeline for the Spider 1.0 dataset.

To run the old Spider version:

```bash
python src/full_pipeline.py     --query "List colleges with the most students."     --spider-dir data/spider_raw/
```

### Optional Flags

```
--data-dir          # directory for CSV tables
--spider-dir        # root of Spider SQLite databases
--embedding-model   # schema embedding model
--tapas-model       # TAPAS QA checkpoint
--max-table-rows    # row limit for TAPAS input
```


---
