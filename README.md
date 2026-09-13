 <div align="center">

# 🧠 Agentic AI for Automated Literature Analysis and Research-Gap Identification

### A Modular AI-Assisted Framework for Healthcare Explainable Artificial Intelligence (XAI)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# 📖 Overview

Systematic literature reviews require researchers to collect relevant papers, inspect abstracts, identify common themes, compare methodologies, and determine areas that may require further investigation.

This project presents a modular **Agentic AI-assisted literature analysis framework** for organizing research papers and identifying **potential research-gap hypotheses**.

The case study focuses on:

> **Explainable Artificial Intelligence in Healthcare**

The system combines literature retrieval, semantic embeddings, clustering, baseline comparisons, ablation analysis, stability analysis, and deterministic evidence extraction.

> **Important:** This project is a research-assistance system. It identifies potential research directions from literature evidence; it does not independently prove or confirm that a research gap exists.

The term **agentic** refers to modular sequential orchestration of research tasks. The current implementation is not an autonomous LLM-based agent.

---

# ✨ Features

* 📚 Literature retrieval from Semantic Scholar and arXiv
* 🧹 Relevance filtering and duplicate removal
* 📝 Corpus metadata and provenance tracking
* 🧠 Sentence-transformer semantic embeddings
* 🔢 K-Means clustering for thematic organization
* 📊 Silhouette coefficient evaluation
* 📈 Calinski–Harabasz index analysis
* 📉 Davies–Bouldin index analysis
* 📊 TF-IDF baseline comparison
* 🔻 TF-IDF + Truncated SVD baseline comparison
* 🧪 Title-only, abstract-only, and title-plus-abstract ablation
* 🔁 K-Means initialization stability analysis
* 📐 Adjusted Rand Index and NMI analysis
* 🔍 Evidence-based potential research-gap extraction
* 📊 Automated figure generation
* 🧪 Lightweight integrity and data-preservation tests
* 🔐 Environment-variable-based API configuration
* 🛡️ Overwrite protection for important experiment outputs

---

# 🛠️ Technology Stack

| Category             | Technologies                                  |
| -------------------- | --------------------------------------------- |
| Programming Language | Python                                        |
| Literature Retrieval | Semantic Scholar API, arXiv                   |
| Data Processing      | Pandas, NumPy                                 |
| Semantic Embeddings  | Sentence Transformers                         |
| Embedding Model      | `all-MiniLM-L6-v2`                            |
| Clustering Algorithm | K-Means                                       |
| Baseline Methods     | TF-IDF, Truncated SVD                         |
| Evaluation Metrics   | Silhouette, Calinski–Harabasz, Davies–Bouldin |
| Stability Metrics    | ARI, NMI                                      |
| Visualization        | Matplotlib                                    |
| Testing              | Python `unittest`                             |
| Configuration        | Environment Variables                         |
| Dataset              | Research Paper Metadata and Abstracts         |

---

# 📂 Project Structure

```text
agentic-ai-xai-research/
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── corpus.csv
│   └── README.md
│
├── docs/
│   └── methodology.md
│
├── outputs/
│   ├── README.md
│   ├── *.json
│   ├── *.csv
│   ├── embeddings.npy
│   └── figures/
│
├── scripts/
│   ├── run_retrieval.py
│   ├── run_analysis.py
│   ├── run_evaluation.py
│   ├── run_ablation.py
│   ├── run_stability.py
│   └── run_gap_analysis.py
│
├── src/
│   └── agentic_xai/
│       ├── __init__.py
│       ├── config.py
│       ├── utils.py
│       ├── embeddings.py
│       ├── retrieval.py
│       ├── clustering.py
│       ├── evaluation.py
│       ├── ablation.py
│       ├── stability.py
│       └── gap_analysis.py
│
└── tests/
    ├── __init__.py
    ├── test_data_integrity.py
    ├── test_imports.py
    └── test_paths.py
```

---

# ⚙️ Research Workflow

```text
Literature Retrieval
        │
        ▼
Relevance Filtering and Deduplication
        │
        ▼
Corpus Construction
        │
        ▼
Title and Abstract Representation
        │
        ▼
Sentence-Transformer Embeddings
        │
        ▼
K-Means Clustering
        │
        ├── Clustering Metrics
        ├── Baseline Comparison
        ├── Input Ablation
        └── Stability Analysis
        │
        ▼
Evidence Extraction
        │
        ▼
Potential Research-Gap Hypotheses
```

---

# 🔬 Research Questions

### RQ1 — Literature Organization

Can semantic representations and clustering organize healthcare XAI literature into useful thematic groups?

### RQ2 — Method Comparison

How does sentence-transformer-based clustering compare with traditional TF-IDF-based clustering methods?

### RQ3 — Robustness

How sensitive are the clustering results to input representation and K-Means initialization?

### RQ4 — Potential Research-Gap Identification

Can abstract-level evidence be grouped into defensible potential research-gap hypotheses for further investigation?

---

# 📊 Dataset and Experimental Summary

The current case-study corpus contains:

| Experimental Item                      |  Value |
| -------------------------------------- | -----: |
| Final unique papers                    |    187 |
| Papers with usable abstracts           |    175 |
| Embedding dimension                    |    384 |
| Evaluated K values                     |    2–8 |
| Main selected K                        |      2 |
| Best MiniLM silhouette coefficient     | 0.0654 |
| Best title-only silhouette coefficient | 0.0790 |
| Seeds selecting K = 2                  | 7 of 8 |
| Mean pairwise ARI for K = 2            | 0.9183 |

The best silhouette coefficient is low, indicating weak global semantic separation in the evaluated corpus. The stability analysis indicates that the two-cluster partition is relatively consistent across K-Means initializations, but stability does not by itself prove that the clusters are semantically meaningful.

---

# 🧠 Semantic Embedding

The main semantic representation uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The input text is constructed using:

```text
Paper Title + Abstract
```

The embeddings are L2-normalized before clustering.

The main clustering experiment evaluates:

```text
K = 2, 3, 4, 5, 6, 7, 8
```

The selected value of K is determined using the highest silhouette coefficient, with deterministic tie handling.

---

# 📊 Baseline Comparison

The main semantic embedding approach is compared with:

### 1. MiniLM Embeddings + K-Means

Uses contextual sentence embeddings generated from the MiniLM model.

### 2. TF-IDF + K-Means

Uses traditional term-frequency and inverse-document-frequency features.

### 3. TF-IDF + Truncated SVD + K-Means

Uses a lower-dimensional representation derived from TF-IDF features.

The baseline comparison is descriptive. Statistical significance testing and confidence intervals are not currently included.

---

# 🧪 Ablation Study

The ablation study evaluates three text-input configurations:

| Input Configuration | Best Silhouette |
| ------------------- | --------------: |
| Title only          |          0.0790 |
| Abstract only       |          0.0616 |
| Title + Abstract    |          0.0654 |

The title-only configuration achieved the highest silhouette coefficient in this experiment. However, this result does not automatically establish that title-only input is superior for the complete literature-analysis task.

---

# 🔁 Stability Analysis

The stability experiment evaluates K-Means using the following random seeds:

```text
42, 0, 1, 2, 10, 20, 50, 100
```

The analysis reports:

* Selected K for each seed
* Best silhouette coefficient
* Mean silhouette coefficient
* Standard deviation of silhouette coefficient
* Adjusted Rand Index (ARI)
* Normalized Mutual Information (NMI)
* Pairwise cluster agreement

Main stability findings:

```text
K = 2 selected in 7 of 8 seeds
Mean pairwise ARI = 0.9183
```

These results suggest relatively stable clustering initialization behavior, while the low silhouette values indicate that the semantic separation remains weak.

---

# 🔍 Potential Research-Gap Analysis

The gap-analysis stage extracts exact evidence-bearing sentences from available abstracts and groups them into research themes.

The current evidence categories include:

* 📊 Evaluation and reproducibility
* 🏥 Clinical utility and real-world usability
* 🛡️ Reliability, robustness, fairness, and uncertainty

The system produces potential research-gap hypotheses such as:

### Evaluation and Reproducibility

> Reproducible and comparative evaluation of healthcare XAI methods remains insufficiently developed, particularly due to inconsistent evaluation practices and limited standardization of explanation-quality assessment.

### Clinical Utility and Real-World Usability

> The clinical utility and real-world usability of healthcare XAI explanations remain insufficiently evaluated across diverse clinical contexts.

### Reliability and Robustness

> Reliability and robustness of healthcare XAI systems under variations in data, model conditions, and deployment contexts remain insufficiently characterized.

These statements are **potential research-gap hypotheses**, not confirmed research gaps.

---

# 📈 Outputs and Visualizations

The project generates outputs for:

* Literature corpus statistics
* Retrieval provenance
* Cluster assignments
* Embedding matrices
* Clustering metrics
* Baseline comparisons
* Ablation experiments
* Stability experiments
* Potential research-gap candidates
* Evidence passages
* Research figures

Important output files include:

```text
outputs/corpus_manifest.csv
outputs/corpus_stats.json
outputs/cluster_assignments.csv
outputs/clustering_results.json
outputs/embeddings.npy
outputs/baseline_results.json
outputs/baseline_summary.csv
outputs/ablation_results.json
outputs/ablation_summary.csv
outputs/stability_results.json
outputs/stability_summary.csv
outputs/gap_analysis.json
outputs/gap_candidates.csv
outputs/gap_evidence.csv
```

Figures are stored in:

```text
outputs/figures/
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/prannoychandola/agentic-ai-xai-research.git
```

## 2. Navigate to the Project

```bash
cd agentic-ai-xai-research
```

## 3. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Copy the example environment file.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### macOS/Linux

```bash
cp .env.example .env
```

Configure the required environment variables before running the retrieval stage.

Example:

```powershell
$env:SEMANTIC_SCHOLAR_API_KEY="YOUR_NEW_API_KEY"
```

> Never commit `.env` files or real API keys to GitHub. Use environment variables for credentials.

---

# ▶️ Running the Project

Run the commands from the project root.

## 1. Literature Retrieval

```bash
python scripts/run_retrieval.py
```

To intentionally regenerate retrieval outputs:

```bash
python scripts/run_retrieval.py --overwrite
```

## 2. Semantic Analysis and Clustering

```bash
python scripts/run_analysis.py
```

To intentionally regenerate clustering outputs:

```bash
python scripts/run_analysis.py --overwrite
```

## 3. Baseline Evaluation

```bash
python scripts/run_evaluation.py
```

To intentionally regenerate baseline outputs:

```bash
python scripts/run_evaluation.py --overwrite
```

## 4. Ablation Study

```bash
python scripts/run_ablation.py
```

To intentionally regenerate ablation outputs:

```bash
python scripts/run_ablation.py --overwrite
```

## 5. Stability Analysis

```bash
python scripts/run_stability.py
```

## 6. Potential Research-Gap Analysis

```bash
python scripts/run_gap_analysis.py
```

---

# ⚠️ Overwrite Protection

The analysis, evaluation, and ablation stages include overwrite protection.

Do not use:

```bash
--overwrite
```

unless you intentionally want to regenerate the corresponding outputs.

Regenerating outputs may:

* Replace existing scientific artifacts
* Require model downloads
* Require additional computation
* Produce different results under changed dependencies
* Make comparison with the reported experiments more difficult

---

# 🧪 Testing

Run the lightweight integrity tests:

```bash
python -m unittest discover -s tests -v
```

The tests verify:

* Package imports
* Project-relative paths
* CSV and JSON readability
* Required output availability
* Data integrity
* Embedding metadata
* Preservation of important artifacts

The tests are designed to avoid:

* API calls
* Model downloads
* Expensive experiments
* Output regeneration

---

# 📌 Limitations

### Corpus Coverage

The case study uses a retrieved literature corpus and is not an exhaustive systematic review of all healthcare XAI publications.

### Abstract-Level Analysis

Most analysis is based on titles and abstracts. Full-text methods, datasets, limitations, and experimental details may not be captured.

### Single Embedding Model

The main experiments use one sentence-transformer model. Other embedding models may produce different representations and clustering behavior.

### Weak Semantic Separation

The best silhouette coefficient is low, indicating weak global separation between clusters.

### No Labelled Ground Truth

The task is unsupervised clustering. Therefore, F1-score, precision, recall, and ROC-AUC are not reported because labelled ground-truth classes are unavailable.

### No Statistical Significance Testing

Baseline and ablation comparisons are descriptive. Statistical significance testing and confidence intervals are not currently included.

### No Human Expert Validation

The current implementation does not include formal clinician or domain-expert validation. Potential research-gap candidates should be reviewed by qualified researchers before being treated as established findings.

### Potential Research-Gap Interpretation

Evidence-support counts are not proof that a research gap exists. The extracted candidates should be treated as hypotheses for further investigation.

---

# 🔮 Future Improvements

* Compare multiple embedding models
* Evaluate additional clustering algorithms
* Use full-text literature instead of abstracts only
* Add topic modelling and keyword-based analysis
* Include expert-labelled evaluation subsets
* Add statistical significance testing
* Improve domain-specific evidence extraction
* Add clinical-context and dataset metadata
* Expand the healthcare XAI literature corpus
* Investigate explanation reliability, fairness, uncertainty, and robustness
* Develop an interactive research-analysis dashboard
* Add automated experiment tracking and versioning

---

# 📜 Data and Licensing Notice

This repository contains literature metadata, a research corpus, abstract-derived evidence, embeddings, and analytical outputs.

Before redistributing or using the data publicly:

* Review the terms of the original data providers.
* Check whether abstracts and extracted passages may be redistributed.
* Respect publisher and repository licensing requirements.
* Use original source records for authoritative bibliographic information.
* Do not treat the included corpus as a complete literature review.

The code is provided under the license included in this repository. Data and generated artifacts may be subject to separate rights and restrictions.

---

# 👨‍💻 Author

## Prannoy Chandola

**Aspiring AI Engineer and AI Researcher**

Interested in:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Natural Language Processing
* Generative AI
* Explainable AI
* Agentic AI
* AI Research

### GitHub

https://github.com/prannoychandola

### LinkedIn

https://www.linkedin.com/in/prannoy-chandola-8a53b5366/

---

# 📜 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

---

<div align="center">

⭐ If you found this project useful, consider giving it a star on GitHub.

</div>
