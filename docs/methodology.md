# Methodology

The project implements a small sequential research-assistance pipeline:

```text
Retrieval -> relevance screening -> deduplication -> semantic embedding -> KMeans clustering -> baseline comparison -> ablation study -> stability analysis -> evidence extraction -> potential research-gap identification
```

## Retrieval and Corpus Construction

`scripts/run_retrieval.py` invokes the package retrieval stage, which queries Semantic Scholar and arXiv independently, handles pagination, rate limits, retries, progressive checkpoints, and deterministic relevance screening. Relevance requires signals for both XAI/interpretability and healthcare/medical application. Records are deduplicated using DOI, source paper ID, and normalized title. The completed corpus contains 187 unique papers; 175 have usable abstracts.

## Semantic Representation and Clustering

`scripts/run_analysis.py` invokes the package analysis stage, which uses `sentence-transformers/all-MiniLM-L6-v2` with title plus abstract text. Embeddings have 384 dimensions and are L2-normalized before clustering. K-Means is evaluated for K=2..8 with `random_state=42` and `n_init=20`. K is selected by the highest silhouette score, with the implementation's documented tie behavior.

## Baseline Comparison

`scripts/run_evaluation.py` invokes the package evaluation stage, which compares the saved MiniLM representation with:

- TF-IDF plus K-Means using lowercase text, English stop-word removal, up to 10,000 features, and unigram/bigram features.
- TF-IDF plus Truncated SVD, L2 normalization, and K-Means.

Both baselines use K=2..8, `random_state=42`, and `n_init=20`. Silhouette, Calinski-Harabasz, Davies-Bouldin, and inertia are reported. Inertia is not directly comparable across different feature spaces.

## Ablation Study

`scripts/run_ablation.py` invokes the package ablation stage, which separately encodes the same 175 papers under three fixed conditions:

1. Title only.
2. Abstract only.
3. Title plus abstract.

Each condition uses the same MiniLM model, L2 normalization, K=2..8, `random_state=42`, and `n_init=20`.

## Stability Analysis

`scripts/run_stability.py` invokes the package stability stage, which reuses the saved title-plus-abstract embedding matrix and tests seeds `42, 0, 1, 2, 10, 20, 50, 100` for K=2..8 with `n_init=20`. K=2 assignment consistency is compared using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI), because cluster labels are arbitrary. Stability describes robustness to initialization, not semantic validity.

## Potential Research-Gap Identification

`scripts/run_gap_analysis.py` invokes the package gap-analysis stage, which performs deterministic evidence extraction from corpus title/abstract text. It searches for limitation, future-work, generalization/validation, clinical-utility, robustness/fairness, uncertainty, evaluation, and reproducibility language, then groups passages into transparent themes.

A candidate requires at least three supporting papers, or at least two explicit limitation/future-work passages. Confidence labels are evidence-based `HIGH`, `MEDIUM`, or `LOW` descriptors, not probabilities and not proof. Exact evidence passages are retained with paper IDs. Cluster membership, citation counts, and low frequency are contextual information only and are not treated as proof of a gap.

The final output is potential/candidate research-gap identification. It does not claim proven, confirmed, definitive, or ground-truth research gaps. No LLM, expert-validation process, or external API is used in this stage.
