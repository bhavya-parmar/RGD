# Exploratory & Component Notebooks

This folder is intended for interactive exploration, prototyping, and qualitative verification of individual pipeline phases:

- `01_exploration_openalex.ipynb`: Inspect raw OpenAlex query results, metadata fields, inverted abstract reconstruction, and citation graph density.
- `02_deduplication_lsh.ipynb`: Tune MinHash permutations and LSH Jaccard similarity thresholds.
- `03_specter_faiss.ipynb`: Test SPECTER2 embedding latency, PCA explained variance ratio, and nearest-neighbor semantic search.
- `04_community_and_frontiers.ipynb`: Visualize graph vs semantic clustering overlap and inspect detected candidate frontiers.
- `05_statistical_validation.ipynb`: Run bootstrap stability analysis and distribution testing.
