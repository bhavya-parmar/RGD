# Research Gap Discovery (RGD)

> **Bridging disconnected academic communities by pairing semantic similarity with citation graphs to reveal high-potential research frontiers.**

**Authors:** Vedant Acharya (23110010) & Bhavya Parmar (23110059)  
**Course:** CS328 - Introduction to Data Science, IIT Gandhinagar

---

## 📌 Motivation

Researchers entering an unfamiliar or rapidly evolving field struggle to efficiently sift through thousands of papers, separate broad surveys from seminal breakthroughs, and discover genuine research gaps. Relying on raw citation counts introduces strong historical bias toward older works, while keyword searches fail to capture cross-disciplinary overlap.

This leaves researchers blind to **isolated academic silos**, where distinct communities solve the same fundamental problem without citing each other, making pinpointing unaddressed research frontiers a slow, manual process.

To address these limitations, **RGD** combines dense semantic similarity with citation-network topology to map the research landscape beyond conventional keyword and citation-based search. By identifying semantically related but weakly connected research communities and validating these patterns statistically, our pipeline isolates candidate research frontiers and overlooked cross-disciplinary opportunities.

---

## 🏗️ System Architecture & Workflow

The pipeline is organized into four sequential phases:

```
[Phase 1: Retrieval & Prep]
Research Query ──> OpenAlex API ──> Citation Expansion ──> MinHash + LSH Deduplication ──> Research Corpus
                                                                                               │
[Phase 2: Dual-View Representation]                                                          │
      ┌────────────────────────────────────────────────────────────────────────────────────────┘
      ├──> SPECTER2 Embeddings ──> PCA/SVD Reduction ──> FAISS Semantic Index
      └──> Directed Citation Graph ──> PageRank (Seminal Influence) + HITS (Hubs & Authorities)
                                                                                               │
[Phase 3: Community & Frontier Detection]                                                     │
      ├──> k-means++ (Semantic Communities) <──┐                                              │
      │                                        ├──> Cross-Community Discrepancy Scoring ──────┤
      └──> Spectral / Louvain (Citation Comm.) <┘   (High Semantic Sim + Low Citation Density)│
                                                                                               │
[Phase 4: Validation & Reporting]                                                             │
      └──> Bootstrap Resampling (Stability) ──> Permutation Hypothesis Testing ──> Landscape & Gap Report
```

1. **Phase 1: Literature Retrieval & Preparation**
   - Queries OpenAlex REST API for seed works.
   - Expands candidate pool via citation links (`referenced_works`) and concept overlaps.
   - Applies MinHash signatures and Locality Sensitive Hashing (LSH) to eliminate near-duplicate preprints and republications.

2. **Phase 2: Semantic & Citation Representation**
   - Encodes paper titles and abstracts into 768-dimensional dense vectors with **SPECTER2**.
   - Applies PCA/SVD to compress embeddings to 64 dimensions while preserving dominant variance.
   - Builds a high-speed **FAISS** index for sub-millisecond nearest neighbor search.
   - Constructs a directed citation network; calculates **PageRank** to highlight seminal works and **HITS** Hub/Authority scores to isolate surveys vs core innovations.

3. **Phase 3: Community & Frontier Detection**
   - Applies **k-means++** clustering on reduced semantic vectors to define topical boundaries.
   - Applies **Spectral Clustering / Louvain Modularity** on the citation graph to isolate citation silos.
   - Calculates cross-community **Frontier Gap Scores**: $\text{Score} = \text{Sim}_{\text{semantic}} \times (1 - \text{Density}_{\text{citation}})$.
   - Surfaces isolated paper pairs working on identical problems without reciprocal citations.

4. **Phase 4: Validation & Analytical Reporting**
   - Validates pattern stability via non-parametric **Bootstrap Resampling**.
   - Tests null hypothesis $H_0$ using **Permutation Testing** against rewired random graph models.
   - Outputs interactive network visualizations, 2D semantic landscape maps, discrepancy heatmaps, and a standalone HTML gap discovery report.

---

## 📂 Repository Structure

```
Project/RGD/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Python package dependencies
├── .env                           # Environment variables & OpenAlex API key
├── .env.example                   # Template environment configuration
├── .gitignore                     # Git ignore rules for data, models, and environments
├── config.yaml                    # Hyperparameters (FAISS dims, LSH bands, clustering k)
├── data/                          # Data storage (git-ignored)
│   ├── raw/                       # Initial OpenAlex JSON/CSV outputs
│   └── processed/                 # Stored corpus after duplicate removal and expansion
├── models/                        # Saved artifacts (git-ignored)
│   ├── embeddings/                # Serialized SPECTER2 embeddings and PCA models
│   └── faiss_indices/             # Compiled FAISS index files
├── notebooks/                     # Exploratory notebooks for testing individual components
└── src/
    ├── __init__.py
    ├── phase1_acquisition/        # Literature Retrieval & Acquisition
    │   ├── __init__.py
    │   ├── openalex_fetch.py      # Initial query retrieval and citation-link expansion
    │   └── deduplication.py       # MinHash and LSH duplicate removal
    ├── phase2_representation/     # Semantic & Citation Representation
    │   ├── __init__.py
    │   ├── embed_specter.py       # SPECTER2 generation and PCA dimensionality reduction
    │   ├── faiss_index.py         # Construction of the semantic similarity index
    │   └── citation_graph.py      # Citation graph construction, PageRank, and HITS
    ├── phase3_detection/          # Community & Frontier Detection
    │   ├── __init__.py
    │   ├── semantic_cluster.py    # Detection of semantic communities
    │   ├── graph_cluster.py       # Detection of citation communities
    │   └── frontier_scoring.py    # Community comparison to isolate candidate frontiers
    ├── phase4_validation/         # Validation & Analysis
    │   ├── __init__.py
    │   ├── hypothesis_test.py     # Bootstrap resampling and statistical pattern validation
    │   └── visualization.py       # Network visualization and analytical dashboard logic
    └── pipeline.py                # Main orchestrator to run phases sequentially
```

---

## 🚀 Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure your OpenAlex credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
OPENALEX_API_KEY=your_openalex_api_key
OPENALEX_EMAIL=your_email@example.com
```

### 3. Run the Pipeline

Execute the full end-to-end pipeline with a live search query:

```bash
python src/pipeline.py --query "physics informed neural networks" --max-seeds 40
```

#### Offline / Mock Mode (Instant Zero-Dependency Verification)

To test the entire pipeline locally without downloading SPECTER2 model weights or hitting external APIs:

```bash
python src/pipeline.py --query "graph neural networks" --max-seeds 30 --mock
```

---

## ⚙️ Configuration Reference (`config.yaml`)

| Parameter Section | Key Parameters | Description |
| :--- | :--- | :--- |
| **`acquisition`** | `default_query`, `max_seed_papers`, `citation_expansion_depth` | Governs OpenAlex seed retrieval and expansion scope |
| **`deduplication`** | `lsh_threshold`, `num_perm`, `shingle_size` | MinHash Jaccard threshold and permutation settings |
| **`representation`** | `specter.model_name`, `pca.n_components`, `faiss.metric` | SPECTER2 model, PCA target dimensions (default 64), FAISS distance |
| **`detection`** | `semantic_clustering.n_clusters`, `graph_clustering.algorithm` | k-means++ and Spectral/Louvain community parameters |
| **`detection`** | `frontier_scoring.semantic_sim_threshold`, `citation_connectivity_threshold` | Cutoffs for isolating high-potential research gaps |
| **`validation`** | `bootstrap.n_iterations`, `hypothesis_test.alpha` | Non-parametric stability iterations and p-value significance |

---

## 📊 Outputs & Reports

Upon completion, all analytical artifacts are organized in `reports/` and `data/`:
- `reports/frontier_report.html`: Interactive summary dashboard detailing isolated research gaps.
- `reports/citation_network.html`: Interactive citation network with community nodes.
- `reports/validated_frontiers.csv`: Ranked candidate frontiers with bootstrap confidence intervals and empirical p-values.
- `reports/semantic_landscape.png`: 2D PCA projection of semantic communities.
- `reports/frontier_discrepancy_matrix.png`: Heatmap of cross-community semantic similarity vs citation density.
