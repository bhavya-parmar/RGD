"""
Semantic Embedding Generation using SPECTER2 and PCA/SVD Dimensionality Reduction.
Encodes scientific papers into dense semantic representations and compresses them
for efficient indexing and clustering.
"""

import os
import pickle
import logging
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class SpecterEmbedder:
    """
    Generates paper embeddings using SPECTER2 (or fallback text encoder).
    Input format standard: Title + ' [SEP] ' + Abstract.
    """

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.tokenizer = None
        self.model = None
        self._init_model()

    def _init_model(self):
        """Lazy load HuggingFace transformers or set mock fallback."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            if self.device == "auto":
                self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.torch_device = self.device

            logger.info(f"Loading SPECTER2 tokenizer and model '{self.model_name}' on {self.torch_device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.torch_device)
            self.model.eval()
            self.has_transformers = True
            logger.info("SPECTER2 successfully loaded.")
        except Exception as e:
            logger.warning(
                f"Could not load HuggingFace model '{self.model_name}' ({e}). "
                "Switching to fallback deterministic embedding generator."
            )
            self.has_transformers = False

    def embed_papers(self, papers_df: pd.DataFrame) -> np.ndarray:
        """
        Compute dense embeddings for a DataFrame containing 'title' and 'abstract'.
        Returns: np.ndarray of shape (N, embedding_dim).
        """
        texts = []
        for _, row in papers_df.iterrows():
            title = str(row.get("title", "")).strip()
            abstract = str(row.get("abstract", "")).strip()
            # Canonical SPECTER format
            combined = f"{title} [SEP] {abstract}" if abstract else title
            texts.append(combined)

        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        if self.has_transformers:
            return self._embed_with_transformers(texts)
        else:
            return self._embed_fallback(texts)

    def _embed_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Run batch inference using PyTorch and HuggingFace SPECTER2."""
        import torch

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.torch_device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Take CLS token representation or mean pool
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        # L2 Normalize for cosine similarity computation
        return normalize(embeddings, norm="l2")

    def _embed_fallback(self, texts: List[str], dim: int = 768) -> np.ndarray:
        """
        Deterministic TF-IDF / hashing fallback when GPU/HuggingFace is not configured.
        Ensures pipeline testing remains unblocked.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info(f"Generating fallback TF-IDF + SVD embeddings for {len(texts)} documents...")
        vectorizer = TfidfVectorizer(max_features=2048, stop_words="english")
        tfidf = vectorizer.fit_transform(texts).toarray()

        if tfidf.shape[1] < dim:
            # Pad with reproducible pseudo-random features
            np.random.seed(42)
            pad = np.random.randn(len(texts), dim - tfidf.shape[1]) * 0.01
            dense = np.hstack([tfidf, pad]).astype(np.float32)
        else:
            svd = TruncatedSVD(n_components=dim, random_state=42)
            dense = svd.fit_transform(tfidf).astype(np.float32)

        return normalize(dense, norm="l2")


class DimensionalityReducer:
    """
    Reduces SPECTER2 embeddings (e.g. 768 -> 64) using PCA or TruncatedSVD
    to accelerate FAISS indexing and community clustering while retaining variance.
    """

    def __init__(self, n_components: int = 64, method: str = "pca", random_state: int = 42):
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.model = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform high-dimensional embeddings."""
        n_samples, n_features = embeddings.shape
        actual_components = min(self.n_components, n_samples - 1, n_features)

        if actual_components < 2:
            logger.warning("Corpus size too small for PCA reduction; preserving raw embeddings.")
            return embeddings

        if self.method.lower() == "svd":
            self.model = TruncatedSVD(n_components=actual_components, random_state=self.random_state)
        else:
            self.model = PCA(n_components=actual_components, random_state=self.random_state)

        reduced = self.model.fit_transform(embeddings)
        explained = getattr(self.model, "explained_variance_ratio_", [0])
        logger.info(
            f"Reduced embeddings from {n_features} to {actual_components} dimensions "
            f"(Cumulative Explained Variance: {np.sum(explained):.2%})."
        )
        return normalize(reduced, norm="l2").astype(np.float32)

    def save(self, filepath: str):
        """Save fitted model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str):
        """Load fitted model from disk."""
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sample_df = pd.DataFrame([
        {"id": "1", "title": "Deep Learning for Physics PDEs", "abstract": "Solving partial differential equations."},
        {"id": "2", "title": "Scientific Machine Learning", "abstract": "Physics-informed algorithms."},
        {"id": "3", "title": "Graph Neural Networks for Molecules", "abstract": "Molecular property prediction."}
    ])
    embedder = SpecterEmbedder()
    emb = embedder.embed_papers(sample_df)
    print("Embedding Shape:", emb.shape)

    reducer = DimensionalityReducer(n_components=2)
    reduced_emb = reducer.fit_transform(emb)
    print("Reduced Shape:", reduced_emb.shape)
