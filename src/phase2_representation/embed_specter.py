"""
Semantic Embedding Generation using SPECTER2 and PCA/SVD Dimensionality Reduction.
Encodes scientific papers into dense semantic representations and compresses them
for efficient indexing and clustering.
"""

import os
import pickle
import logging
from importlib import metadata
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
from packaging.requirements import Requirement
from packaging.version import Version
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class SpecterEmbedder:
    """
    Generates paper embeddings using SPECTER2 (base model, plus the proximity
    adapter when the installed `adapters` and `transformers` versions are compatible).
    Input format standard: Title + ' [SEP] ' + Abstract.
    """

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        adapter_name: Optional[str] = "allenai/specter2",
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto"
    ):
        """
        model_name: SPECTER2 base model (tokenizer + transformer weights).
        adapter_name: task adapter on top of the base. "allenai/specter2" is the
            proximity adapter, recommended for similarity, kNN search and
            clustering. None uses the base model alone.
        """
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.tokenizer = None
        self.model = None
        self._init_model()

    def _init_model(self):
        """Load the SPECTER2 tokenizer and model. Errors are raised, not hidden."""
        if self.device == "auto":
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.torch_device = self.device

        logger.info(f"Loading SPECTER2 tokenizer and model '{self.model_name}' on {self.torch_device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self._load_model_with_adapter().to(self.torch_device)
        self.model.eval()
        logger.info("SPECTER2 successfully loaded.")

    @staticmethod
    def _adapters_compatible() -> bool:
        """
        True if `adapters` is installed and the installed `transformers` version
        satisfies the range that `adapters` declares (e.g. transformers~=4.57.6).
        `adapters` patches transformers internals, so a mismatch can break it.
        """
        try:
            adapters_version = metadata.version("adapters")
        except metadata.PackageNotFoundError:
            logger.warning("`adapters` not installed; using SPECTER2 base model without adapter.")
            return False

        transformers_version = Version(metadata.version("transformers"))
        for req_str in metadata.requires("adapters") or []:
            req = Requirement(req_str)
            if req.name == "transformers" and transformers_version not in req.specifier:
                logger.warning(
                    f"adapters {adapters_version} requires transformers{req.specifier}, "
                    f"but transformers {transformers_version} is installed; "
                    "using SPECTER2 base model without adapter."
                )
                return False
        return True

    def _load_model_with_adapter(self):
        """
        Load the base model and activate the task adapter when possible.
        Falls back to the base model alone if no adapter is configured or the
        installed `adapters`/`transformers` versions are incompatible.
        """
        if not self.adapter_name or not self._adapters_compatible():
            return AutoModel.from_pretrained(self.model_name)

        from adapters import AutoAdapterModel

        model = AutoAdapterModel.from_pretrained(self.model_name)
        model.load_adapter(self.adapter_name, source="hf", load_as="specter2_task", set_active=True)
        logger.info(f"Activated SPECTER2 adapter '{self.adapter_name}'.")
        return model

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
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        return self._embed_with_transformers(texts)

    def _embed_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Run batch inference using PyTorch and HuggingFace SPECTER2."""

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
                # CLS token (position 0): SPECTER is trained to put the paper-level
                # representation here, so no mean pooling over tokens
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        # L2 Normalize for cosine similarity computation
        return normalize(embeddings, norm="l2")


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
