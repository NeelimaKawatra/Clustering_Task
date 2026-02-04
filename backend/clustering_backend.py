import time
from typing import Dict, Any, List, Tuple
import numpy as np
import streamlit as st

from .activity_logger import ActivityLogger


# --------- CACHE EMBEDDER (loads once) ----------
@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


class ClusteringConfig:
    def __init__(self):
        self.default_params = {
            "min_topic_size": 5,
            "min_samples": 3,
            "n_neighbors": 15,
            "n_components": 5,
            "metric": "cosine",
            "random_state": 42,
        }

    def get_optimal_parameters(self, text_count: int) -> Dict[str, Any]:
        p = self.default_params.copy()
        if text_count < 50:
            p["min_topic_size"] = 3; p["min_samples"] = 2; p["n_neighbors"] = 10
        elif text_count < 100:
            p["min_topic_size"] = 5; p["min_samples"] = 3; p["n_neighbors"] = 15
        elif text_count < 200:
            p["min_topic_size"] = 8; p["min_samples"] = 4; p["n_neighbors"] = 15
        else:
            p["min_topic_size"] = 10; p["min_samples"] = 5; p["n_neighbors"] = 20
        return p

    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        for k in ["min_topic_size", "n_neighbors", "n_components"]:
            if k not in params:
                return False, f"Missing required parameter: {k}"
        return True, "Parameters are valid"


class OptimizedClusteringModel:
    """
    Fast pipeline for 1–2 word noisy text:
    BGE-small + Char ngrams → PCA → UMAP → HDBSCAN

    Reporting improvements:
    - Confidence: distance-to-centroid (0..1) in reduced space
    - Top keywords: word-level TF-IDF (human readable), NOT char fragments
    """
    def __init__(self):
        self.embedder = None
        self.char_vectorizer = None
        self.pca = None
        self.reducer = None
        self.clusterer = None
        self.is_setup = False

    def setup_model(self, params: Dict[str, Any]) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import PCA
            import umap
            import hdbscan

            self.embedder = get_embedder()

            # Char n-grams for robust clustering on typos/truncations
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=200
            )

            # PCA before UMAP (speed)
            self.pca = PCA(n_components=100)

            self.reducer = umap.UMAP(
                n_neighbors=params.get("n_neighbors", 15),
                n_components=params.get("n_components", 5),
                metric="cosine",
                random_state=params.get("random_state", 42)
            )

            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=params.get("min_topic_size", 5),
                min_samples=params.get("min_samples", 3),
                metric="euclidean",
                prediction_data=True
            )

            self.is_setup = True
            return True
        except Exception as e:
            print(e)
            return False

    def fit_transform(self, texts: List[str]):
        if not self.is_setup:
            raise ValueError("Model not setup.")

        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import euclidean_distances

        # -------------------------
        # 1) Embeddings for clustering
        # -------------------------
        emb_semantic = self.embedder.encode(texts, normalize_embeddings=True)
        emb_char = self.char_vectorizer.fit_transform(texts).toarray()
        combined = np.hstack([emb_semantic, emb_char])

        combined = self.pca.fit_transform(combined)
        reduced = self.reducer.fit_transform(combined)

        labels = self.clusterer.fit_predict(reduced)
        labels_arr = np.asarray(labels)

        # -------------------------
        # 2) Confidence (0..1): distance-to-centroid per cluster
        # -------------------------
        conf = np.zeros(len(texts), dtype=float)

        for k in set(labels_arr):
            if k == -1:
                continue
            idx = np.where(labels_arr == k)[0]
            if idx.size == 0:
                continue
            centroid = reduced[idx].mean(axis=0, keepdims=True)
            d = euclidean_distances(reduced[idx], centroid).ravel()
            max_d = d.max() if d.size else 1.0
            conf[idx] = 1.0 - (d / (max_d + 1e-9))

        # outliers
        conf[labels_arr == -1] = 0.0

        # -------------------------
        # 3) Human-readable keywords (word TF-IDF) for display only
        # -------------------------
        # Clean ONLY for keyword extraction (keep clustering unchanged)
        clean_texts = [re.sub(r"[^a-zA-Z\s]", " ", t).lower() for t in texts]

        word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
            max_features=2000,
            min_df=1
        )
        X_words = word_vectorizer.fit_transform(clean_texts)
        word_features = word_vectorizer.get_feature_names_out()

        topic_keywords: Dict[int, List[str]] = {}
        for k in set(labels_arr):
            if k == -1:
                continue
            mask = labels_arr == k
            if not np.any(mask):
                topic_keywords[k] = []
                continue
            mean_vec = X_words[mask].mean(axis=0)
            mean_arr = np.asarray(mean_vec).ravel()
            top_idx = mean_arr.argsort()[-5:][::-1]
            topic_keywords[k] = [word_features[i] for i in top_idx if mean_arr[i] > 0]

        metadata = {
            "model_type": "Fast BGE + Char + HDBSCAN",
            "topic_keywords": topic_keywords,
            # Optional: keep the original HDBSCAN membership probs for debugging
            "hdbscan_membership_probabilities": getattr(self.clusterer, "probabilities_", None).tolist()
            if getattr(self.clusterer, "probabilities_", None) is not None else None,
        }

        return labels_arr.tolist(), conf.tolist(), metadata


class ClusteringBackend:
    def __init__(self, logger: ActivityLogger):
        self.logger = logger
        self.config = ClusteringConfig()
        self.model = OptimizedClusteringModel()

    def get_clustering_parameters(self, text_count: int, session_id: str) -> Dict[str, Any]:
        params = self.config.get_optimal_parameters(text_count)
        return params

    def run_clustering(self, texts: List[str], params: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        start = time.time()

        # (optional) validate params like before
        ok, msg = self.config.validate_parameters(params)
        if not ok:
            raise ValueError(msg)

        # setup + run
        setup_t0 = time.time()
        setup_ok = self.model.setup_model(params)
        setup_time = time.time() - setup_t0
        if not setup_ok:
            raise ValueError("Failed to setup clustering model")

        t1 = time.time()
        topics, probs, meta = self.model.fit_transform(texts)
        clust_time = time.time() - t1

        # stats (same as your old schema)
        unique = set(topics)
        outliers = sum(1 for t in topics if t == -1)
        clustered = len(topics) - outliers
        success_rate = (clustered / len(topics)) * 100 if topics else 0.0
        n_clusters = len([t for t in unique if t != -1])

        high = sum(1 for p in probs if p >= 0.7)
        med  = sum(1 for p in probs if 0.3 <= p < 0.7)
        low  = sum(1 for p in probs if p < 0.3)

        total = time.time() - start

        # persist trained components
        st.session_state["trained_model_pack"] = {
            "embedder": self.model.embedder,
            "char_vectorizer": self.model.char_vectorizer,
            "umap": self.model.reducer,
            "hdbscan": self.model.clusterer,
        }

        return {
            "success": True,
            "topics": topics,
            "probabilities": probs,          # <-- this is your distance-based confidence now
            "predictions": topics,
            "texts": texts,
            "metadata": meta,
            "statistics": {
                "n_clusters": n_clusters,
                "outliers": outliers,
                "clustered": clustered,
                "success_rate": success_rate,
                "total_texts": len(texts),
            },
            "confidence_analysis": {
                "high_confidence": high,
                "medium_confidence": med,
                "low_confidence": low,
                "avg_confidence": (sum(probs) / len(probs)) if probs else 0.0
            },
            "performance": {
                "total_time": total,
                "setup_time": setup_time,
                "clustering_time": clust_time
            },
            "parameters_used": params
        }

    def get_cluster_details(self, results: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        topics = results["topics"]
        texts = results["texts"]
        probs = results["probabilities"]  # distance-based confidence
        topic_keywords = results["metadata"]["topic_keywords"]

        details = {}
        for cid in sorted(set(topics)):
            idxs = [i for i, t in enumerate(topics) if t == cid]
            details[cid] = {
                "size": len(idxs),
                "keywords": topic_keywords.get(cid, []),
                "texts": [texts[i] for i in idxs],
                "confidences": [probs[i] for i in idxs]
            }

        return details
