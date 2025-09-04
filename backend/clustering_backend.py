# backend/clustering_backend.py
import time
from typing import Dict, Any, List, Tuple
import numpy as np

from .activity_logger import ActivityLogger

class ClusteringConfig:
    def __init__(self):
        self.default_params = {
            "min_topic_size": 5,   # used as K for KMeans in this pipeline
            "min_samples": 3,      # reserved for density-based methods
            "n_neighbors": 15,     # reserved for UMAP
            "n_components": 5,
            "metric": "cosine",
            "random_state": 42,
            "embedding_model": "all-MiniLM-L6-v2"
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
        if params["min_topic_size"] < 2: return False, "min_topic_size must be at least 2"
        if params["n_neighbors"] < 5:    return False, "n_neighbors must be at least 5"
        if params["n_components"] < 2:   return False, "n_components must be at least 2"
        return True, "Parameters are valid"


class OptimizedClusteringModel:
    """TF-IDF -> SVD -> KMeans with confidence + keywords."""
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.reducer = None
        self.is_setup = False
        self.attempts_made: List[str] = []

    def setup_model(self, params: Dict[str, Any]) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.decomposition import TruncatedSVD

            self.vectorizer = TfidfVectorizer(
                max_features=1000, stop_words='english',
                ngram_range=(1, 2), max_df=0.9, min_df=2
            )
            self.reducer = TruncatedSVD(
                n_components=min(params.get("n_components", 5), 50),
                random_state=params.get("random_state", 42)
            )
            n_clusters = max(2, params.get("min_topic_size", 5))
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=params.get("random_state", 42),
                n_init=10, max_iter=300
            )
            self.is_setup = True
            return True
        except ImportError as e:
            self.attempts_made.append(f"Import error: {e}")
            return False
        except Exception as e:
            self.attempts_made.append(f"Setup error: {e}")
            return False

    def fit_transform(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        if not self.is_setup:
            raise ValueError("Model not setup. Call setup_model first.")

        from sklearn.metrics.pairwise import euclidean_distances

        # Vectorize and reduce
        X = self.vectorizer.fit_transform(texts)           # (n_docs x n_features)
        X_reduced = self.reducer.fit_transform(X)          # (n_docs x n_components)

        labels = self.model.fit_predict(X_reduced)         # KMeans in reduced space
        centroids = self.model.cluster_centers_
        dists = euclidean_distances(X_reduced, centroids)

        # Confidence: invert normalized distance, clipped
        max_dist = np.max(dists) if dists.size else 1.0
        confidences = [max(0.1, 1.0 - (dists[i, lbl] / max_dist)) for i, lbl in enumerate(labels)]

        # Keywords (FIXED: compute in TF-IDF space)
        feature_names = self.vectorizer.get_feature_names_out()
        topic_keywords: Dict[int, List[str]] = {}
        for k in set(labels):
            if k < 0: continue
            # mean TF-IDF vector for this cluster (sparse mean)
            mask = (labels == k)
            if not np.any(mask):
                topic_keywords[k] = []
                continue
            mean_vec = X[mask].mean(axis=0)           # 1 x n_features
            mean_arr = np.asarray(mean_vec).ravel()
            top_idx = mean_arr.argsort()[-5:][::-1]
            topic_keywords[k] = [feature_names[i] for i in top_idx]

        metadata = {
            "model_type": "KMeans",
            "n_features": int(X.shape[1]),
            "n_components": int(X_reduced.shape[1]),
            "topic_keywords": topic_keywords
        }
        return labels.tolist(), confidences, metadata


class ClusteringBackend:
    def __init__(self, logger: ActivityLogger):
        self.logger = logger
        self.config = ClusteringConfig()
        self.model = OptimizedClusteringModel()

    def get_clustering_parameters(self, text_count: int, session_id: str) -> Dict[str, Any]:
        params = self.config.get_optimal_parameters(text_count)
        self.logger.log_activity("parameters_suggested", session_id, {
            "text_count": text_count,
            "suggested_params": params
        })
        return params

    def run_clustering(self, texts: List[str], params: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        start = time.time()
        self.logger.log_activity("clustering_started", session_id, {
            "text_count": len(texts),
            "parameters": params
        })
        try:
            ok, msg = self.config.validate_parameters(params)
            if not ok:
                raise ValueError(f"Invalid parameters: {msg}")

            t0 = time.time()
            setup_ok = self.model.setup_model(params)
            setup_time = time.time() - t0
            if not setup_ok:
                raise ValueError("Failed to setup clustering model")

            t1 = time.time()
            topics, probs, meta = self.model.fit_transform(texts)
            clust_time = time.time() - t1

            unique = set(topics)
            n_clusters = len([t for t in unique if t != -1])
            outliers = sum(1 for t in topics if t == -1)
            clustered = len(topics) - outliers
            success_rate = (clustered / len(topics)) * 100 if topics else 0.0

            high = sum(1 for p in probs if p >= 0.7)
            med  = sum(1 for p in probs if 0.3 <= p < 0.7)
            low  = sum(1 for p in probs if p < 0.3)

            total = time.time() - start
            results = {
                "success": True,
                "topics": topics,
                "probabilities": probs,
                "predictions": topics,
                "texts": texts,
                "metadata": meta,
                "statistics": {
                    "n_clusters": n_clusters,
                    "outliers": outliers,
                    "clustered": clustered,
                    "success_rate": success_rate,
                    "total_texts": len(texts)
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
            self.logger.log_performance("clustering_completed", session_id, total, {
                "clusters_found": n_clusters,
                "success_rate": success_rate,
                "model_type": meta.get("model_type", "unknown")
            })
            return results

        except Exception as e:
            dur = time.time() - start
            self.logger.log_error("clustering_failed", session_id, str(e), {
                "duration": dur,
                "parameters": params,
                "text_count": len(texts),
                "attempts_made": getattr(self.model, "attempts_made", [])
            })
            return {"success": False, "error": str(e), "duration": dur,
                    "debug_info": {"attempts_made": getattr(self.model, "attempts_made", [])}}

    def get_cluster_details(self, results: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        if not results.get("success"):
            return {"error": "No valid clustering results"}

        topics = results["topics"]
        probs  = results["probabilities"]
        texts  = results["texts"]
        topic_keywords = results["metadata"].get("topic_keywords", {})

        details: Dict[int, Dict[str, Any]] = {}
        for cid in sorted([t for t in set(topics) if t != -1]):
            idxs = [i for i, t in enumerate(topics) if t == cid]
            ctexts = [texts[i] for i in idxs]
            cprobs = [probs[i] for i in idxs]

            avg_conf = sum(cprobs)/len(cprobs) if cprobs else 0.0
            hi = sum(1 for p in cprobs if p >= 0.7)
            pairs = sorted(zip(ctexts, cprobs), key=lambda x: x[1], reverse=True)[:10]

            details[cid] = {
                "size": len(ctexts),
                "avg_confidence": avg_conf,
                "high_confidence_count": hi,
                "keywords": topic_keywords.get(cid, []),
                "top_texts": pairs,
                "all_texts": ctexts,
                "confidences": cprobs
            }

        # outliers (if any)
        out_idx = [i for i, t in enumerate(topics) if t == -1]
        if out_idx:
            otexts = [texts[i] for i in out_idx]
            oprobs = [probs[i] for i in out_idx]
            details[-1] = {
                "size": len(otexts),
                "avg_confidence": (sum(oprobs)/len(oprobs)) if oprobs else 0.0,
                "keywords": ["outlier"],
                "texts": otexts,
                "confidences": oprobs
            }
        return details
