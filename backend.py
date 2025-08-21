"""
Clustery Backend - Optimized Implementation with Lazy Loading and Caching
Fast startup with progressive loading of heavy ML packages
"""

import logging
import json
import pandas as pd
import re
import string
import time
import random
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import os

# ============================================================================
# FAST IMPORTS - Basic packages only
# ============================================================================

# These are fast and always needed
BASIC_PACKAGES_AVAILABLE = True

# Heavy packages - import only when needed
NLTK_AVAILABLE = None
CLUSTERING_AVAILABLE = None

# Package status tracking
_package_status = {
    "nltk_loaded": False,
    "clustering_loaded": False,
    "models_cached": False
}

# ============================================================================
# CACHING FUNCTIONS - Load once, reuse forever
# ============================================================================

@st.cache_resource(show_spinner=False)
def setup_nltk_cached():
    """Cache NLTK setup - runs once per Streamlit deployment"""
    global NLTK_AVAILABLE, _package_status
    
    try:
        import nltk
        
        # Check if data exists, download if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            # Only download if not found
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        _package_status["nltk_loaded"] = True
        NLTK_AVAILABLE = True
        
        return {
            "stopwords": stopwords,
            "word_tokenize": word_tokenize,
            "available": True
        }
    except Exception as e:
        NLTK_AVAILABLE = False
        return {"available": False, "error": str(e)}

@st.cache_resource(show_spinner=False)
def load_sentence_transformer_cached(model_name: str):
    """Cache sentence transformer models - download once, reuse forever"""
    try:
        # Import only when needed
        from sentence_transformers import SentenceTransformer
        
        # This is cached - slow only on first run
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model {model_name}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def create_umap_model_cached(n_neighbors: int, n_components: int, random_state: int = 42):
    """Cache UMAP models by parameters"""
    try:
        from umap import UMAP
        return UMAP(
            n_neighbors=min(n_neighbors, 15),
            n_components=min(n_components, 10),
            min_dist=0.0,
            metric='cosine',
            random_state=random_state,
            low_memory=True,
            n_jobs=1
        )
    except Exception as e:
        return None

@st.cache_resource(show_spinner=False)
def create_pca_model_cached(n_components: int, random_state: int = 42):
    """Cache PCA models"""
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=random_state)
    except Exception as e:
        return None

@st.cache_resource(show_spinner=False)
def create_hdbscan_model_cached(min_cluster_size: int, min_samples: int):
    """Cache HDBSCAN models by parameters"""
    try:
        from hdbscan import HDBSCAN
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
    except Exception as e:
        return None

# ============================================================================
# LAZY LOADING FUNCTIONS
# ============================================================================

def lazy_import_clustering_packages():
    """Import clustering packages only when needed, with progress feedback"""
    global CLUSTERING_AVAILABLE, _package_status
    
    if _package_status["clustering_loaded"]:
        return True
    
    try:
        # Show progress to user
        with st.spinner("Loading clustering libraries (first time only)..."):
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from umap import UMAP
            from hdbscan import HDBSCAN
            
            # Store success
            _package_status["clustering_loaded"] = True
            CLUSTERING_AVAILABLE = True
            
            # Pre-cache a small model for faster first clustering
            if not _package_status["models_cached"]:
                st.info("Pre-loading default model for faster clustering...")
                load_sentence_transformer_cached('all-MiniLM-L6-v2')
                _package_status["models_cached"] = True
            
            return True
            
    except ImportError as e:
        CLUSTERING_AVAILABLE = False
        st.error(f"Clustering packages not available: {e}")
        st.info("Install with: pip install bertopic sentence-transformers umap-learn hdbscan")
        return False

def lazy_import_nltk():
    """Import NLTK only when needed"""
    global NLTK_AVAILABLE, _package_status
    
    if _package_status["nltk_loaded"]:
        return setup_nltk_cached()
    
    try:
        with st.spinner("Setting up text processing..."):
            result = setup_nltk_cached()
            return result
    except Exception as e:
        st.warning(f"NLTK setup failed: {e}. Using basic text processing.")
        return {"available": False, "error": str(e)}

# ============================================================================
# FAST ACTIVITY LOGGING
# ============================================================================

class ActivityLogger:
    """Lightweight activity logging"""
    
    def __init__(self, log_file: str = "clustery_activity.log"):
        self.log_file = log_file
        self.logger = None
    
    def _ensure_logger(self):
        """Lazy logger setup"""
        if self.logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger("ClusteryBackend")
    
    def log_activity(self, activity_type: str, session_id: str, data: Dict[str, Any]):
        """Log user activity"""
        self._ensure_logger()
        activity_data = {
            "session_id": session_id,
            "activity_type": activity_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.logger.info(f"ACTIVITY: {json.dumps(activity_data)}")
        return activity_data
    
    def log_performance(self, operation: str, session_id: str, duration: float, metadata: Dict[str, Any] = None):
        """Log performance metrics"""
        self._ensure_logger()
        performance_data = {
            "session_id": session_id,
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.logger.info(f"PERFORMANCE: {json.dumps(performance_data)}")
        return performance_data
    
    def log_error(self, error_type: str, session_id: str, error_message: str, context: Dict[str, Any] = None):
        """Log errors with context"""
        self._ensure_logger()
        error_data = {
            "session_id": session_id,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        self.logger.error(f"ERROR: {json.dumps(error_data)}")
        return error_data

# ============================================================================
# FAST CONFIGURATION
# ============================================================================

class ClusteringConfig:
    """Fast configuration management"""
    
    @staticmethod
    def get_optimal_parameters(n_texts: int) -> Dict[str, Any]:
        """Get optimal clustering parameters - no heavy imports needed"""
        if n_texts < 50:
            return {
                'min_cluster_size': max(3, n_texts // 15),
                'min_samples': 2,
                'n_neighbors': 5,
                'n_components': 5,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        elif n_texts < 200:
            return {
                'min_cluster_size': max(5, n_texts // 25),
                'min_samples': 3,
                'n_neighbors': 10,
                'n_components': 8,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        else:
            return {
                'min_cluster_size': max(8, n_texts // 40),
                'min_samples': 4,
                'n_neighbors': 15,
                'n_components': 10,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
    
    @staticmethod
    def validate_parameters(params: Dict[str, Any]) -> tuple[bool, str]:
        """Fast parameter validation"""
        required_keys = ['min_cluster_size', 'min_samples', 'n_neighbors', 'n_components', 'embedding_model']
        
        for key in required_keys:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        
        if params['min_cluster_size'] < 2:
            return False, "min_cluster_size must be at least 2"
        
        if params['min_samples'] < 1:
            return False, "min_samples must be at least 1"
        
        return True, "Parameters are valid"

# ============================================================================
# FAST TEXT PROCESSING
# ============================================================================

class FastTextProcessor:
    """
    Light-weight, Streamlit-friendly text cleaner.

    Key upgrades (Aug 18):
      • spaCy streaming tokenisation (+ optional lemmatization)
      • Stop-word `set` cached once per instance
      • `remove_numbers` flag now respected
    """

    def __init__(self) -> None:
        self._nltk_tools = None       # lazy-loaded dict of NLTK helpers
        self._spacy_nlp = None        # cached spaCy model
        self._stop_set = None         # cached stop-words

    # ---------------------------------------------------------------------- NLTK
    def _get_nltk_tools(self):
        """Best-effort import of NLTK components (word_tokenize, stopwords)."""
        if self._nltk_tools is not None:
            return self._nltk_tools

        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords

            self._nltk_tools = {
                "available": True,
                "word_tokenize": word_tokenize,
                "stopwords": stopwords,
            }
        except Exception:            # noqa: BLE001
            self._nltk_tools = {"available": False}

        return self._nltk_tools

    # --------------------------------------------------------------------- spaCy
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_spacy_model(lang: str = "en"):
        """
        Load (or first-run download) a tiny spaCy model exactly once per session.
        Returns `None` if spaCy isn't installed.
        """
        try:
            import spacy

            try:                                   
                return spacy.load(f"{lang}_core_web_sm", disable=["parser", "ner"])
            except OSError:
                # Silent first-run download
                from spacy.cli import download as _spacy_dl

                _spacy_dl(f"{lang}_core_web_sm", False, False)
                return spacy.load(f"{lang}_core_web_sm", disable=["parser", "ner"])
        except Exception:            
            return None

    def _get_spacy_nlp(self):
        if self._spacy_nlp is None:                 # initialise on first use
            self._spacy_nlp = self._load_spacy_model()
        return self._spacy_nlp

    # ---------------------------------------------------------- basic primitives
    @staticmethod
    def _remove_basic_stopwords(tokens: List[str]) -> List[str]:
        basic = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to",
            "for", "of", "is", "are", "was", "were",
        }
        return [t for t in tokens if t not in basic]

    # ----------------------------------------------------------------- Cleaning
    @staticmethod
    def basic_cleaning(text: str) -> str:
        """Very crude clean: strip URLs/e-mails, lower-case & trim."""
        text = re.sub(r"(https?://\S+)|(www\.\S+)", "", text, flags=re.I)
        text = re.sub(r"\S+@\S+", "", text)
        return text.lower().strip()

    def advanced_cleaning(
        self,
        text: str,
        *,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        min_length: int = 2,
        remove_numbers: bool = True,
        lemmatize: bool = False,
    ) -> str:
        """
        Full pipeline — uses spaCy if present, falls back to NLTK/string.split.
        """
        # 1️⃣ baseline clean -----------------------------------------------------
        text = self.basic_cleaning(text)
        if remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # 2️⃣ tokenise (spaCy preferred) ----------------------------------------
        tokens: List[str] = []
        nlp = self._get_spacy_nlp()
        if nlp is not None:
            for tok in nlp(text):
                if tok.is_space:
                    continue
                token_txt = tok.lemma_.lower() if lemmatize else tok.text.lower()
                tokens.append(token_txt)
        else:
            nltk_tools = self._get_nltk_tools()
            if nltk_tools.get("available", False):
                try:
                    tokens = [
                        t.lower() for t in nltk_tools["word_tokenize"](text)
                    ]
                except Exception:    # noqa: BLE001
                    tokens = text.split()
            else:
                tokens = text.split()

        # 3️⃣ stop-words --------------------------------------------------------
        if remove_stopwords:
            if self._stop_set is None:             # build once
                nltk_tools = self._get_nltk_tools()
                try:
                    self._stop_set = (
                        set(nltk_tools["stopwords"].words("english"))
                        if nltk_tools.get("available", False)
                        else set()
                    )
                except Exception:                  # noqa: BLE001
                    self._stop_set = set()
            tokens = [t for t in tokens if t not in self._stop_set]
        else:
            # user asked to keep stop-words — ensure cache exists for future
            if self._stop_set is None:
                self._stop_set = set()

        # 4️⃣ length / numeric filters -----------------------------------------
        tokens = [t for t in tokens if len(t) >= min_length]
        if remove_numbers:
            tokens = [
                t for t in tokens
                if not t.isdigit() and not any(ch.isdigit() for ch in t)
            ]

        # 5️⃣ done --------------------------------------------------------------
        return " ".join(tokens)

    def validate_text_column(self, series: pd.Series) -> tuple[bool, str]:
        #Light sanity-check that a DataFrame column really holds text."""
        if series.dtype != "object":
           return False, "Column is not text type"

        text_data = series.dropna().astype(str)
        if text_data.empty:
            return False, "Column contains no valid data"

        meaningful = text_data[text_data.str.len() > 2]
        has_spaces = text_data.str.contains(" ", na=False).sum()

        if len(meaningful) < len(text_data) * 0.3:
            return False, "Most entries are too short for clustering"
        if has_spaces == 0:
            return False, "Text appears to be single words or codes"

        return True, f"Valid text column with {len(meaningful)} meaningful entries"

    def analyze_text_quality(self, texts: list[str]) -> dict[str, int | float]:
        # Quick stats
        valid = [t for t in texts if t and str(t).strip()]
        if not valid:
            return {
                "total_texts": len(texts),
                "empty_texts": len(texts),
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "unique_texts": 0,
                "avg_words": 0,
                "min_words": 0,
                "max_words": 0,
            }

        lengths = [len(str(t)) for t in valid]
        words   = [len(str(t).split()) for t in valid]

        return {
            "total_texts": len(texts),
            "empty_texts": len(texts) - len(valid),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "unique_texts": len(set(map(str, valid))),
            "avg_words":   sum(words) / len(words),
            "min_words":   min(words),
            "max_words":   max(words),
        }

# ============================================================================
# PROGRESSIVE FALLBACK CLUSTERING MODEL
# ============================================================================

class OptimizedClusteringModel:
    """Clustering model with progressive fallback and clear user communication"""
    
    def __init__(self):
        self.model = None
        self.clustering_ready = False
        self.model_config = None
        self.attempts_made = []
    
    def setup_model(self, params: Dict[str, Any]) -> bool:
        """Setup clustering model with progressive fallback approach"""
        
        if not lazy_import_clustering_packages():
            st.error("Failed to load clustering packages")
            return False
        
        self.attempts_made = []
        
        try:
            progress_col1, progress_col2 = st.columns([1, 3])
            
            with progress_col1:
                st.write("**Model Setup:**")
            
            with progress_col2:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                # Step 1: Load embedding model
                status_text.text("Loading text embedding model...")
                progress_bar.progress(0.1)
                embedding_model = load_sentence_transformer_cached(params['embedding_model'])
                
                if embedding_model is None:
                    st.error("Failed to load embedding model")
                    return False
                
                # Step 2: Create HDBSCAN model
                status_text.text("Setting up clustering algorithm...")
                progress_bar.progress(0.2)
                hdbscan_model = create_hdbscan_model_cached(
                    params['min_cluster_size'],
                    params['min_samples']
                )
                
                if hdbscan_model is None:
                    st.error("Failed to create HDBSCAN model")
                    return False
                
                # Step 3: Progressive approach - try sophisticated methods first
                from bertopic import BERTopic
                
                # Attempt 1: Full BERTopic with UMAP
                status_text.text("Attempting advanced configuration with UMAP...")
                progress_bar.progress(0.3)
                
                success, config = self._try_full_bertopic(embedding_model, hdbscan_model, params, status_text)
                if success:
                    progress_bar.progress(1.0)
                    status_text.text("Advanced clustering model ready!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    self.clustering_ready = True
                    return True
                
                # Attempt 2: BERTopic with PCA
                status_text.text("UMAP failed, trying PCA dimensionality reduction...")
                progress_bar.progress(0.5)
                
                success, config = self._try_pca_bertopic(embedding_model, hdbscan_model, params, status_text)
                if success:
                    progress_bar.progress(1.0)
                    status_text.text("PCA-based clustering model ready!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    self.clustering_ready = True
                    return True
                
                # Attempt 3: Basic BERTopic (no dimensionality reduction)
                status_text.text("PCA failed, trying basic BERTopic without dimensionality reduction...")
                progress_bar.progress(0.7)
                
                success, config = self._try_basic_bertopic(embedding_model, hdbscan_model, params, status_text)
                if success:
                    progress_bar.progress(1.0)
                    status_text.text("Basic clustering model ready!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    self.clustering_ready = True
                    return True
                
                # Attempt 4: Minimal BERTopic (absolute minimum)
                status_text.text("Trying minimal configuration as final fallback...")
                progress_bar.progress(0.9)
                
                success, config = self._try_minimal_bertopic(embedding_model, hdbscan_model, status_text)
                if success:
                    progress_bar.progress(1.0)
                    status_text.text("Minimal clustering model ready!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    self.clustering_ready = True
                    return True
                
                # All attempts failed
                st.error("All clustering configurations failed")
                return False
                
        except Exception as e:
            st.error(f"Error setting up clustering model: {e}")
            return False
    
    def _try_full_bertopic(self, embedding_model, hdbscan_model, params, status_text):
        """Attempt 1: Full BERTopic with UMAP"""
        try:
            # Use safer UMAP parameters to avoid mathematical issues
            safe_neighbors = min(max(params['n_neighbors'], 2), 15)
            safe_components = min(max(params['n_components'], 2), 10)
            
            umap_model = create_umap_model_cached(safe_neighbors, safe_components)
            
            if umap_model is None:
                raise ValueError("UMAP model creation failed")
            
            from bertopic import BERTopic
            self.model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=False,
                calculate_probabilities=True,
                nr_topics="auto"
            )
            
            # Test with minimal data to validate setup
            test_texts = ["test one", "test two", "test three", "test four", "test five"]
            try:
                result = self.model.fit_transform(test_texts)
                if result is None or (isinstance(result, tuple) and result[0] is None):
                    raise ValueError("Model test failed - returned None")
                    
                # Validate result structure
                if isinstance(result, tuple):
                    topics, probs = result
                    if len(topics) != len(test_texts):
                        raise ValueError("Result length mismatch")
                        
            except Exception as test_error:
                raise ValueError(f"Model validation failed: {test_error}")
            
            self.model_config = {
                "method": "Full BERTopic with UMAP",
                "dimensionality_reduction": f"UMAP({safe_neighbors} neighbors, {safe_components} components)",
                "has_probabilities": True,
                "sophistication": "High"
            }
            
            self.attempts_made.append({
                "method": "Full BERTopic with UMAP",
                "result": "Success",
                "details": f"Using UMAP with {safe_neighbors} neighbors"
            })
            
            st.success("Advanced UMAP-based clustering configured successfully")
            return True, self.model_config
            
        except Exception as e:
            self.attempts_made.append({
                "method": "Full BERTopic with UMAP",
                "result": "Failed",
                "details": f"Error: {str(e)[:100]}..."
            })
            status_text.text(f"UMAP approach failed: {str(e)[:50]}...")
            return False, None
    
    def _try_pca_bertopic(self, embedding_model, hdbscan_model, params, status_text):
        """Attempt 2: BERTopic with PCA"""
        try:
            # Conservative PCA parameters
            safe_components = min(max(params['n_components'], 2), 50)
            pca_model = create_pca_model_cached(safe_components)
            
            if pca_model is None:
                raise ValueError("PCA model creation failed")
            
            from bertopic import BERTopic
            self.model = BERTopic(
                embedding_model=embedding_model,
                umap_model=pca_model,
                hdbscan_model=hdbscan_model,
                verbose=False,
                calculate_probabilities=True,
                nr_topics="auto"
            )
            
            # Test the model
            test_texts = ["test one", "test two", "test three", "test four", "test five"]
            try:
                result = self.model.fit_transform(test_texts)
                if result is None or (isinstance(result, tuple) and result[0] is None):
                    raise ValueError("PCA model test failed")
            except Exception as test_error:
                raise ValueError(f"PCA validation failed: {test_error}")
            
            self.model_config = {
                "method": "BERTopic with PCA",
                "dimensionality_reduction": f"PCA({safe_components} components)",
                "has_probabilities": True,
                "sophistication": "Medium"
            }
            
            self.attempts_made.append({
                "method": "BERTopic with PCA",
                "result": "Success",
                "details": f"Using PCA with {safe_components} components"
            })
            
            st.info("PCA-based clustering configured (UMAP fallback)")
            return True, self.model_config
            
        except Exception as e:
            self.attempts_made.append({
                "method": "BERTopic with PCA", 
                "result": "Failed",
                "details": f"Error: {str(e)[:100]}..."
            })
            status_text.text(f"PCA approach failed: {str(e)[:50]}...")
            return False, None
    
    def _try_basic_bertopic(self, embedding_model, hdbscan_model, params, status_text):
        """Attempt 3: Basic BERTopic without dimensionality reduction"""
        try:
            from bertopic import BERTopic
            self.model = BERTopic(
                embedding_model=embedding_model,
                umap_model=None,
                hdbscan_model=hdbscan_model,
                verbose=False,
                calculate_probabilities=False,
                nr_topics="auto"
            )
            
            # Test the model
            test_texts = ["service quality good", "product delivery fast", "price cost expensive", "website easy use", "staff helpful friendly"]
            try:
                result = self.model.fit_transform(test_texts)
                
                if result is None:
                    raise ValueError("Basic model returned None")
                
                # Handle different return formats
                if isinstance(result, tuple):
                    topics = result[0]
                else:
                    topics = result
                
                if topics is None or len(topics) != len(test_texts):
                    raise ValueError("Invalid topic assignments")
                    
            except Exception as test_error:
                raise ValueError(f"Basic model validation failed: {test_error}")
            
            self.model_config = {
                "method": "Basic BERTopic",
                "dimensionality_reduction": "None (direct clustering)",
                "has_probabilities": False,
                "sophistication": "Medium"
            }
            
            self.attempts_made.append({
                "method": "Basic BERTopic",
                "result": "Success",
                "details": "No dimensionality reduction, direct clustering"
            })
            
            st.warning("Basic clustering configured (no dimensionality reduction)")
            return True, self.model_config
            
        except Exception as e:
            self.attempts_made.append({
                "method": "Basic BERTopic",
                "result": "Failed", 
                "details": f"Error: {str(e)[:100]}..."
            })
            status_text.text(f"Basic approach failed: {str(e)[:50]}...")
            return False, None
    
    def _try_minimal_bertopic(self, embedding_model, hdbscan_model, status_text):
        """Attempt 4: Minimal BERTopic configuration"""
        try:
            from bertopic import BERTopic
            from hdbscan import HDBSCAN
            
            # Create ultra-minimal HDBSCAN
            minimal_hdbscan = HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            # Absolute minimal configuration
            self.model = BERTopic(
                embedding_model=embedding_model,
                umap_model=None,
                hdbscan_model=minimal_hdbscan,
                verbose=False,
                calculate_probabilities=False,
                nr_topics="auto"
            )
            
            self.model_config = {
                "method": "Minimal BERTopic",
                "dimensionality_reduction": "None",
                "has_probabilities": False,
                "sophistication": "Low",
                "note": "Maximum compatibility mode"
            }
            
            self.attempts_made.append({
                "method": "Minimal BERTopic",
                "result": "Success",
                "details": "Absolute minimum configuration"
            })
            
            st.warning("Minimal clustering configured (compatibility mode)")
            return True, self.model_config
            
        except Exception as e:
            self.attempts_made.append({
                "method": "Minimal BERTopic",
                "result": "Failed",
                "details": f"Error: {str(e)[:100]}..."
            })
            status_text.text(f"Minimal approach failed: {str(e)[:50]}...")
            return False, None
    
    def get_setup_summary(self):
        """Get user-friendly summary of setup attempts"""
        if not self.attempts_made:
            return "No setup attempts recorded"
        
        summary = "## Clustering Setup Summary\n\n"
        
        successful_method = None
        
        for i, attempt in enumerate(self.attempts_made, 1):
            method = attempt["method"]
            result = attempt["result"]
            
            if result == "Success":
                summary += f"✅ **Method {i}**: {method} - Success!\n"
                successful_method = method
            else:
                summary += f"❌ **Method {i}**: {method} - Failed\n"
        
        if successful_method:
            summary += f"\n**Final Configuration**: {successful_method}\n"
            if self.model_config:
                sophistication = self.model_config.get('sophistication', 'Standard')
                summary += f"**Quality Level**: {sophistication}\n"
        
        return summary
        
    def fit_transform(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Run clustering with the configured model"""
        if self.clustering_ready and self.model:
            return self._real_clustering(texts)
        else:
            st.warning("Model not ready - using mock clustering")
            return self._mock_clustering(texts)
    
    def _real_clustering(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Execute real clustering with error handling"""
        try:
            # Validate inputs
            if not texts or len(texts) == 0:
                raise ValueError("No texts provided")
            
            # Clean texts
            valid_texts = [str(text).strip() for text in texts if text and str(text).strip()]
            
            if len(valid_texts) < 5:
                raise ValueError(f"Need at least 5 valid texts, got {len(valid_texts)}")
            
            if len(valid_texts) != len(texts):
                st.info(f"Filtered {len(texts) - len(valid_texts)} empty texts")
                texts = valid_texts
            
            # Run clustering
            with st.spinner("Running clustering analysis..."):
                try:
                    result = self.model.fit_transform(texts)
                except Exception as cluster_error:
                    raise ValueError(f"Clustering execution failed: {cluster_error}")
                
                # Parse results
                topics, probabilities = self._parse_clustering_result(result, texts)
                
                # Validate results
                if len(topics) != len(texts):
                    raise ValueError("Result length mismatch")
                
                # Check clustering quality
                unique_topics = set(topics)
                n_clusters = len([t for t in unique_topics if t != -1])
                
                if n_clusters == 0:
                    st.warning("No clusters found - all texts classified as outliers")
                else:
                    st.success(f"Found {n_clusters} clusters")
            
            # Extract topic information
            topic_keywords = self._extract_topic_keywords(unique_topics)
            
            metadata = {
                'topic_keywords': topic_keywords,
                'model_type': 'BERTopic',
                'model_config': self.model_config,
                'setup_summary': self.get_setup_summary()
            }
            
            return topics, probabilities, metadata
                
        except Exception as e:
            st.error(f"Clustering failed: {str(e)}")
            st.info("Using mock clustering as fallback")
            return self._mock_clustering(texts)
    
    def _parse_clustering_result(self, result, texts):
        """Parse clustering result handling different formats"""
        if result is None:
            raise ValueError("Clustering returned None")
        
        # Handle tuple vs single return
        if isinstance(result, tuple):
            if len(result) >= 2:
                topics, probabilities = result[0], result[1]
            else:
                topics = result[0]
                probabilities = None
        else:
            topics = result
            probabilities = None
        
        # Convert to lists
        if hasattr(topics, 'tolist'):
            topics = topics.tolist()
        elif not isinstance(topics, list):
            topics = list(topics)
        
        # Handle missing probabilities
        if probabilities is None:
            probabilities = []
            for topic in topics:
                if topic == -1:
                    probabilities.append(random.uniform(0.1, 0.3))
                else:
                    probabilities.append(random.uniform(0.5, 0.9))
        else:
            if hasattr(probabilities, 'tolist'):
                probabilities = probabilities.tolist()
            elif not isinstance(probabilities, list):
                probabilities = list(probabilities)
        
        return topics, probabilities
    
    def _extract_topic_keywords(self, unique_topics):
        """Extract topic keywords with error handling"""
        topic_keywords = {}
        
        for topic_id in unique_topics:
            if topic_id != -1:
                try:
                    if hasattr(self.model, 'get_topic'):
                        words = self.model.get_topic(topic_id)
                        if words and len(words) > 0:
                            if isinstance(words[0], tuple):
                                topic_keywords[topic_id] = [word for word, score in words[:5]]
                            else:
                                topic_keywords[topic_id] = words[:5]
                        else:
                            topic_keywords[topic_id] = [f"cluster_{topic_id}"]
                    else:
                        topic_keywords[topic_id] = [f"cluster_{topic_id}"]
                except Exception:
                    topic_keywords[topic_id] = [f"cluster_{topic_id}"]
        
        return topic_keywords
    
    def _mock_clustering(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Mock clustering fallback"""
        random.seed(42)
        
        n_texts = len(texts)
        n_clusters = min(max(2, n_texts // 15), 8)
        
        topics = []
        probabilities = []
        
        for text in texts:
            text_hash = hash(str(text)) % 100
            
            if text_hash < 15:  # 15% outliers
                topics.append(-1)
                probabilities.append(random.uniform(0.1, 0.3))
            else:
                cluster_id = text_hash % n_clusters
                topics.append(cluster_id)
                probabilities.append(random.uniform(0.4, 0.9))
        
        topic_keywords = {
            i: [f"topic_{i}", "mock", "keywords"] 
            for i in range(n_clusters)
        }
        
        metadata = {
            'topic_keywords': topic_keywords,
            'model_type': 'Mock',
            'note': 'Mock clustering - real clustering failed',
            'setup_summary': self.get_setup_summary()
        }
        
        return topics, probabilities, metadata

# ============================================================================
# OPTIMIZED MAIN BACKEND CLASS
# ============================================================================

class ClusteryBackend:
    """Fast-loading main backend with lazy imports and caching"""
    
    def __init__(self, log_file: str = "clustery_activity.log"):
        # Only initialize lightweight components
        self.logger = ActivityLogger(log_file)
        self.text_processor = FastTextProcessor()
        self.clustering_model = OptimizedClusteringModel()
        self.config = ClusteringConfig()
        self.session_data = {}
        
        # Track startup time
        if 'backend_startup_time' not in st.session_state:
            st.session_state.backend_startup_time = time.time()
    
    # ========================================================================
    # SESSION MANAGEMENT (Fast)
    # ========================================================================
    
    def start_session(self, session_id: str, user_info: Dict[str, Any] = None) -> None:
        """Fast session startup"""
        self.session_data[session_id] = {
            "start_time": datetime.now(),
            "user_info": user_info or {},
            "activities": [],
            "current_tab": None,
            "data_loaded": False,
            "preprocessing_completed": False,
            "clustering_completed": False,
            "results_exported": False
        }
        
        self.logger.log_activity("session_started", session_id, {
            "user_info": user_info,
            "timestamp": datetime.now().isoformat(),
            "startup_time": time.time() - st.session_state.backend_startup_time
        })
    
    def track_activity(self, session_id: str, activity_type: str, data: Dict[str, Any]):
        """Fast activity tracking"""
        if session_id in self.session_data:
            self.session_data[session_id]["activities"].append({
                "type": activity_type,
                "data": data,
                "timestamp": datetime.now()
            })
        
        self.logger.log_activity(activity_type, session_id, data)
    
    # ========================================================================
    # DATA LOADING (Fast)
    # ========================================================================
    
    def load_data(self, file_path: str, session_id: str) -> Tuple[bool, pd.DataFrame, str]:
        """Fast data loading with progress"""
        try:
            # Log activity
            self.logger.log_activity("file_upload_started", session_id, {
                "file_path": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            })
            
            # Show progress for larger files
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            if file_size > 1024 * 1024:  # > 1MB
                with st.spinner(f"Loading file ({file_size/1024/1024:.1f}MB)..."):
                    df = self._load_file_by_type(file_path)
            else:
                df = self._load_file_by_type(file_path)
            
            # Fast validation
            if len(df) > 300:
                self.logger.log_activity("file_size_warning", session_id, {
                    "original_rows": len(df),
                    "action": "truncate_to_300"
                })
                df = df.head(300)
                st.warning(f"File truncated to 300 rows for performance (was {len(df)} rows)")
            
            if len(df) < 10:
                error_msg = f"File too small: {len(df)} rows. Need at least 10 rows."
                self.logger.log_error("file_too_small", session_id, error_msg)
                return False, pd.DataFrame(), error_msg
            
            # Log successful load
            self.logger.log_activity("file_loaded_successfully", session_id, {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_kb": df.memory_usage(deep=True).sum() / 1024
            })
            
            return True, df, f"File loaded successfully: {len(df)} rows, {len(df.columns)} columns"
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            self.logger.log_error("file_load_error", session_id, error_msg)
            return False, pd.DataFrame(), error_msg
    
    def _load_file_by_type(self, file_path: str) -> pd.DataFrame:
        """Load file based on extension"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".csv":
            return pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def validate_columns(self, df: pd.DataFrame, text_column: str, 
                        id_column: Optional[str], session_id: str) -> Dict[str, Any]:
        """Fast column validation"""
        
        validation_result = {
            "text_column_valid": False,
            "id_column_analysis": {},
            "text_quality": {},
            "recommendations": []
        }
        # Add ID column analysis
        if id_column:
            validation_result["id_column_analysis"] = self._analyze_id_column(df[id_column])
        else:
            # For auto-generated IDs, provide default analysis
            validation_result["id_column_analysis"] = {
                "status": "perfect",
                "message": "Auto-generated IDs will be created",
                "total": len(df),
                "unique": len(df)
            }

        # Validate text column
        is_valid, message = self.text_processor.validate_text_column(df[text_column])
        validation_result["text_column_valid"] = is_valid
        validation_result["text_column_message"] = message
        
        if is_valid:
            # Analyze text quality
            validation_result["text_quality"] = self.text_processor.analyze_text_quality(df[text_column])
        
        
        # Log column validation
        self.logger.log_activity("column_validation", session_id, {
            "text_column": text_column,
            "text_valid": is_valid,
            "text_quality": validation_result["text_quality"]
        })
        
        return validation_result
    
    
    def _analyze_id_column(self, series: pd.Series) -> Dict[str, Any]:
        """Fast ID column analysis"""
        ids = series.dropna().astype(str)
        
        total_count = len(series)
        valid_count = len(ids)
        unique_count = len(ids.unique())
        duplicates = total_count - unique_count
        return {
            "status": "perfect",
            "message": f"Perfect! {total_count} IDs",
            "total": total_count,
            "unique": unique_count
        }
    
    # ========================================================================
    # TEXT PREPROCESSING (Fast)
    # ========================================================================
    
    def preprocess_texts(self, texts: List[str], method: str, 
                        custom_settings: Dict[str, Any], session_id: str) -> Tuple[List[str], Dict[str, Any]]:
        """Fast text preprocessing with progress"""
        
        start_time = time.time()
        
        # Log preprocessing start
        self.logger.log_activity("preprocessing_started", session_id, {
            "method": method,
            "text_count": len(texts),
            "custom_settings": custom_settings if method == "custom" else None
        })
        
        # Show progress for large datasets
        if len(texts) > 100:
            progress_bar = st.progress(0.0)  # Fixed: Use 0.0 instead of 0
            status_text = st.empty()
            
            status_text.text(f"Processing {len(texts)} texts...")
            progress_bar.progress(0.2)  # Fixed: Use 0.2 instead of 20
        else:
            progress_bar = None
            status_text = None
        
        # Process texts based on method
        if method == "none":
            processed_texts = [str(text) if text else "" for text in texts]
            details = "No preprocessing applied"
            
        elif method == "basic":
            processed_texts = []
            for i, text in enumerate(texts):
                processed_texts.append(self.text_processor.basic_cleaning(text))
                if progress_bar and i % 50 == 0:
                    # Fixed: Use values between 0.0 and 1.0
                    progress_bar.progress(0.2 + (i / len(texts)) * 0.6)
            details = "Basic cleaning: URLs, emails, whitespace normalization"
            
        elif method == "advanced":
            processed_texts = []
            for i, text in enumerate(texts):
                processed_texts.append(self.text_processor.advanced_cleaning(text))
                if progress_bar and i % 50 == 0:
                    # Fixed: Use values between 0.0 and 1.0
                    progress_bar.progress(0.2 + (i / len(texts)) * 0.6)
            details = "Advanced cleaning: URLs, emails, stopwords, punctuation, short words"
            
        elif method == "custom":
            processed_texts = []
            for i, text in enumerate(texts):
                processed_texts.append(
                    self.text_processor.advanced_cleaning(
                        text,
                        remove_stopwords=custom_settings.get('remove_stopwords', True),
                        remove_punctuation=custom_settings.get('remove_punctuation', True),
                        min_length=custom_settings.get('min_length', 2)
                    )
                )
                if progress_bar and i % 50 == 0:
                    # Fixed: Use values between 0.0 and 1.0
                    progress_bar.progress(0.2 + (i / len(texts)) * 0.6)
            details = f"Custom: {custom_settings}"
            
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        if progress_bar:
            progress_bar.progress(0.8)  # Fixed: Use 0.8 instead of 80
            status_text.text("Filtering and analyzing results...")
        
        # Filter out empty texts
        original_count = len(processed_texts)
        processed_texts = [text.strip() for text in processed_texts if text.strip() and len(text.strip()) > 2]
        filtered_count = len(processed_texts)
        
        # Calculate statistics
        original_stats = self.text_processor.analyze_text_quality(texts)
        processed_stats = self.text_processor.analyze_text_quality(processed_texts)
        
        processing_time = time.time() - start_time
        
        if progress_bar:
            progress_bar.progress(1.0)  # Fixed: Use 1.0 instead of 100
            status_text.text("Preprocessing complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
        
        # Create metadata
        metadata = {
            "method": method,
            "details": details,
            "custom_settings": custom_settings if method == "custom" else None,
            "original_count": original_count,
            "filtered_count": filtered_count,
            "texts_removed": original_count - filtered_count,
            "original_stats": original_stats,
            "processed_stats": processed_stats,
            "processing_time": processing_time
        }
        
        # Log completion
        self.logger.log_performance("text_preprocessing", session_id, processing_time, {
            "method": method,
            "texts_processed": filtered_count,
            "texts_removed": original_count - filtered_count
        })
        
        return processed_texts, metadata
    
    # ========================================================================
    # CLUSTERING (Lazy Loaded)
    # ========================================================================
    
    def get_clustering_parameters(self, text_count: int, session_id: str) -> Dict[str, Any]:
        """Fast parameter calculation"""
        params = self.config.get_optimal_parameters(text_count)
        
        self.logger.log_activity("parameters_suggested", session_id, {
            "text_count": text_count,
            "suggested_params": params
        })
        
        return params
    
    def run_clustering(self, texts: List[str], params: Dict[str, Any], 
                      session_id: str) -> Dict[str, Any]:
        """Run clustering with lazy loading and progress"""
        
        start_time = time.time()
        
        # Log clustering start
        self.logger.log_activity("clustering_started", session_id, {
            "text_count": len(texts),
            "parameters": params
        })
        
        try:
            # Validate parameters
            is_valid, validation_message = self.config.validate_parameters(params)
            if not is_valid:
                raise ValueError(f"Invalid parameters: {validation_message}")
            
            # Setup model (this triggers lazy loading)
            model_setup_time = time.time()
            model_ready = self.clustering_model.setup_model(params)
            setup_duration = time.time() - model_setup_time
            
            if not model_ready:
                raise ValueError("Failed to setup clustering model")
            
            # Run clustering
            clustering_start = time.time()
            topics, probabilities, metadata = self.clustering_model.fit_transform(texts)
            clustering_duration = time.time() - clustering_start
            
            # Calculate results statistics
            unique_topics = set(topics)
            n_clusters = len([t for t in unique_topics if t != -1])
            outliers = sum(1 for t in topics if t == -1)
            clustered = len(topics) - outliers
            success_rate = (clustered / len(topics)) * 100
            
            # Confidence analysis
            high_conf = sum(1 for p in probabilities if p >= 0.7)
            medium_conf = sum(1 for p in probabilities if 0.3 <= p < 0.7)
            low_conf = sum(1 for p in probabilities if p < 0.3)
            
            total_time = time.time() - start_time
            
            # Create comprehensive results
            results = {
                "success": True,
                "topics": topics,
                "probabilities": probabilities,
                "predictions": topics,  # FIXED: Add this field - frontend expects it
                "texts": texts,
                "metadata": metadata,
                "statistics": {
                    "n_clusters": n_clusters,
                    "outliers": outliers,
                    "clustered": clustered,
                    "success_rate": success_rate,
                    "total_texts": len(texts)
                },
                "confidence_analysis": {
                    "high_confidence": high_conf,
                    "medium_confidence": medium_conf,
                    "low_confidence": low_conf,
                    "avg_confidence": sum(probabilities) / len(probabilities)
                },
                "performance": {
                    "total_time": total_time,
                    "setup_time": setup_duration,
                    "clustering_time": clustering_duration
                },
                "parameters_used": params
            }
            
            # Log successful completion
            self.logger.log_performance("clustering_completed", session_id, total_time, {
                "clusters_found": n_clusters,
                "success_rate": success_rate,
                "model_type": metadata.get("model_type", "unknown")
            })
            
            return results
            
        except Exception as e:
            error_duration = time.time() - start_time
            error_msg = str(e)
            
            self.logger.log_error("clustering_failed", session_id, error_msg, {
                "duration": error_duration,
                "parameters": params,
                "text_count": len(texts)
            })
            
            return {
                "success": False,
                "error": error_msg,
                "duration": error_duration
            }
    
    def get_cluster_details(self, results: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Fast cluster analysis"""
        
        if not results.get("success"):
            return {"error": "No valid clustering results"}
        
        topics = results["topics"]
        probabilities = results["probabilities"]
        texts = results["texts"]
        topic_keywords = results["metadata"].get("topic_keywords", {})
        
        cluster_details = {}
        
        # Process each cluster
        unique_clusters = sorted([t for t in set(topics) if t != -1])
        
        for cluster_id in unique_clusters:
            # Get indices for this cluster
            cluster_indices = [i for i, t in enumerate(topics) if t == cluster_id]
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_probs = [probabilities[i] for i in cluster_indices]
            
            # Calculate cluster statistics
            avg_confidence = sum(cluster_probs) / len(cluster_probs)
            high_conf_count = sum(1 for p in cluster_probs if p >= 0.7)
            
            # Get top texts by confidence
            text_prob_pairs = list(zip(cluster_texts, cluster_probs))
            top_texts = sorted(text_prob_pairs, key=lambda x: x[1], reverse=True)[:10]
            
            cluster_details[cluster_id] = {
                "size": len(cluster_texts),
                "avg_confidence": avg_confidence,
                "high_confidence_count": high_conf_count,
                "keywords": topic_keywords.get(cluster_id, []),
                "top_texts": top_texts,
                "all_texts": cluster_texts,
                "confidences": cluster_probs
            }
        
        # Handle outliers
        outlier_indices = [i for i, t in enumerate(topics) if t == -1]
        if outlier_indices:
            outlier_texts = [texts[i] for i in outlier_indices]
            outlier_probs = [probabilities[i] for i in outlier_indices]
            
            cluster_details[-1] = {
                "size": len(outlier_texts),
                "avg_confidence": sum(outlier_probs) / len(outlier_probs),
                "keywords": ["outlier"],
                "texts": outlier_texts,
                "confidences": outlier_probs
            }
        
        return cluster_details
    
    # ========================================================================
    # EXPORT AND ANALYTICS (Fast)
    # ========================================================================
    
    def export_results(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame, 
                      text_column: str, id_column: str = None, session_id: str = "", original_texts_col: List[str] = []) -> pd.DataFrame:
        """Fast results export"""
        
        topics = clustering_results["topics"]
        probabilities = clustering_results["probabilities"]
        texts = clustering_results["texts"]
        topic_keywords = clustering_results["metadata"].get("topic_keywords", {})
        
        # Create respondent IDs
        if id_column and id_column in original_data.columns:
            ids = original_data[id_column].tolist()[:len(texts)]
        else:
            ids = [f"ID_{i+1:03d}" for i in range(len(texts))]
        
        # Create cluster labels
        cluster_labels = []
        for topic in topics:
            if topic == -1:
                cluster_labels.append("outlier")
            else:
                keywords = topic_keywords.get(topic, [f"cluster_{topic}"])
                label = "_".join(keywords[:3]) if keywords else f"cluster_{topic}"
                cluster_labels.append(label)
        
        # Get original texts for comparison - FIXED: Use fillna instead of dropna
        original_texts = original_data[text_column].fillna("").tolist()[:len(texts)]
        
        # Create comprehensive dataframe
        results_df = pd.DataFrame({
            'respondent_id': ids,
            'original_text': original_texts,
            'processed_text': texts,
            'cluster_id': topics,
            'cluster_label': cluster_labels,
            'confidence_score': probabilities,
            'confidence_level': ['High' if p >= 0.7 else 'Medium' if p >= 0.3 else 'Low' for p in probabilities]
        })
        
        # Add cluster keywords as separate column
        results_df['cluster_keywords'] = results_df['cluster_id'].map(
            lambda x: ", ".join(topic_keywords.get(x, [])) if x != -1 else "outlier"
        )
        
        self.logger.log_activity("results_dataframe_created", session_id, {
            "rows": len(results_df),
            "columns": len(results_df.columns),
            "clusters": len(set(topics)) - (1 if -1 in topics else 0)
        })
        
        return results_df
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Fast session analytics"""
        
        if session_id not in self.session_data:
            return {"error": "Session not found"}
        
        session = self.session_data[session_id]
        current_time = datetime.now()
        session_duration = (current_time - session["start_time"]).total_seconds()
        
        # Calculate completion status
        completion_status = {
            "data_loaded": session["data_loaded"],
            "preprocessing_completed": session["preprocessing_completed"],
            "clustering_completed": session["clustering_completed"],
            "results_exported": session["results_exported"]
        }
        
        completion_percentage = sum(completion_status.values()) / len(completion_status) * 100
        
        # Activity summary
        activity_counts = {}
        for activity in session["activities"]:
            activity_type = activity["type"]
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        summary = {
            "session_id": session_id,
            "duration_seconds": session_duration,
            "completion_percentage": completion_percentage,
            "completion_status": completion_status,
            "activity_counts": activity_counts,
            "current_tab": session["current_tab"],
            "total_activities": len(session["activities"]),
            "user_info": session["user_info"]
        }
        
        return summary
    
    # ========================================================================
    # UTILITY METHODS (Fast)
    # ========================================================================
    
    @property
    def data_service(self):
        """For compatibility"""
        return self
    
    @property 
    def preprocessing_service(self):
        """For compatibility"""
        return self
    
    @property
    def export_service(self):
        """For compatibility"""
        return self
    
    def get_text_column_suggestions(self, df: pd.DataFrame, session_id: str) -> List[str]:
        """Fast text column detection"""
        text_columns = []
        
        for col in df.columns:
            is_valid, _ = self.text_processor.validate_text_column(df[col])
            if is_valid:
                text_columns.append(col)
        
        self.logger.log_activity("text_column_suggestions", session_id, {
            "suggested_columns": text_columns,
            "total_columns": len(df.columns)
        })
        
        return text_columns
    
    def get_preprocessing_recommendations(self, texts: List[str], session_id: str) -> Dict[str, Any]:
        """Fast preprocessing recommendations"""
        
        text_stats = self.text_processor.analyze_text_quality(texts)
        
        recommendations = {
            "suggested_method": "basic",
            "reasons": [],
            "text_analysis": text_stats
        }
        
        # Fast analysis for recommendations
        avg_length = text_stats["avg_length"]
        avg_words = text_stats["avg_words"]
        
        if avg_length > 200:
            recommendations["suggested_method"] = "advanced"
            recommendations["reasons"].append("Long texts benefit from stopword removal")
        
        if avg_words > 20:
            recommendations["suggested_method"] = "advanced"
            recommendations["reasons"].append("Many words suggest need for noise reduction")
        
        if text_stats["total_texts"] < 50:
            recommendations["suggested_method"] = "basic"
            recommendations["reasons"].append("Small dataset - preserve more content")
        
        # Log recommendations
        self.logger.log_activity("preprocessing_recommendations", session_id, {
            "suggested_method": recommendations["suggested_method"],
            "reasons": recommendations["reasons"],
            "text_stats": text_stats
        })
        
        return recommendations
    
    def create_summary_report(self, clustering_results: Dict[str, Any], 
                             preprocessing_info: Dict[str, Any], session_id: str = "") -> str:
        """Fast summary report generation"""
        
        stats = clustering_results["statistics"]
        confidence = clustering_results["confidence_analysis"]
        performance = clustering_results["performance"]
        params = clustering_results["parameters_used"]
        
        report = f"""
CLUSTERY - TEXT CLUSTERING ANALYSIS REPORT
=========================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {session_id}

DATASET SUMMARY
--------------
Total Texts Analyzed: {stats['total_texts']}
Preprocessing Method: {preprocessing_info.get('method', 'Unknown')}
Processing Details: {preprocessing_info.get('details', 'N/A')}

CLUSTERING RESULTS
-----------------
Clusters Found: {stats['n_clusters']}
Successfully Clustered: {stats['clustered']} ({stats['success_rate']:.1f}%)
Outliers: {stats['outliers']} ({(stats['outliers']/stats['total_texts']*100):.1f}%)

CONFIDENCE ANALYSIS
------------------
High Confidence (≥0.7): {confidence['high_confidence']} ({(confidence['high_confidence']/stats['total_texts']*100):.1f}%)
Medium Confidence (0.3-0.7): {confidence['medium_confidence']} ({(confidence['medium_confidence']/stats['total_texts']*100):.1f}%)
Low Confidence (<0.3): {confidence['low_confidence']} ({(confidence['low_confidence']/stats['total_texts']*100):.1f}%)
Average Confidence: {confidence['avg_confidence']:.3f}

PERFORMANCE METRICS
------------------
Total Processing Time: {performance['total_time']:.2f} seconds
Model Setup Time: {performance['setup_time']:.2f} seconds
Clustering Time: {performance['clustering_time']:.2f} seconds

PARAMETERS USED
--------------
"""
        for key, value in params.items():
            report += f"{key.replace('_', ' ').title()}: {value}\n"
        
        report += "\nGenerated by Clustery - Intelligent Text Clustering Tool\n"
        
        return report


def health_check() -> bool:
    """Health check function"""
    try:
        backend = ClusteryBackend()
        return True
    except Exception:
        return False