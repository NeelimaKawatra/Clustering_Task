import os
import time
import json
import logging
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# ============================================================================
# ACTIVITY LOGGER CLASS
# ============================================================================

class ActivityLogger:
    """Logger for tracking user activities and system events"""
    
    def __init__(self, log_file: str = "clustery_activity.log"):
        self.log_file = log_file
        
        # Set up logging
        self.logger = logging.getLogger("ClusteryActivity")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler if it doesn't exist
        if not self.logger.handlers:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "logs"
            if log_dir != "logs":
                os.makedirs(log_dir, exist_ok=True)
            elif not os.path.exists("logs"):
                os.makedirs("logs", exist_ok=True)
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_activity(self, activity_type: str, session_id: str, 
                    details: Dict[str, Any], user_id: Optional[str] = None):
        """Log user activity with structured data"""
        
        activity_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "activity_type": activity_type,
            "details": details
        }
        
        # Log as structured JSON
        self.logger.info(json.dumps(activity_data))
    
    def log_error(self, error_type: str, session_id: str, 
                 error_details: str, metadata: Dict[str, Any] = None):
        """Log errors with structured data"""
        
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "error_type": error_type,
            "error_details": error_details,
            "metadata": metadata or {}
        }
        
        # Log as structured JSON
        self.logger.error(json.dumps(error_data))
    
    def log_performance(self, operation: str, session_id: str, 
                       duration: float, metadata: Dict[str, Any] = None):
        """Log performance metrics"""
        
        perf_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {}
        }
        
        self.logger.info(json.dumps(perf_data))

# ============================================================================
# FAST TEXT PROCESSOR CLASS
# ============================================================================

class FastTextProcessor:
    """Fast text processing and validation"""
    
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                              'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 
                              'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'])
    
    def validate_text_column(self, series: pd.Series) -> Tuple[bool, str]:
        """Fast text column validation"""
        
        # Check if column exists and has data
        if series.empty:
            return False, "Column is empty"
        
        # Remove null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return False, "Column contains only null values"
        
        # Check if most entries are strings and have reasonable length
        text_count = 0
        total_checked = min(100, len(non_null_series))  # Sample first 100 for speed
        
        for value in non_null_series.head(total_checked):
            if isinstance(value, str) and len(str(value).strip()) > 5:
                text_count += 1
        
        text_ratio = text_count / total_checked
        
        if text_ratio < 0.5:
            return False, f"Only {text_ratio:.1%} of entries appear to be valid text"
        
        return True, f"Good text column: {text_ratio:.1%} valid entries"
    
    def analyze_text_quality(self, texts) -> Dict[str, Any]:
        """Fast text quality analysis"""
        
        # Convert pandas Series to list if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = list(texts)
        
        if not texts:
            return {
                "total_texts": 0,
                "avg_length": 0,
                "avg_words": 0,
                "empty_texts": 0,
                "short_texts": 0,
                "unique_texts": 0,
                "sample_size": 0
            }
        
        # Analyze sample for speed
        sample_size = min(100, len(texts))
        sample_texts = texts[:sample_size]
        
        lengths = []
        word_counts = []
        empty_count = 0
        short_count = 0
        unique_texts = set()
        
        for text in sample_texts:
            text_str = str(text).strip()
            
            if not text_str:
                empty_count += 1
                continue
            
            length = len(text_str)
            words = len(text_str.split())
            
            lengths.append(length)
            word_counts.append(words)
            unique_texts.add(text_str.lower())  # Add to unique set
            
            if length < 10:
                short_count += 1
        
        avg_length = np.mean(lengths) if lengths else 0
        avg_words = np.mean(word_counts) if word_counts else 0
        
        return {
            "total_texts": len(texts),
            "avg_length": avg_length,
            "avg_words": avg_words,
            "empty_texts": empty_count,
            "short_texts": short_count,
            "unique_texts": len(unique_texts),
            "sample_size": sample_size
        }
    
    def basic_cleaning(self, text: str) -> str:
        """Basic text cleaning"""
        
        if not text:
            return ""
        
        text = str(text).strip()
        
        # Remove URLs
        import re
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def advanced_cleaning(self, text: str, remove_stopwords: bool = True, 
                         remove_punctuation: bool = True, min_length: int = 2,
                         remove_numbers: bool = True) -> str:
        """Advanced text cleaning"""
        
        if not text:
            return ""
        
        # Start with basic cleaning
        text = self.basic_cleaning(text)
        
        if not text:
            return ""
        
        import re
        
        # Remove numbers if requested
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if requested
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.lower().split()
        
        # Filter words
        filtered_words = []
        for word in words:
            # Check minimum length
            if len(word) < min_length:
                continue
            
            # Remove stopwords if requested
            if remove_stopwords and word in self.stop_words:
                continue
            
            filtered_words.append(word)
        
        return ' '.join(filtered_words)

# ============================================================================
# CLUSTERING CONFIG CLASS
# ============================================================================

class ClusteringConfig:
    """Configuration and parameter management for clustering"""
    
    def __init__(self):
        self.default_params = {
            "min_topic_size": 5,
            "min_samples": 3,
            "n_neighbors": 15,
            "n_components": 5,
            "metric": "cosine",
            "random_state": 42,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    
    def get_optimal_parameters(self, text_count: int) -> Dict[str, Any]:
        """Get optimal parameters based on text count"""
        
        params = self.default_params.copy()
        
        # Adjust min_topic_size based on dataset size
        if text_count < 50:
            params["min_topic_size"] = 3
            params["min_samples"] = 2
        elif text_count < 100:
            params["min_topic_size"] = 5
            params["min_samples"] = 3
        elif text_count < 200:
            params["min_topic_size"] = 8
            params["min_samples"] = 4
        else:
            params["min_topic_size"] = 10
            params["min_samples"] = 5
        
        # Adjust n_neighbors based on dataset size
        if text_count < 50:
            params["n_neighbors"] = 10
        elif text_count < 150:
            params["n_neighbors"] = 15
        else:
            params["n_neighbors"] = 20
        
        return params
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate clustering parameters"""
        
        required_keys = ["min_topic_size", "n_neighbors", "n_components"]
        
        for key in required_keys:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        
        # Validate ranges
        if params["min_topic_size"] < 2:
            return False, "min_topic_size must be at least 2"
        
        if params["n_neighbors"] < 5:
            return False, "n_neighbors must be at least 5"
        
        if params["n_components"] < 2:
            return False, "n_components must be at least 2"
        
        return True, "Parameters are valid"

# ============================================================================
# OPTIMIZED CLUSTERING MODEL CLASS
# ============================================================================

class OptimizedClusteringModel:
    """Optimized clustering model with lazy loading"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_setup = False
        self.attempts_made = []
    
    def setup_model(self, params: Dict[str, Any]) -> bool:
        """Setup the clustering model"""
        
        try:
            # Lazy import of heavy dependencies
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.decomposition import TruncatedSVD
            
            # Setup vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.9,
                min_df=2
            )
            
            # Setup dimensionality reduction
            self.reducer = TruncatedSVD(
                n_components=min(params.get("n_components", 5), 50),
                random_state=params.get("random_state", 42)
            )
            
            # Setup clustering model
            n_clusters = max(2, params.get("min_topic_size", 5))
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=params.get("random_state", 42),
                n_init=10,
                max_iter=300
            )
            
            self.is_setup = True
            return True
            
        except ImportError as e:
            self.attempts_made.append(f"Import error: {str(e)}")
            return False
        except Exception as e:
            self.attempts_made.append(f"Setup error: {str(e)}")
            return False
    
    def fit_transform(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Fit and transform texts"""
        
        if not self.is_setup:
            raise ValueError("Model not setup. Call setup_model first.")
        
        try:
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            
            # Reduce dimensions
            X_reduced = self.reducer.fit_transform(X)
            
            # Fit clustering
            cluster_labels = self.model.fit_predict(X_reduced)
            
            # Calculate simple confidence scores (distance from centroid)
            from sklearn.metrics.pairwise import euclidean_distances
            
            centroids = self.model.cluster_centers_
            distances = euclidean_distances(X_reduced, centroids)
            
            # Get distance to assigned cluster for each point
            confidences = []
            for i, label in enumerate(cluster_labels):
                if label >= 0:
                    dist = distances[i][label]
                    # Convert distance to confidence (lower distance = higher confidence)
                    confidence = max(0.1, 1.0 - (dist / np.max(distances)))
                    confidences.append(confidence)
                else:
                    confidences.append(0.1)  # Low confidence for outliers
            
            # Generate topic keywords
            feature_names = self.vectorizer.get_feature_names_out()
            topic_keywords = {}
            
            for cluster_id in set(cluster_labels):
                if cluster_id >= 0:
                    # Get top terms for this cluster
                    cluster_center = centroids[cluster_id]
                    top_indices = cluster_center.argsort()[-5:][::-1]
                    keywords = [feature_names[i] for i in top_indices]
                    topic_keywords[cluster_id] = keywords
            
            metadata = {
                "model_type": "KMeans",
                "n_features": X.shape[1],
                "n_components": X_reduced.shape[1],
                "topic_keywords": topic_keywords
            }
            
            return cluster_labels.tolist(), confidences, metadata
            
        except Exception as e:
            self.attempts_made.append(f"Clustering error: {str(e)}")
            raise e

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
                original_length = len(df)
                df = df.head(300)
                st.warning(f"File truncated to 300 rows for performance (was {original_length} rows)")
            
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
        if id_column and id_column != "Auto-generate IDs":
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
            # Analyze text quality - convert Series to list
            validation_result["text_quality"] = self.text_processor.analyze_text_quality(df[text_column].tolist())
        
        # Log column validation
        self.logger.log_activity("column_validation", session_id, {
            "text_column": text_column,
            "text_valid": is_valid,
            "text_quality": validation_result["text_quality"]
        })
        
        return validation_result

    def _analyze_id_column(self, series: pd.Series) -> Dict[str, Any]:
        """Simple ID column analysis - check if numeric"""
        
        total_count = len(series)
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {
                "status": "empty",
                "message": "Column is empty",
                "total": total_count,
                "unique": 0
            }
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(series):
            return {
                "status": "numeric",
                "message": "Good numeric ID column",
                "total": total_count,
                "unique": len(non_null_series.unique())
            }
        else:
            return {
                "status": "non_numeric", 
                "message": "Non-numeric column selected",
                "total": total_count,
                "unique": len(non_null_series.unique())
            }
    
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
    
    # ========================================================================
    # TEXT PREPROCESSING (Fast)
    # ========================================================================
    
    def preprocess_texts(self, texts: List[str], method: str, 
                    custom_settings: Dict[str, Any], session_id: str) -> Tuple[List[str], Dict[str, Any]]:
        """Process texts with timing and proper metadata"""
        
        start_time = time.time()
        
        processed_texts = []
        valid_row_indices = []
        
        for i, text in enumerate(texts):
            # Process text based on method
            if method == "none":
                processed_text = str(text) if text else ""
            elif method == "basic":
                processed_text = self.text_processor.basic_cleaning(text)
            elif method == "advanced":
                processed_text = self.text_processor.advanced_cleaning(text)
            elif method == "custom":
                processed_text = self.text_processor.advanced_cleaning(text, **custom_settings)
            else:
                processed_text = self.text_processor.basic_cleaning(text)  # fallback
            
            # Only keep valid texts for clustering
            if processed_text and processed_text.strip() and len(processed_text.strip()) > 2:
                processed_texts.append(processed_text.strip())
                valid_row_indices.append(i)
        
        # Create metadata
        metadata = {
            "method": method,
            "details": f"{method} preprocessing",
            "original_count": len(texts),
            "filtered_count": len(processed_texts),
            "texts_removed": len(texts) - len(processed_texts),
            "valid_row_indices": valid_row_indices,
            "processing_time": time.time() - start_time,
            "original_stats": self.text_processor.analyze_text_quality(texts),
            "processed_stats": self.text_processor.analyze_text_quality(processed_texts)
        }
        
        return processed_texts, metadata
    
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
    
    # ========================================================================
    # CLUSTERING (Lazy Loaded)
    # ========================================================================
    
    def get_clustering_parameters(self, text_count: int, session_id: str) -> Dict[str, Any]:
        """Get optimal parameters for clustering based on text count"""
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
                "predictions": topics,
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
                "duration": error_duration,
                "debug_info": {
                    "attempts_made": self.clustering_model.attempts_made if hasattr(self.clustering_model, 'attempts_made') else []
                }
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
                "avg_confidence": sum(outlier_probs) / len(outlier_probs) if outlier_probs else 0,
                "keywords": ["outlier"],
                "texts": outlier_texts,
                "confidences": outlier_probs
            }
        
        return cluster_details
    
    # ========================================================================
    # EXPORT AND ANALYTICS (Fast)
    # ========================================================================
    
    def export_results(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame, 
                      text_column: str, id_column: str = None, session_id: str = "") -> pd.DataFrame:
        """Create properly aligned output table with simplified logic"""
        
        if not clustering_results.get("success"):
            raise ValueError("Cannot export unsuccessful clustering results")
        
        # Get clustering data
        topics = clustering_results["topics"]
        probabilities = clustering_results["probabilities"] 
        processed_texts = clustering_results["texts"]
        
        # Get alignment information from session state
        row_alignment = st.session_state.get('row_alignment', list(range(len(processed_texts))))
        original_texts = st.session_state.get('original_texts', [])
        clean_ids = st.session_state.get('clean_ids', [])
        user_selections = st.session_state.get('user_selections', {})
        
        # Create output dataframe
        output_data = []
        
        # Process each clustered text
        for proc_idx, (topic, prob, proc_text) in enumerate(zip(topics, probabilities, processed_texts)):
            # Get the original row this processed text came from
            orig_row_idx = row_alignment[proc_idx] if proc_idx < len(row_alignment) else proc_idx
            
            # Build the row data
            row_data = {}
            
            # Add auto-generated ID
            if orig_row_idx < len(clean_ids):
                row_data['auto_generated_id'] = clean_ids[orig_row_idx]
            else:
                row_data['auto_generated_id'] = f"ID_{orig_row_idx+1:03d}"
            
            # Add user's original ID column if they selected one
            if not user_selections.get('id_is_auto_generated', True):
                user_id_col = user_selections.get('id_column_choice')
                if user_id_col and user_id_col in original_data.columns:
                    if orig_row_idx < len(original_data):
                        row_data[f'user_id_{user_id_col}'] = str(original_data[user_id_col].iloc[orig_row_idx])
                    else:
                        row_data[f'user_id_{user_id_col}'] = "N/A"
            
            # Add original text
            user_text_col = user_selections.get('text_column_choice', text_column)
            if orig_row_idx < len(original_texts):
                row_data[f'original_{user_text_col}'] = original_texts[orig_row_idx]
            else:
                row_data[f'original_{user_text_col}'] = "N/A"
            
            # Add processed text
            row_data['processed_text'] = proc_text
            
            # Add clustering results
            row_data['cluster_id'] = topic
            row_data['confidence_score'] = prob
            
            # Add confidence level
            if prob >= 0.7:
                row_data['confidence_level'] = 'High'
            elif prob >= 0.3:
                row_data['confidence_level'] = 'Medium'
            else:
                row_data['confidence_level'] = 'Low'
            
            # Add cluster label if keywords available
            topic_keywords = clustering_results.get("metadata", {}).get("topic_keywords", {})
            if topic in topic_keywords and topic != -1:
                keywords = topic_keywords[topic][:3]  # Top 3 keywords
                row_data['cluster_label'] = "_".join(keywords)
            else:
                row_data['cluster_label'] = "outlier" if topic == -1 else f"cluster_{topic}"
            
            output_data.append(row_data)
        
        # Add rows that were filtered out during preprocessing
        # These will have clustering results as None/NaN
        total_original_rows = len(original_data)
        processed_row_indices = set(row_alignment[:len(processed_texts)])
        
        for orig_idx in range(total_original_rows):
            if orig_idx not in processed_row_indices:
                # This row was filtered out during preprocessing
                row_data = {}
                
                # Add IDs
                row_data['auto_generated_id'] = clean_ids[orig_idx] if orig_idx < len(clean_ids) else f"ID_{orig_idx+1:03d}"
                
                if not user_selections.get('id_is_auto_generated', True):
                    user_id_col = user_selections.get('id_column_choice')
                    if user_id_col and user_id_col in original_data.columns:
                        row_data[f'user_id_{user_id_col}'] = str(original_data[user_id_col].iloc[orig_idx])
                
                # Add original text
                user_text_col = user_selections.get('text_column_choice', text_column)
                if orig_idx < len(original_texts):
                    row_data[f'original_{user_text_col}'] = original_texts[orig_idx]
                else:
                    row_data[f'original_{user_text_col}'] = str(original_data[user_text_col].iloc[orig_idx])
                
                # Mark as filtered out
                row_data['processed_text'] = "[FILTERED OUT - empty or too short]"
                row_data['cluster_id'] = None
                row_data['confidence_score'] = None
                row_data['confidence_level'] = "Not Clustered"
                row_data['cluster_label'] = "not_clustered"
                
                output_data.append(row_data)
        
        # Sort by auto_generated_id to maintain original order
        output_data.sort(key=lambda x: x['auto_generated_id'])
        
        results_df = pd.DataFrame(output_data)
        
        # Log export activity
        self.logger.log_activity("export_results", session_id, {
            "total_rows": len(results_df),
            "clustered_rows": len([row for row in output_data if row['cluster_id'] is not None]),
            "filtered_rows": len([row for row in output_data if row['cluster_id'] is None])
        })
        
        return results_df
    
    def create_summary_export(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame, 
                         text_column: str, id_column: str = None, session_id: str = "") -> pd.DataFrame:
        """Create summary export with only essential columns"""
        
        # Get the detailed results first
        detailed_df = self.export_results(clustering_results, original_data, text_column, id_column, session_id)
        
        # Get user selections to determine column names
        user_selections = st.session_state.get('user_selections', {})
        
        # Define the columns we want in summary
        summary_columns = ['auto_generated_id']
        
        # Add user ID column if it exists
        user_id_cols = [col for col in detailed_df.columns if col.startswith('user_id_')]
        if user_id_cols:
            summary_columns.extend(user_id_cols)
        
        # Add original text column
        original_text_cols = [col for col in detailed_df.columns if col.startswith('original_')]
        if original_text_cols:
            summary_columns.extend(original_text_cols)
        
        # Add clustering results
        clustering_columns = ['cluster_id', 'confidence_score', 'confidence_level', 'cluster_label']
        summary_columns.extend(clustering_columns)
        
        # Filter to only include columns that exist in the detailed dataframe
        available_columns = [col for col in summary_columns if col in detailed_df.columns]
        
        summary_df = detailed_df[available_columns].copy()
        
        return summary_df
    
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


def health_check() -> bool:
    """Health check function"""
    try:
        backend = ClusteryBackend()
        return True
    except Exception:
        return False
    


