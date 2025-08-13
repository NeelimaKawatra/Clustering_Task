"""
Clustery Backend - Complete Implementation
All backend services in one file for simplicity
"""

import logging
import json
import pandas as pd
import re
import string
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import os

# Optional imports with graceful fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

# ============================================================================
# ACTIVITY LOGGING
# ============================================================================

class ActivityLogger:
    """Centralized activity logging for user behavior tracking"""
    
    def __init__(self, log_file: str = "clustery_activity.log"):
        self.log_file = log_file
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logging configuration"""
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
        """Log user activity with structured data"""
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
# CONFIGURATION
# ============================================================================

class ClusteringConfig:
    """Configuration management for clustering parameters"""
    
    @staticmethod
    def get_optimal_parameters(n_texts: int) -> Dict[str, Any]:
        """Get optimal clustering parameters based on dataset size"""
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
        """Validate clustering parameters"""
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
# TEXT PROCESSING
# ============================================================================

class TextProcessor:
    """Advanced text processing with multiple cleaning levels"""
    
    def __init__(self):
        self.nltk_ready = self._setup_nltk()
    
    def _setup_nltk(self) -> bool:
        """Setup NLTK data if available"""
        if not NLTK_AVAILABLE:
            return False
        
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            return True
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                return True
            except:
                return False
    
    def basic_cleaning(self, text: str) -> str:
        """Basic text cleaning: URLs, emails, whitespace"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_cleaning(self, text: str, remove_stopwords: bool = True, 
                         remove_punctuation: bool = True, min_length: int = 2) -> str:
        """Advanced text cleaning with customizable options"""
        if pd.isna(text) or text == "":
            return ""
        
        # Basic cleaning first
        text = self.basic_cleaning(text)
        
        # Remove punctuation if requested
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        if self.nltk_ready:
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
        else:
            tokens = text.split()
        
        # Remove stopwords if requested
        if remove_stopwords:
            if self.nltk_ready:
                try:
                    stop_words = set(stopwords.words('english'))
                    tokens = [token for token in tokens if token not in stop_words]
                except:
                    tokens = self._remove_basic_stopwords(tokens)
            else:
                tokens = self._remove_basic_stopwords(tokens)
        
        # Filter by minimum length
        tokens = [token for token in tokens if len(token) >= min_length]
        
        # Remove digits-only tokens
        tokens = [token for token in tokens if not token.isdigit()]
        
        return ' '.join(tokens)
    
    def _remove_basic_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove basic English stopwords (fallback when NLTK unavailable)"""
        basic_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        return [token for token in tokens if token not in basic_stopwords]
    
    def validate_text_column(self, series: pd.Series) -> Tuple[bool, str]:
        """Validate if a pandas Series contains suitable text for clustering"""
        if series.dtype != 'object':
            return False, "Column is not text type"
        
        # Remove null values and convert to string
        text_data = series.dropna().astype(str)
        
        if len(text_data) == 0:
            return False, "Column contains no valid data"
        
        # Check for meaningful text
        meaningful_texts = text_data[text_data.str.len() > 2]
        has_spaces = text_data.str.contains(' ', na=False).sum()
        
        if len(meaningful_texts) < len(text_data) * 0.3:
            return False, "Most entries are too short for clustering"
        
        if has_spaces == 0:
            return False, "Text appears to be single words or codes"
        
        return True, f"Valid text column with {len(meaningful_texts)} meaningful entries"
    
    def analyze_text_quality(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text quality and provide detailed statistics"""
        valid_texts = [text for text in texts if text and str(text).strip()]
        
        if not valid_texts:
            return {
                'total_texts': len(texts),
                'empty_texts': len(texts),
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'unique_texts': 0,
                'avg_words': 0,
                'min_words': 0,
                'max_words': 0
            }
        
        lengths = [len(str(text)) for text in valid_texts]
        word_counts = [len(str(text).split()) for text in valid_texts]
        
        return {
            'total_texts': len(texts),
            'empty_texts': len(texts) - len(valid_texts),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'unique_texts': len(set(str(text) for text in valid_texts)),
            'avg_words': sum(word_counts) / len(word_counts),
            'min_words': min(word_counts),
            'max_words': max(word_counts)
        }

# ============================================================================
# CLUSTERING MODEL
# ============================================================================

class ClusteringModel:
    """BERTopic clustering model with fallback to mock clustering"""
    
    def __init__(self):
        self.model = None
        self.clustering_ready = CLUSTERING_AVAILABLE
    
    def setup_model(self, params: Dict[str, Any]) -> bool:
        """Setup BERTopic model with given parameters"""
        if not self.clustering_ready:
            return False
        
        try:
            # Setup UMAP
            umap_model = UMAP(
                n_neighbors=min(params['n_neighbors'], 15),
                n_components=min(params['n_components'], 10),
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            # Setup HDBSCAN
            hdbscan_model = HDBSCAN(
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            # Setup embedding model
            embedding_model = SentenceTransformer(params['embedding_model'])
            
            # Create BERTopic model
            self.model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=False,
                calculate_probabilities=True,
                nr_topics="auto"
            )
            
            return True
            
        except Exception as e:
            print(f"Error setting up BERTopic model: {e}")
            return False
    
    def fit_transform(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Run clustering on texts"""
        if self.clustering_ready and self.model:
            return self._real_clustering(texts)
        else:
            return self._mock_clustering(texts)
    
    def _real_clustering(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Real BERTopic clustering"""
        try:
            topics, probabilities = self.model.fit_transform(texts)
            
            # Extract topic information
            topic_info = self.model.get_topic_info()
            
            # Get topic keywords
            topic_keywords = {}
            for topic_id in set(topics):
                if topic_id != -1:
                    words = self.model.get_topic(topic_id)[:5]
                    topic_keywords[topic_id] = [word for word, score in words]
            
            metadata = {
                'topic_keywords': topic_keywords,
                'topic_info': topic_info.to_dict() if hasattr(topic_info, 'to_dict') else {},
                'model_type': 'BERTopic'
            }
            
            return topics.tolist(), probabilities.tolist(), metadata
            
        except Exception as e:
            print(f"Error in real clustering: {e}")
            return self._mock_clustering(texts)
    
    def _mock_clustering(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Mock clustering for testing/demo purposes"""
        random.seed(42)  # For consistent results
        
        # Generate reasonable number of clusters
        n_clusters = min(max(2, len(texts) // 20), 8)
        
        # Generate topics (cluster assignments)
        topics = []
        for _ in range(len(texts)):
            if random.random() > 0.15:  # 85% get assigned to clusters
                topics.append(random.randint(0, n_clusters - 1))
            else:  # 15% are outliers
                topics.append(-1)
        
        # Generate realistic probabilities
        probabilities = []
        for topic in topics:
            if topic == -1:  # Outliers have low confidence
                probabilities.append(random.uniform(0.1, 0.3))
            else:  # Clustered texts have higher confidence
                probabilities.append(random.uniform(0.4, 0.95))
        
        # Mock topic keywords
        mock_keywords = {
            0: ["service", "customer", "support"],
            1: ["quality", "product", "good"],
            2: ["price", "cost", "expensive"],
            3: ["delivery", "shipping", "fast"],
            4: ["website", "online", "easy"],
            5: ["staff", "helpful", "friendly"],
            6: ["payment", "secure", "safe"],
            7: ["experience", "overall", "satisfied"]
        }
        
        topic_keywords = {i: mock_keywords.get(i, ["topic", "words", "here"]) 
                         for i in range(n_clusters)}
        
        metadata = {
            'topic_keywords': topic_keywords,
            'model_type': 'Mock',
            'note': 'This is mock data for demonstration'
        }
        
        return topics, probabilities, metadata

# ============================================================================
# MAIN BACKEND CLASS
# ============================================================================

class ClusteryBackend:
    """Main backend interface - all services in one class"""
    
    def __init__(self, log_file: str = "clustery_activity.log"):
        self.logger = ActivityLogger(log_file)
        self.text_processor = TextProcessor()
        self.clustering_model = ClusteringModel()
        self.config = ClusteringConfig()
        self.session_data = {}
    
    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================
    
    def start_session(self, session_id: str, user_info: Dict[str, Any] = None) -> None:
        """Start a new user session"""
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
            "timestamp": datetime.now().isoformat()
        })
    
    def track_activity(self, session_id: str, activity_type: str, data: Dict[str, Any]):
        """Track user activity"""
        if session_id in self.session_data:
            self.session_data[session_id]["activities"].append({
                "type": activity_type,
                "data": data,
                "timestamp": datetime.now()
            })
        
        self.logger.log_activity(activity_type, session_id, data)
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    def load_data(self, file_path: str, session_id: str) -> Tuple[bool, pd.DataFrame, str]:
        """Load CSV or Excel file with validation"""
        try:
            # Log activity
            self.logger.log_activity("file_upload_started", session_id, {
                "file_path": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            })
            
            # Determine file type and load
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                error_msg = f"Unsupported file format: {file_ext}"
                self.logger.log_error("unsupported_file_format", session_id, error_msg)
                return False, pd.DataFrame(), error_msg
            
            # Validate file size
            if len(df) > 300:
                self.logger.log_activity("file_size_warning", session_id, {
                    "original_rows": len(df),
                    "action": "truncate_to_300"
                })
                df = df.head(300)
            
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
            
            return True, df, "File loaded successfully"
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            self.logger.log_error("file_load_error", session_id, error_msg)
            return False, pd.DataFrame(), error_msg
    
    def validate_columns(self, df: pd.DataFrame, text_column: str, 
                        id_column: Optional[str], session_id: str) -> Dict[str, Any]:
        """Validate selected columns and return analysis"""
        
        validation_result = {
            "text_column_valid": False,
            "id_column_analysis": {},
            "text_quality": {},
            "recommendations": []
        }
        
        # Validate text column
        is_valid, message = self.text_processor.validate_text_column(df[text_column])
        validation_result["text_column_valid"] = is_valid
        validation_result["text_column_message"] = message
        
        if is_valid:
            # Analyze text quality
            validation_result["text_quality"] = self.text_processor.analyze_text_quality(df[text_column])
        
        # Analyze ID column
        if id_column:
            id_analysis = self._analyze_id_column(df[id_column])
            validation_result["id_column_analysis"] = id_analysis
        else:
            validation_result["id_column_analysis"] = {
                "status": "auto_generate",
                "message": "Will auto-generate sequential IDs"
            }
        
        # Log column validation
        self.logger.log_activity("column_validation", session_id, {
            "text_column": text_column,
            "id_column": id_column,
            "text_valid": is_valid,
            "text_quality": validation_result["text_quality"]
        })
        
        return validation_result
    
    def _analyze_id_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze ID column for duplicates and validity"""
        ids = series.dropna().astype(str)
        
        total_count = len(series)
        valid_count = len(ids)
        unique_count = len(ids.unique())
        duplicates = total_count - unique_count
        
        if duplicates > 0:
            return {
                "status": "duplicates",
                "message": f"Found {duplicates} duplicate IDs. Will use first occurrence.",
                "total": total_count,
                "unique": unique_count,
                "duplicates": duplicates
            }
        elif valid_count < total_count:
            missing = total_count - valid_count
            return {
                "status": "missing",
                "message": f"{missing} missing IDs. Will auto-generate for missing entries.",
                "total": total_count,
                "unique": unique_count,
                "missing": missing
            }
        else:
            return {
                "status": "perfect",
                "message": f"Perfect! {unique_count} unique IDs",
                "total": total_count,
                "unique": unique_count
            }
    
    # ========================================================================
    # TEXT PREPROCESSING
    # ========================================================================
    
    def preprocess_texts(self, texts: List[str], method: str, 
                        custom_settings: Dict[str, Any], session_id: str) -> Tuple[List[str], Dict[str, Any]]:
        """Process texts with specified method and return results with metadata"""
        
        start_time = time.time()
        
        # Log preprocessing start
        self.logger.log_activity("preprocessing_started", session_id, {
            "method": method,
            "text_count": len(texts),
            "custom_settings": custom_settings if method == "custom" else None
        })
        
        # Process texts based on method
        if method == "none":
            processed_texts = [str(text) if text else "" for text in texts]
            details = "No preprocessing applied"
            
        elif method == "basic":
            processed_texts = [self.text_processor.basic_cleaning(text) for text in texts]
            details = "Basic cleaning: URLs, emails, whitespace normalization"
            
        elif method == "advanced":
            processed_texts = [self.text_processor.advanced_cleaning(text) for text in texts]
            details = "Advanced cleaning: URLs, emails, stopwords, punctuation, short words"
            
        elif method == "custom":
            processed_texts = [
                self.text_processor.advanced_cleaning(
                    text,
                    remove_stopwords=custom_settings.get('remove_stopwords', True),
                    remove_punctuation=custom_settings.get('remove_punctuation', True),
                    min_length=custom_settings.get('min_length', 2)
                ) for text in texts
            ]
            details = f"Custom: {custom_settings}"
            
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        # Filter out empty texts
        original_count = len(processed_texts)
        processed_texts = [text.strip() for text in processed_texts if text.strip() and len(text.strip()) > 2]
        filtered_count = len(processed_texts)
        
        # Calculate statistics
        original_stats = self.text_processor.analyze_text_quality(texts)
        processed_stats = self.text_processor.analyze_text_quality(processed_texts)
        
        processing_time = time.time() - start_time
        
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
    # CLUSTERING
    # ========================================================================
    
    def get_clustering_parameters(self, text_count: int, session_id: str) -> Dict[str, Any]:
        """Get optimal clustering parameters for given dataset size"""
        
        params = self.config.get_optimal_parameters(text_count)
        
        self.logger.log_activity("parameters_suggested", session_id, {
            "text_count": text_count,
            "suggested_params": params
        })
        
        return params
    
    def run_clustering(self, texts: List[str], params: Dict[str, Any], 
                      session_id: str) -> Dict[str, Any]:
        """Run clustering on texts with given parameters"""
        
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
            
            # Setup model
            model_setup_time = time.time()
            model_ready = self.clustering_model.setup_model(params)
            setup_duration = time.time() - model_setup_time
            
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
        """Extract detailed information about each cluster"""
        
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
    # EXPORT AND ANALYTICS
    # ========================================================================
    
    def export_results(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame, 
                      text_column: str, id_column: str = None, session_id: str = "") -> pd.DataFrame:
        """Create comprehensive results dataframe for export"""
        
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
        
        # Get original texts for comparison
        original_texts = original_data[text_column].tolist()[:len(texts)]
        
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
        """Get comprehensive session summary"""
        
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
    # UTILITY METHODS
    # ========================================================================
    
    @property
    def data_service(self):
        """For compatibility with frontend expecting data_service"""
        return self
    
    @property 
    def preprocessing_service(self):
        """For compatibility with frontend expecting preprocessing_service"""
        return self
    
    @property
    def export_service(self):
        """For compatibility with frontend expecting export_service"""
        return self
    
    def get_text_column_suggestions(self, df: pd.DataFrame, session_id: str) -> List[str]:
        """Identify potential text columns in the dataframe"""
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
        """Analyze texts and provide preprocessing recommendations"""
        
        text_stats = self.text_processor.analyze_text_quality(texts)
        
        recommendations = {
            "suggested_method": "basic",
            "reasons": [],
            "text_analysis": text_stats
        }
        
        # Analyze text characteristics to suggest best preprocessing
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
        """Create a comprehensive text summary report"""
        
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