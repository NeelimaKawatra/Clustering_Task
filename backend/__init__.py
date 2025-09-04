# backend/__init__.py
from typing import Dict, Any
from datetime import datetime

from .activity_logger import ActivityLogger
from .preprocessing_backend import FastTextProcessor

from .data_loading_backend import DataLoadingBackend
from .preprocessing_backend import PreprocessingBackend
from .clustering_backend import ClusteringBackend
from .results_backend import ResultsBackend
from .finetuning_backend import FineTuningBackend, get_finetuning_backend

class ClusteryBackend:
    """Facade that composes service backends and preserves old public API."""
    def __init__(self, log_file: str = "clustery_activity.log"):
        self.logger = ActivityLogger(log_file)
        self.text_processor = FastTextProcessor()

        self.data_loader = DataLoadingBackend(self.logger, self.text_processor)
        self.preprocessing = PreprocessingBackend(self.logger, self.text_processor)
        self.clustering = ClusteringBackend(self.logger)
        self.results = ResultsBackend(self.logger)
        self.finetuning = get_finetuning_backend()  # available if you need to use it here

        self.session_data: Dict[str, Any] = {}

    # ----- Session -----
    def start_session(self, session_id: str, user_info: Dict[str, Any] | None = None) -> None:
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
            "user_info": user_info, "timestamp": datetime.now().isoformat()
        })

    def track_activity(self, session_id: str, activity_type: str, data: Dict[str, Any]):
        # In-memory (can keep datetime), but logger needs JSON-safe payloads
        now = datetime.now()
        if session_id in self.session_data:
            self.session_data[session_id]["activities"].append({
                "type": activity_type,
                "data": data,
                "timestamp": now
            })
        safe_details = {"timestamp": now.isoformat(), **data}
        self.logger.log_activity(activity_type, session_id, safe_details)

    # ----- Data loading -----
    def load_data(self, file_path: str, session_id: str):
        ok, df, msg = self.data_loader.load_data(file_path, session_id)
        if session_id in self.session_data and ok:
            self.session_data[session_id]["data_loaded"] = True
        return ok, df, msg

    def validate_columns(self, df, text_column, id_column, session_id):
        return self.data_loader.validate_columns(df, text_column, id_column, session_id)

    def get_text_column_suggestions(self, df, session_id):
        return self.data_loader.get_text_column_suggestions(df, session_id)

    # ----- Preprocessing -----
    def preprocess_texts(self, texts, method, custom_settings, session_id):
        processed, meta = self.preprocessing.preprocess_texts(texts, method, custom_settings, session_id)
        if session_id in self.session_data:
            self.session_data[session_id]["preprocessing_completed"] = True if processed else False
        return processed, meta

    def get_preprocessing_recommendations(self, texts, session_id):
        return self.preprocessing.get_preprocessing_recommendations(texts, session_id)

    # ----- Clustering -----
    def get_clustering_parameters(self, text_count, session_id):
        return self.clustering.get_clustering_parameters(text_count, session_id)

    def run_clustering(self, texts, params, session_id):
        result = self.clustering.run_clustering(texts, params, session_id)
        if session_id in self.session_data and result.get("success"):
            self.session_data[session_id]["clustering_completed"] = True
        return result

    def get_cluster_details(self, results, session_id):
        return self.clustering.get_cluster_details(results, session_id)

    # ----- Results / Export -----
    def export_results(self, clustering_results, original_data, text_column, session_id: str = ""):
        df = self.results.export_results(clustering_results, original_data, text_column, session_id)
        if session_id in self.session_data:
            self.session_data[session_id]["results_exported"] = True
        return df

    def create_essential_export(self, clustering_results, original_data, text_column, session_id: str = ""):
        return self.results.create_essential_export(clustering_results, original_data, text_column, session_id)

    def create_detailed_export(self, clustering_results, original_data, text_column, session_id: str = ""):
        return self.results.create_detailed_export(clustering_results, original_data, text_column, session_id)


    def create_summary_report(self, clustering_results, preprocessing_info, session_id: str = ""):
        return self.results.create_summary_report(clustering_results, preprocessing_info, session_id)

    def get_session_analytics(self, session_id: str):
        return self.results.get_session_analytics(self.session_data, session_id)

# simple health check
def health_check() -> bool:
    try:
        _ = ClusteryBackend()
        return True
    except Exception:
        return False
