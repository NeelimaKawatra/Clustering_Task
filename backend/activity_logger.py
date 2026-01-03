# backend/activity_logger.py
"""
Professional activity logger for Clustery - Focused on debugging value
No emojis, no duplicates, only actionable information
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional


class ActivityLogger:
    """Minimal, high-signal logging for debugging"""
    
    def __init__(self, log_file: str = "clustery_activity.log"):
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Single file for ALL users
        self.all_users_file = os.path.join(self.log_dir, "all_users.json")
        self.user_journeys: Dict[str, Dict[str, Any]] = self._load_all_users()
        
        # Track what we've already logged (prevent duplicates)
        self.logged_events: Dict[str, set] = {}
        
        # Setup traditional logger
        self.logger = logging.getLogger("ClusteryActivity")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Activity log - human readable
            activity_handler = logging.FileHandler(
                os.path.join(self.log_dir, "activity.log"),
                encoding='utf-8'
            )
            activity_handler.setLevel(logging.INFO)
            
            # Error log - errors only
            error_handler = logging.FileHandler(
                os.path.join(self.log_dir, "errors.log"),
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            
            # Console - errors only
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            
            # Simple, readable format
            formatter = logging.Formatter(
                "%(asctime)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            activity_handler.setFormatter(formatter)
            error_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(activity_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)
    
    # =========================================================================
    # DUPLICATE PREVENTION
    # =========================================================================
    
    def _should_log(self, session_id: str, event_key: str) -> bool:
        """Check if we should log this event (prevent duplicates)"""
        if session_id not in self.logged_events:
            self.logged_events[session_id] = set()
        
        if event_key in self.logged_events[session_id]:
            return False  # Already logged
        
        self.logged_events[session_id].add(event_key)
        return True
    
    def _clear_event(self, session_id: str, event_key: str):
        """Clear an event so it can be logged again (e.g., after file change)"""
        if session_id in self.logged_events:
            self.logged_events[session_id].discard(event_key)
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    def _load_all_users(self) -> Dict[str, Dict[str, Any]]:
        """Load all users from single JSON file"""
        if os.path.exists(self.all_users_file):
            try:
                with open(self.all_users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                backup_file = self.all_users_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.all_users_file, backup_file)
                print(f"Warning: Corrupted log backed up to {backup_file}")
                return {}
            except Exception:
                return {}
        return {}
    
    def _save_all_users(self):
        """Save ALL users to single JSON file"""
        try:
            temp_file = self.all_users_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_journeys, f, indent=2, ensure_ascii=False)
            
            if os.path.exists(self.all_users_file):
                os.replace(temp_file, self.all_users_file)
            else:
                os.rename(temp_file, self.all_users_file)
        except Exception as e:
            print(f"Error saving logs: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # =========================================================================
    # USER JOURNEY TRACKING (NO EMOJIS - CLEAR STATUS)
    # =========================================================================
    
    def _init_user_journey(self, session_id: str):
        """Initialize a new user journey"""
        if session_id not in self.user_journeys:
            self.user_journeys[session_id] = {
                "user_id": session_id,
                "started_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "stages": {
                    "session_start": {"status": "COMPLETED", "time": datetime.now().isoformat()},
                    "file_upload": {"status": "PENDING"},
                    "column_selection": {"status": "PENDING"},
                    "preprocessing": {"status": "PENDING"},
                    "clustering": {"status": "PENDING"},
                    "finetuning": {"status": "PENDING"},
                    "export": {"status": "PENDING"}
                },
                "errors": [],
                "warnings": []
            }
            self._save_all_users()
    
    def _update_stage(self, session_id: str, stage: str, status: str, details: Dict[str, Any] = None):
        """Update a stage in user's journey"""
        self._init_user_journey(session_id)
        
        self.user_journeys[session_id]["stages"][stage] = {
            "status": status,
            "time": datetime.now().isoformat(),
            "details": details or {}
        }
        self.user_journeys[session_id]["updated_at"] = datetime.now().isoformat()
        self._save_all_users()
    
    def _add_error(self, session_id: str, stage: str, error: str, details: Dict[str, Any] = None):
        """Add error to user's journey"""
        self._init_user_journey(session_id)
        
        self.user_journeys[session_id]["errors"].append({
            "stage": stage,
            "error": error,
            "time": datetime.now().isoformat(),
            "details": details or {}
        })
        
        self._update_stage(session_id, stage, "FAILED", {"error": error, **(details or {})})
    
    def _add_warning(self, session_id: str, stage: str, warning: str, details: Dict[str, Any] = None):
        """Add warning (non-fatal issue)"""
        self._init_user_journey(session_id)
        
        self.user_journeys[session_id]["warnings"].append({
            "stage": stage,
            "warning": warning,
            "time": datetime.now().isoformat(),
            "details": details or {}
        })
        self._save_all_users()
    
    # =========================================================================
    # PUBLIC API - MEANINGFUL LOGGING
    # =========================================================================
    
    def log_activity(self, activity_type: str, session_id: str, details: Dict[str, Any], user_id: Optional[str] = None):
        """Log user activity with duplicate prevention"""
        
        self._init_user_journey(session_id)
        
        # =====================================================================
        # SESSION START
        # =====================================================================
        if activity_type == "session_started":
            if self._should_log(session_id, "session_started"):
                self.logger.info(f"[{session_id}] Session started")
        
        # =====================================================================
        # FILE UPLOAD
        # =====================================================================
        elif activity_type == "file_upload_started":
            filename = details.get("file_path", "unknown").split("/")[-1].split("\\")[-1]
            size = details.get("file_size", 0)
            
            # Clear previous file-related events (new file = fresh start)
            self._clear_event(session_id, "file_loaded")
            self._clear_event(session_id, "column_validated")
            
            if self._should_log(session_id, f"file_upload:{filename}"):
                self.logger.info(f"[{session_id}] Uploading file: {filename} ({size} bytes)")
        
        elif activity_type == "file_loaded_successfully":
            rows = details.get("rows", 0)
            cols = details.get("columns", 0)
            columns = details.get("column_names", [])
            
            self._update_stage(session_id, "file_upload", "COMPLETED", {
                "rows": rows,
                "columns": cols,
                "column_names": columns
            })
            
            if self._should_log(session_id, "file_loaded"):
                self.logger.info(
                    f"[{session_id}] File loaded successfully: "
                    f"{rows} rows × {cols} columns | "
                    f"Columns: {', '.join(columns)}"
                )
        
        # =====================================================================
        # COLUMN VALIDATION - THE PROBLEM AREA
        # =====================================================================
        elif activity_type == "column_validation":
            text_col = details.get("entry_column", "?")
            text_valid = details.get("text_valid", False)
            quality = details.get("text_quality", {})
            
            # Only log ONCE per column validation attempt
            event_key = f"column_validation:{text_col}:{text_valid}"
            
            if not self._should_log(session_id, event_key):
                return  # Already logged this exact validation
            
            if text_valid:
                # SUCCESS
                self._update_stage(session_id, "column_selection", "COMPLETED", {
                    "text_column": text_col,
                    "total_texts": quality.get("total_texts", 0),
                    "avg_words": quality.get("avg_words", 0),
                    "avg_chars": quality.get("avg_chars", 0),
                    "min_words": quality.get("min_words", 0),
                    "max_words": quality.get("max_words", 0)
                })
                
                self.logger.info(
                    f"[{session_id}] Column '{text_col}' validated successfully | "
                    f"{quality.get('total_texts', 0)} texts | "
                    f"Avg length: {quality.get('avg_words', 0):.1f} words | "
                    f"Range: {quality.get('min_words', 0)}-{quality.get('max_words', 0)} words"
                )
            
            else:
                # FAILURE - LOG DETAILED REASON
                validation_errors = details.get("validation_errors", [])
                available_columns = details.get("available_columns", [])
                column_types = details.get("column_types", {})
                
                # Build detailed error message
                error_msg = f"Column '{text_col}' validation failed"
                error_details = {
                    "selected_column": text_col,
                    "available_columns": available_columns,
                    "validation_errors": validation_errors
                }
                
                # Add type information if available
                if text_col in column_types:
                    detected_type = column_types[text_col]
                    error_details["detected_type"] = detected_type
                    error_msg += f" | Detected type: {detected_type} (expected: string/text)"
                
                # Add quality issues if present
                if quality:
                    error_details["quality_issues"] = quality
                    if quality.get("empty_count", 0) > 0:
                        error_msg += f" | {quality['empty_count']} empty values"
                    if quality.get("avg_words", 0) < 1:
                        error_msg += f" | Avg words too low: {quality['avg_words']:.2f}"
                
                # Add specific validation errors
                if validation_errors:
                    error_msg += f" | Errors: {'; '.join(validation_errors)}"
                
                self._add_error(session_id, "column_selection", error_msg, error_details)
                
                self.logger.error(
                    f"[{session_id}] {error_msg} | "
                    f"Available columns: {', '.join(available_columns)}"
                )
        
        # =====================================================================
        # PREPROCESSING
        # =====================================================================
        elif activity_type == "preprocessing":
            method = details.get("method", "unknown")
            original = details.get("original_count", 0)
            final = details.get("final_count", 0)
            filtered = original - final
            
            self._update_stage(session_id, "preprocessing", "COMPLETED", {
                "method": method,
                "original_count": original,
                "final_count": final,
                "filtered_count": filtered,
                "filter_rate": f"{(filtered/original*100):.1f}%" if original > 0 else "0%"
            })
            
            if self._should_log(session_id, f"preprocessing:{method}"):
                self.logger.info(
                    f"[{session_id}] Preprocessing completed ({method}) | "
                    f"{original} → {final} texts ({filtered} filtered, {filtered/original*100:.1f}%)"
                )
        
        # =====================================================================
        # CLUSTERING
        # =====================================================================
        elif activity_type == "clustering_started":
            text_count = details.get("text_count", 0)
            params = details.get("parameters", {})
            
            self._update_stage(session_id, "clustering", "IN_PROGRESS", {
                "text_count": text_count,
                "parameters": params
            })
            
            if self._should_log(session_id, "clustering_started"):
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                self.logger.info(
                    f"[{session_id}] Clustering started | "
                    f"{text_count} texts | Parameters: {param_str}"
                )
        
        # =====================================================================
        # EXPORT
        # =====================================================================
        elif activity_type == "export":
            export_type = details.get("export_type", "unknown")
            rows = details.get("export_info", {}).get("rows", 0)
            
            self._update_stage(session_id, "export", "COMPLETED", {
                "type": export_type,
                "rows": rows
            })
            
            if self._should_log(session_id, f"export:{export_type}"):
                self.logger.info(f"[{session_id}] Exported {rows} rows as {export_type}")
        
        # =====================================================================
        # IGNORE NOISE
        # =====================================================================
        elif activity_type in ["tab_visit", "preprocessing_recommendations", 
                              "parameters_suggested", "unified_reset"]:
            pass  # Silently ignore
    
    def log_error(self, error_type: str, session_id: str, error_details: str, metadata: Optional[Dict[str, Any]] = None):
        """Log errors (no duplicates)"""
        
        stage_map = {
            "file_load_error": "file_upload",
            "clustering_failed": "clustering",
            "preprocessing_error": "preprocessing",
            "export_error": "export"
        }
        
        stage = stage_map.get(error_type, "unknown")
        
        # Only log if not already logged
        event_key = f"error:{error_type}:{error_details}"
        if not self._should_log(session_id, event_key):
            return
        
        self._add_error(session_id, stage, error_details, metadata)
        
        # Create detailed error message
        error_msg = f"[{session_id}] ERROR in {stage}: {error_details}"
        if metadata:
            metadata_str = " | ".join([f"{k}={v}" for k, v in metadata.items()])
            error_msg += f" | {metadata_str}"
        
        self.logger.error(error_msg)
    
    def log_performance(self, operation: str, session_id: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        
        if operation == "clustering_completed":
            clusters = metadata.get("clusters_found", "?")
            success_rate = metadata.get("success_rate", 0)
            
            self._update_stage(session_id, "clustering", "COMPLETED", {
                "duration": duration,
                "clusters_found": clusters,
                "success_rate": success_rate,
                "metadata": metadata
            })
            
            if self._should_log(session_id, "clustering_completed"):
                self.logger.info(
                    f"[{session_id}] Clustering completed in {duration:.2f}s | "
                    f"{clusters} clusters found | Success rate: {success_rate:.1f}%"
                )
        
        # Save to performance log
        perf_log = os.path.join(self.log_dir, "performance.log")
        with open(perf_log, 'a', encoding='utf-8') as f:
            metadata_str = json.dumps(metadata or {}, ensure_ascii=False)
            f.write(f"{datetime.now().isoformat()} | {session_id} | {operation} | {duration:.2f}s | {metadata_str}\n")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_user_journey(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's journey"""
        return self.user_journeys.get(session_id)
    
    def print_journey(self, session_id: str):
        """Print a user's journey (for debugging)"""
        journey = self.get_user_journey(session_id)
        if not journey:
            print(f"No journey found for {session_id}")
            return
        
        print(f"\n{'='*100}")
        print(f"USER JOURNEY: {session_id}")
        print(f"Started: {journey['started_at']}")
        print(f"Updated: {journey['updated_at']}")
        print(f"{'='*100}\n")
        
        print(f"{'Stage':<20} {'Status':<12} {'Details'}")
        print(f"{'-'*100}")
        
        for stage, data in journey['stages'].items():
            status = data.get('status', 'PENDING')
            details_dict = data.get('details', {})
            
            # Format details based on stage
            if stage == "file_upload" and details_dict:
                cols = details_dict.get('column_names', [])
                details_str = f"{details_dict.get('rows', '?')} rows × {details_dict.get('columns', '?')} cols | Columns: {', '.join(cols[:3])}"
            elif stage == "column_selection" and details_dict:
                details_str = f"Column: {details_dict.get('text_column', '?')} | {details_dict.get('total_texts', '?')} texts | Avg: {details_dict.get('avg_words', 0):.1f} words"
            elif stage == "preprocessing" and details_dict:
                details_str = f"{details_dict.get('original_count', '?')}→{details_dict.get('final_count', '?')} texts ({details_dict.get('method', '?')}) | {details_dict.get('filter_rate', '?')} filtered"
            elif stage == "clustering" and details_dict:
                if status == "COMPLETED":
                    details_str = f"{details_dict.get('clusters_found', '?')} clusters in {details_dict.get('duration', 0):.2f}s | {details_dict.get('success_rate', 0):.1f}% success"
                elif status == "FAILED":
                    details_str = f"Error: {details_dict.get('error', 'Unknown')}"
                else:
                    details_str = f"{details_dict.get('text_count', '?')} texts"
            else:
                details_str = ""
            
            print(f"{stage:<20} {status:<12} {details_str}")
        
        if journey.get('errors'):
            print(f"\n{'='*100}")
            print(f"ERRORS ({len(journey['errors'])})")
            print(f"{'='*100}")
            for i, err in enumerate(journey['errors'], 1):
                print(f"\n{i}. [{err['stage'].upper()}] {err['error']}")
                if err.get('details'):
                    for key, val in err['details'].items():
                        print(f"   - {key}: {val}")
        
        if journey.get('warnings'):
            print(f"\n{'='*100}")
            print(f"WARNINGS ({len(journey['warnings'])})")
            print(f"{'='*100}")
            for i, warn in enumerate(journey['warnings'], 1):
                print(f"{i}. [{warn['stage']}] {warn['warning']}")
        
        print(f"\n{'='*100}\n")