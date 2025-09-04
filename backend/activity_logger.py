# backend/activity_logger.py
import os, json, logging
from datetime import datetime
from typing import Dict, Any, Optional

class ActivityLogger:
    """Structured JSON logger for analytics + debugging."""

    def __init__(self, log_file: str = "clustery_activity.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("ClusteryActivity")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            log_dir = os.path.dirname(log_file) or "logs"
            os.makedirs(log_dir, exist_ok=True)

            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)

            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_activity(
        self, activity_type: str, session_id: str, details: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        self.logger.info(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "activity_type": activity_type,
            "details": details
        }))

    def log_error(
        self, error_type: str, session_id: str, error_details: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.logger.error(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "error_type": error_type,
            "error_details": error_details,
            "metadata": metadata or {}
        }))

    def log_performance(
        self, operation: str, session_id: str, duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.logger.info(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {}
        }))
