# backend/data_loading_backend.py
import os
import streamlit as st
import pandas as pd
import pandas.api.types as ptypes
from typing import Dict, Any, Tuple, List, Optional

from .activity_logger import ActivityLogger
from .preprocessing_backend import FastTextProcessor


class DataLoadingBackend:
    def __init__(self, logger: ActivityLogger, text_processor: FastTextProcessor):
        self.logger = logger
        self.text_processor = text_processor

    def load_data(self, file_path: str, session_id: str) -> Tuple[bool, pd.DataFrame, str]:
        try:
            self.logger.log_activity("file_upload_started", session_id, {
                "file_path": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            })

            size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            if size > 1024 * 1024:
                with st.spinner(f"Loading file ({size/1024/1024:.1f}MB)..."):
                    df = self._load_file_by_type(file_path)
            else:
                df = self._load_file_by_type(file_path)

            # ✅ Make DataFrame Arrow-friendly to avoid PyArrow conversion errors
            df = self._make_arrow_safe(df)

            if len(df) > 300:
                self.logger.log_activity("file_size_warning", session_id, {
                    "original_rows": len(df),
                    "action": "truncate_to_300"
                })
                orig = len(df)
                df = df.head(300)
                st.warning(f"File truncated to 300 rows for performance (was {orig} rows)")

            if len(df) < 10:
                msg = f"File too small: {len(df)} rows. Need at least 10 rows."
                self.logger.log_error("file_too_small", session_id, msg)
                return False, pd.DataFrame(), msg

            self.logger.log_activity("file_loaded_successfully", session_id, {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_kb": float(df.memory_usage(deep=True).sum()) / 1024.0
            })
            return True, df, f"File loaded successfully"

        except Exception as e:
            msg = f"Error loading file: {e}"
            self.logger.log_error("file_load_error", session_id, msg)
            return False, pd.DataFrame(), msg

    def _load_file_by_type(self, file_path: str) -> pd.DataFrame:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(file_path)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        raise ValueError(f"Unsupported file format: {ext}")

    def validate_columns(self, df: pd.DataFrame, text_column: str, id_column: Optional[str], session_id: str) -> Dict[str, Any]:
        result = {
            "text_column_valid": False,
            "id_column_analysis": {},
            "text_quality": {},
            "recommendations": []
        }

        if id_column and id_column != "Auto-generate IDs":
            result["id_column_analysis"] = self._analyze_id_column(df[id_column])
        else:
            result["id_column_analysis"] = {
                "status": "perfect",
                "message": "Auto-generated IDs will be created",
                "total": len(df),
                "unique": len(df)
            }

        is_valid, msg = self.text_processor.validate_text_column(df[text_column])
        result["text_column_valid"] = is_valid
        result["text_column_message"] = msg
        if is_valid:
            result["text_quality"] = self.text_processor.analyze_text_quality(df[text_column].tolist())

        self.logger.log_activity("column_validation", session_id, {
            "text_column": text_column,
            "text_valid": is_valid,
            "text_quality": result["text_quality"]
        })
        return result

    def _analyze_id_column(self, series: pd.Series) -> Dict[str, Any]:
        total = len(series)
        non_null = series.dropna()
        if len(non_null) == 0:
            return {"status": "empty", "message": "Column is empty", "total": total, "unique": 0}
        if pd.api.types.is_numeric_dtype(series):
            return {
                "status": "numeric",
                "message": "Good numeric ID column",
                "total": total,
                "unique": int(non_null.nunique())
            }
        return {
            "status": "non_numeric",
            "message": "Non-numeric column selected",
            "total": total,
            "unique": int(non_null.nunique())
        }

    def get_text_column_suggestions(self, df: pd.DataFrame, session_id: str) -> List[str]:
        cols = []
        for col in df.columns:
            ok, _ = self.text_processor.validate_text_column(df[col])
            if ok:
                cols.append(col)
        self.logger.log_activity("text_column_suggestions", session_id, {
            "suggested_columns": cols,
            "total_columns": len(df.columns)
        })
        return cols

    # --------------------------- Arrow-safe casting --------------------------- #
    def _make_arrow_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dtypes so Streamlit -> PyArrow conversion does not error:
        - Replace common string sentinels with NA
        - Cast mostly-parseable datetimes to datetime64
        - Cast mostly-numeric objects to Int64/Float64 (nullable)
        - Cast remaining objects to pandas 'string' dtype
        - Normalize ints/floats/bools to nullable pandas dtypes
        """
        df = df.copy()
        NON_NUMERIC_TOKENS = {"Non-Text", "NA", "N/A", "na", "n/a", "null", "None", ""}

        for col in df.columns:
            s = df[col]

            # Replace string sentinels with NA to help numeric/datetime coercion
            if ptypes.is_object_dtype(s):
                df[col] = s.replace(list(NON_NUMERIC_TOKENS), pd.NA)

            s = df[col]  # refresh reference

            # Datetime: if ~90% parseable, cast to datetime
            if ptypes.is_object_dtype(s):
                try_dt = pd.to_datetime(s, errors="coerce", utc=False)
                if try_dt.notna().mean() >= 0.9:
                    df[col] = try_dt
                    continue

            # Mostly numeric? cast to Int64/Float64
            if ptypes.is_object_dtype(s):
                s_num = pd.to_numeric(s, errors="coerce")
                if s_num.notna().mean() >= 0.9:
                    if (s_num.dropna() % 1 == 0).all():
                        df[col] = s_num.astype("Int64")   # nullable int
                    else:
                        df[col] = s_num.astype("Float64") # nullable float
                    continue
                else:
                    # treat as text
                    df[col] = s.astype("string")
                    continue

            # Normalize standard dtypes to Arrow-friendly nullable ones
            if ptypes.is_integer_dtype(s):
                df[col] = s.astype("Int64")
            elif ptypes.is_float_dtype(s):
                df[col] = s.astype("Float64")
            elif ptypes.is_bool_dtype(s):
                df[col] = s.astype("boolean")
            elif ptypes.is_string_dtype(s) or ptypes.is_object_dtype(s):
                df[col] = s.astype("string")

        return df
