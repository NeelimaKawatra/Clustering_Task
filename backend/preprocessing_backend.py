# backend/preprocessing_backend.py
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from .activity_logger import ActivityLogger


class FastTextProcessor:
    """Fast, dependency-light text validation and cleaning (embedded here)."""

    def __init__(self):
        self.stop_words = set([
            'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
            'this','that','these','those','i','you','he','she','it','we','they'
        ])

    def validate_text_column(self, series: pd.Series) -> Tuple[bool, str]:
        if series.empty:
            return False, "Column is empty"
        non_null = series.dropna()
        if len(non_null) == 0:
            return False, "Column contains only null values"

        total_checked = min(100, len(non_null))
        texty = sum(
            1 for v in non_null.head(total_checked)
            if isinstance(v, str) and len(v.strip()) > 5
        )
        ratio = texty / total_checked
        if ratio < 0.5:
            return False, f"Only {ratio:.1%} of entries appear to be valid text"
        return True, f"Good text column: {ratio:.1%} valid entries"

    def analyze_text_quality(self, texts) -> Dict[str, Any]:
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        else:
            texts = list(texts)

        if not texts:
            return {
                "total_texts": 0, "avg_length": 0, "avg_words": 0, "empty_texts": 0,
                "short_texts": 0, "unique_texts": 0, "sample_size": 0
            }

        sample = texts[:min(100, len(texts))]
        lengths, words = [], []
        empty, short = 0, 0
        uniq = set()

        for t in sample:
            s = str(t).strip()
            if not s:
                empty += 1
                continue
            L = len(s); W = len(s.split())
            lengths.append(L); words.append(W)
            uniq.add(s.lower())
            if L < 10:
                short += 1

        return {
            "total_texts": len(texts),
            "avg_length": float(np.mean(lengths)) if lengths else 0,
            "avg_words": float(np.mean(words)) if words else 0,
            "empty_texts": empty,
            "short_texts": short,
            "unique_texts": len(uniq),
            "sample_size": len(sample)
        }

    def basic_cleaning(self, text: str) -> str:
        if not text:
            return ""
        import re
        s = str(text).strip()
        s = re.sub(r'http\S+|www\.\S+', '', s)
        s = re.sub(r'\S+@\S+', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def advanced_cleaning(
        self, text: str, remove_stopwords: bool = True, remove_punctuation: bool = True,
        min_length: int = 2, remove_numbers: bool = True
    ) -> str:
        if not text:
            return ""
        import re
        s = self.basic_cleaning(text)
        if remove_numbers:
            s = re.sub(r'\d+', '', s)
        if remove_punctuation:
            s = re.sub(r'[^\w\s]', ' ', s)
        toks = s.lower().split()
        kept = [w for w in toks
                if len(w) >= min_length and not (remove_stopwords and w in self.stop_words)]
        return ' '.join(kept)


class PreprocessingBackend:
    def __init__(
        self,
        logger: ActivityLogger,
        text_processor: Optional[FastTextProcessor] = None
    ):
        self.logger = logger
        self.text_processor = text_processor or FastTextProcessor()

    def preprocess_texts(
        self, texts: List[str], method: str, custom_settings: Dict[str, Any], session_id: str
    ) -> Tuple[List[str], Dict[str, Any]]:
        start = time.time()
        processed, valid_idx = [], []

        for i, t in enumerate(texts):
            if method == "none":
                pt = str(t) if t else ""
            elif method == "basic":
                pt = self.text_processor.basic_cleaning(t)
            elif method == "advanced":
                pt = self.text_processor.advanced_cleaning(t)
            elif method == "custom":
                pt = self.text_processor.advanced_cleaning(t, **custom_settings)
            else:
                pt = self.text_processor.basic_cleaning(t)

            if pt and pt.strip() and len(pt.strip()) > 2:
                processed.append(pt.strip())
                valid_idx.append(i)

        meta = {
            "method": method,
            "details": f"{method} preprocessing",
            "original_count": len(texts),
            "filtered_count": len(processed),
            "texts_removed": len(texts) - len(processed),
            "valid_row_indices": valid_idx,
            "processing_time": time.time() - start,
            "original_stats": self.text_processor.analyze_text_quality(texts),
            "processed_stats": self.text_processor.analyze_text_quality(processed)
        }
        return processed, meta

    def get_preprocessing_recommendations(self, texts: List[str], session_id: str) -> Dict[str, Any]:
        stats = self.text_processor.analyze_text_quality(texts)
        rec = {"suggested_method": "basic", "reasons": [], "text_analysis": stats}

        if stats["avg_length"] > 200:
            rec["suggested_method"] = "advanced"
            rec["reasons"].append("Long texts benefit from stopword removal")
        if stats["avg_words"] > 20:
            rec["suggested_method"] = "advanced"
            rec["reasons"].append("Many words suggest need for noise reduction")
        if stats["total_texts"] < 50:
            rec["suggested_method"] = "basic"
            rec["reasons"].append("Small dataset - preserve more content")

        self.logger.log_activity("preprocessing_recommendations", session_id, {
            "suggested_method": rec["suggested_method"],
            "reasons": rec["reasons"],
            "text_stats": stats
        })
        return rec
