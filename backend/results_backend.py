# backend/results_backend.py
from typing import Dict, Any, List
import pandas as pd
import streamlit as st
from .activity_logger import ActivityLogger
class ResultsBackend:
    def __init__(self, logger: ActivityLogger):
        self.logger = logger

    def export_results(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame,
                       text_column: str, id_column: str | None = None, session_id: str = "") -> pd.DataFrame:
        if not clustering_results.get("success"):
            raise ValueError("Cannot export unsuccessful clustering results")

        topics = clustering_results["topics"]
        probs = clustering_results["probabilities"]
        processed_texts = clustering_results["texts"]

        row_alignment = st.session_state.get('row_alignment', list(range(len(processed_texts))))
        original_texts = st.session_state.get('original_texts', [])
        clean_ids = st.session_state.get('clean_ids', [])
        user_sel = st.session_state.get('user_selections', {})

        out_rows: List[Dict[str, Any]] = []
        topic_keywords = clustering_results.get("metadata", {}).get("topic_keywords", {})

        # clustered rows
        for pidx, (topic, prob, ptxt) in enumerate(zip(topics, probs, processed_texts)):
            oidx = row_alignment[pidx] if pidx < len(row_alignment) else pidx
            row: Dict[str, Any] = {}

            row['auto_generated_id'] = clean_ids[oidx] if oidx < len(clean_ids) else f"ID_{oidx+1:03d}"

            if not user_sel.get('id_is_auto_generated', True):
                uid_col = user_sel.get('id_column_choice')
                if uid_col and uid_col in original_data.columns:
                    row[f'user_id_{uid_col}'] = str(original_data[uid_col].iloc[oidx]) if oidx < len(original_data) else "N/A"

            user_text_col = user_sel.get('text_column_choice', text_column)
            if oidx < len(original_texts):
                row[f'original_{user_text_col}'] = original_texts[oidx]
            else:
                row[f'original_{user_text_col}'] = "N/A"

            row['processed_text'] = ptxt
            row['cluster_id'] = topic
            row['confidence_score'] = prob
            row['confidence_level'] = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.3 else 'Low'

            if topic in topic_keywords and topic != -1:
                kws = topic_keywords[topic][:3]
                row['cluster_label'] = "_".join(kws)
            else:
                row['cluster_label'] = "outlier" if topic == -1 else f"cluster_{topic}"

            out_rows.append(row)

        # unclustered (filtered) rows
        total_original = len(original_data)
        processed_oidx = set(row_alignment[:len(processed_texts)])
        for oidx in range(total_original):
            if oidx not in processed_oidx:
                row: Dict[str, Any] = {}
                row['auto_generated_id'] = clean_ids[oidx] if oidx < len(clean_ids) else f"ID_{oidx+1:03d}"
                if not user_sel.get('id_is_auto_generated', True):
                    uid_col = user_sel.get('id_column_choice')
                    if uid_col and uid_col in original_data.columns:
                        row[f'user_id_{uid_col}'] = str(original_data[uid_col].iloc[oidx])

                user_text_col = user_sel.get('text_column_choice', text_column)
                if oidx < len(original_texts):
                    row[f'original_{user_text_col}'] = original_texts[oidx]
                else:
                    row[f'original_{user_text_col}'] = str(original_data[user_text_col].iloc[oidx])

                row['processed_text'] = "[FILTERED OUT - empty or too short]"
                row['cluster_id'] = None
                row['confidence_score'] = None
                row['confidence_level'] = "Not Clustered"
                row['cluster_label'] = "not_clustered"
                out_rows.append(row)

        out_rows.sort(key=lambda r: r['auto_generated_id'])
        df = pd.DataFrame(out_rows)

        self.logger.log_activity("export_results", session_id, {
            "total_rows": len(df),
            "clustered_rows": len([r for r in out_rows if r['cluster_id'] is not None]),
            "filtered_rows": len([r for r in out_rows if r['cluster_id'] is None])
        })
        return df

    def create_summary_export(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame,
                              text_column: str, id_column: str | None = None, session_id: str = "") -> pd.DataFrame:
        detailed = self.export_results(clustering_results, original_data, text_column, id_column, session_id)
        cols = ['auto_generated_id']
        cols += [c for c in detailed.columns if c.startswith('user_id_')]
        cols += [c for c in detailed.columns if c.startswith('original_')]
        cols += ['cluster_id', 'confidence_score', 'confidence_level', 'cluster_label']
        cols = [c for c in cols if c in detailed.columns]
        return detailed[cols].copy()

    def create_summary_report(self, clustering_results: Dict[str, Any],
                              preprocessing_info: Dict[str, Any], session_id: str = "") -> str:
        s = clustering_results["statistics"]
        c = clustering_results["confidence_analysis"]
        p = clustering_results["performance"]
        params = clustering_results["parameters_used"]

        report = f"""
CLUSTERY - TEXT CLUSTERING ANALYSIS REPORT
=========================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {session_id}

DATASET SUMMARY
--------------
Total Texts Analyzed: {s['total_texts']}
Preprocessing Method: {preprocessing_info.get('method', 'Unknown')}
Processing Details: {preprocessing_info.get('details', 'N/A')}

CLUSTERING RESULTS
-----------------
Clusters Found: {s['n_clusters']}
Successfully Clustered: {s['clustered']} ({s['success_rate']:.1f}%)
Outliers: {s['outliers']} ({(s['outliers']/s['total_texts']*100):.1f}%)

CONFIDENCE ANALYSIS
------------------
High Confidence (≥0.7): {c['high_confidence']} ({(c['high_confidence']/s['total_texts']*100):.1f}%)
Medium Confidence (0.3-0.7): {c['medium_confidence']} ({(c['medium_confidence']/s['total_texts']*100):.1f}%)
Low Confidence (<0.3): {c['low_confidence']} ({(c['low_confidence']/s['total_texts']*100):.1f}%)
Average Confidence: {c['avg_confidence']:.3f}

PERFORMANCE METRICS
------------------
Total Processing Time: {p['total_time']:.2f} seconds
Model Setup Time: {p['setup_time']:.2f} seconds
Clustering Time: {p['clustering_time']:.2f} seconds

PARAMETERS USED
--------------
"""
        for k, v in params.items():
            report += f"{k.replace('_', ' ').title()}: {v}\n"
        report += "\nGenerated by Clustery - Intelligent Text Clustering Tool\n"
        return report

    def get_session_analytics(self, session_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        from datetime import datetime
        if session_id not in session_data:
            return {"error": "Session not found"}
        session = session_data[session_id]
        duration = (datetime.now() - session["start_time"]).total_seconds()

        completion = {
            "data_loaded": session["data_loaded"],
            "preprocessing_completed": session["preprocessing_completed"],
            "clustering_completed": session["clustering_completed"],
            "results_exported": session["results_exported"]
        }
        pct = (sum(bool(v) for v in completion.values()) / len(completion)) * 100
        counts = {}
        for a in session["activities"]:
            t = a["type"]
            counts[t] = counts.get(t, 0) + 1

        return {
            "session_id": session_id,
            "duration_seconds": duration,
            "completion_percentage": pct,
            "completion_status": completion,
            "activity_counts": counts,
            "current_tab": session["current_tab"],
            "total_activities": len(session["activities"]),
            "user_info": session["user_info"]
        }
