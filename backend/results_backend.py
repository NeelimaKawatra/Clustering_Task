# backend/results_backend.py
from typing import Dict, Any, List
import pandas as pd
import streamlit as st
from .activity_logger import ActivityLogger
class ResultsBackend:
    def __init__(self, logger: ActivityLogger):
        self.logger = logger

    def export_results(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame,
                       text_column: str, session_id: str = "") -> pd.DataFrame:
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

            # Use entryID from df; keep numeric type if available
            row['entryID'] = (
                original_data['entryID'].iloc[oidx]
                if 'entryID' in original_data.columns else (oidx + 1)
            )

            # Stable subjectID (preferred user-selected column, else entryID)
            uid_col = user_sel.get('id_column_choice')
            if (not user_sel.get('id_is_auto_generated', True)) and uid_col and uid_col in original_data.columns:
                row['subjectID'] = original_data[uid_col].iloc[oidx]
            else:
                row['subjectID'] = row['entryID']

            user_text_col = user_sel.get('text_column_choice', text_column)

            # Store original text under a fixed column name
            if oidx < len(original_texts):
                row['original_text'] = original_texts[oidx]
            else:
                row['original_text'] = str(original_data[user_text_col].iloc[oidx]) if user_text_col in original_data.columns else "N/A"

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
                row['entryID'] = (
                    original_data['entryID'].iloc[oidx]
                    if 'entryID' in original_data.columns else (oidx + 1)
                )
                uid_col = user_sel.get('id_column_choice')
                if (not user_sel.get('id_is_auto_generated', True)) and uid_col and uid_col in original_data.columns:
                    row['subjectID'] = original_data[uid_col].iloc[oidx]
                else:
                    row['subjectID'] = row['entryID']

                user_text_col = user_sel.get('text_column_choice', text_column)

                if oidx < len(original_texts):
                    row['original_text'] = original_texts[oidx]
                else:
                    row['original_text'] = str(original_data[user_text_col].iloc[oidx]) if user_text_col in original_data.columns else "N/A"

                row['processed_text'] = None
                row['cluster_id'] = None
                row['confidence_score'] = None
                row['confidence_level'] = None
                row['cluster_label'] = None
                out_rows.append(row)

        df = pd.DataFrame(out_rows)
    
        df['entryID'] = pd.to_numeric(df['entryID'], errors='coerce')
        df = df.sort_values('entryID', kind='stable').reset_index(drop=True)
        df['entryID'] = df['entryID'].astype('Int64')

        self.logger.log_activity("export_results", session_id, {
            "total_rows": len(df),
            "clustered_rows": len([r for r in out_rows if r['cluster_id'] is not None]),
            "filtered_rows": len([r for r in out_rows if r['cluster_id'] is None])
        })
        return df

    def create_essential_export(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame,
                              text_column: str, session_id: str = "") -> pd.DataFrame:
        # Build a minimal, Arrow-friendly summary with exactly four columns
        detailed = self.export_results(clustering_results, original_data, text_column, session_id)
        cols = ['entryID', 'original_text', 'cluster_id', 'cluster_label']
        # keep only the requested columns (in order)
        return detailed[cols].copy()

    def create_detailed_export(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame,
                               text_column: str, session_id: str = "") -> pd.DataFrame:
        # Build a detailed export with all relevant columns
        detailed = self.export_results(clustering_results, original_data, text_column, session_id)
        cols = [
            'entryID',
            'subjectID',
            'original_text',
            'processed_text',
            'cluster_id',
            'cluster_label',
            'confidence_score',
            'confidence_level',
        ]
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
