from typing import Dict, Any, List
import pandas as pd
import streamlit as st
from .activity_logger import ActivityLogger

class ResultsBackend:
    def __init__(self, logger: ActivityLogger):
        self.logger = logger

    def export_results(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame,
                       entry_column: str, session_id: str = "") -> pd.DataFrame:
        if not clustering_results.get("success"):
            raise ValueError("Cannot export unsuccessful clustering results")

        topics = clustering_results["topics"]
        probs = clustering_results["probabilities"]
        processed_texts = clustering_results["texts"]

        row_alignment = st.session_state.get('row_alignment', list(range(len(processed_texts))))
        original_texts = st.session_state.get('original_texts', [])
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

            # Use the resolved subject ID column; fall back to entryID only if missing
            uid_col = user_sel.get('id_column_choice') or st.session_state.get('subjectID')
            if uid_col in original_data.columns:
                row['subjectID'] = original_data[uid_col].iloc[oidx]
            else:
                row['subjectID'] = (
                    original_data['entryID'].iloc[oidx]
                    if 'entryID' in original_data.columns else (oidx + 1)
                )

            user_entry_col = user_sel.get('entry_column_choice', entry_column)

            # Store original text under a fixed column name
            if oidx < len(original_texts):
                row['original_text'] = original_texts[oidx]
            else:
                row['original_text'] = str(original_data[user_entry_col].iloc[oidx]) if user_entry_col in original_data.columns else "N/A"

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
                
                uid_col = user_sel.get('id_column_choice') or st.session_state.get('subjectID')
                if uid_col in original_data.columns:
                    row['subjectID'] = original_data[uid_col].iloc[oidx]
                else:
                    row['subjectID'] = (
                        original_data['entryID'].iloc[oidx]
                        if 'entryID' in original_data.columns else (oidx + 1)
                    )

                user_entry_col = user_sel.get('entry_column_choice', entry_column)

                if oidx < len(original_texts):
                    row['original_text'] = original_texts[oidx]
                else:
                    row['original_text'] = str(original_data[user_entry_col].iloc[oidx]) if user_entry_col in original_data.columns else "N/A"

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
                              entry_column: str, session_id: str = "") -> pd.DataFrame:
        detailed = self.export_results(clustering_results, original_data, entry_column, session_id)
        cols = ['entryID', 'original_text', 'cluster_id', 'cluster_label']
        return detailed[cols].copy()

    def create_detailed_export(self, clustering_results: Dict[str, Any], original_data: pd.DataFrame,
                               entry_column: str, session_id: str = "") -> pd.DataFrame:
        detailed = self.export_results(clustering_results, original_data, entry_column, session_id)
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

    def create_summary_report(
        self,
        clustering_results: Dict[str, Any],
        preprocessing_settings: Dict[str, Any],
        session_id: str = ""
    ) -> str:
        """Generate a plain-text summary report of clustering results."""
        if not clustering_results.get("success"):
            return "Clustering was not successful. No summary report available."

        stats = clustering_results.get("statistics", {})
        confidence = clustering_results.get("confidence_analysis", {})
        performance = clustering_results.get("performance", {})
        metadata = clustering_results.get("metadata", {})

        lines = []
        lines.append("=== Clustery Summary Report ===")
        lines.append(f"Session ID: {session_id}")
        lines.append("")
        lines.append("=== Preprocessing Settings ===")
        for k, v in preprocessing_settings.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("=== Clustering Statistics ===")
        for k, v in stats.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("=== Confidence Analysis ===")
        for k, v in confidence.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("=== Performance ===")
        for k, v in performance.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("=== Topic Keywords ===")
        for cid, kws in metadata.get("topic_keywords", {}).items():
            lines.append(f"Cluster {cid}: {', '.join(kws)}")
        
        self.logger.log_activity("create_summary_report", session_id, {
            "rows": stats.get("total_texts", 0),
            "clusters": stats.get("n_clusters", 0)
        })

        return "\n".join(lines)
