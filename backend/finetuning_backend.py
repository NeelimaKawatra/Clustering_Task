# backend/finetuning_backend.py
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import streamlit as st

class FineTuningBackend:
    """Human-in-the-loop editing of cluster assignments and labels."""
    def __init__(self):
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.clusters: Dict[str, Dict[str, Any]] = {}
        self.entry_to_cluster: Dict[str, str] = {}
        self.next_cluster_id = 0
        self.initialized = False

    def initialize_from_clustering_results(self, clustering_results: Dict[str, Any],
                                           original_data: pd.DataFrame,
                                           text_column: str,
                                           subject_id_column: str | None = None) -> bool:
        try:
            self.entries.clear()
            self.clusters.clear()
            self.entry_to_cluster.clear()
            self.next_cluster_id = 0

            topics = clustering_results["topics"]
            texts = clustering_results["texts"]
            probs = clustering_results["probabilities"]
            row_alignment = st.session_state.get('row_alignment', list(range(len(texts))))
            clean_ids = st.session_state.get('clean_ids', [])

            for pidx, (topic, text, prob) in enumerate(zip(topics, texts, probs)):
                oidx = row_alignment[pidx] if pidx < len(row_alignment) else pidx
                entry_id = clean_ids[oidx] if oidx < len(clean_ids) else f"ID_{oidx+1:03d}"

                subject_id = None
                if subject_id_column and subject_id_column in original_data.columns and oidx < len(original_data):
                    subject_id = str(original_data[subject_id_column].iloc[oidx])

                self.entries[entry_id] = {
                    "entryID": entry_id,
                    "subjectID": subject_id,
                    "entry_text": text,
                    "original_text": original_data[text_column].iloc[oidx] if oidx < len(original_data) else text,
                    "probability": prob,
                    "original_row_index": oidx
                }
                cluster_id = f"cluster_{topic}" if topic >= 0 else "outliers"
                self.entry_to_cluster[entry_id] = cluster_id
                if topic >= 0:
                    self.next_cluster_id = max(self.next_cluster_id, topic + 1)

            tkw = clustering_results.get("metadata", {}).get("topic_keywords", {})
            for topic in set(topics):
                cid = f"cluster_{topic}" if topic >= 0 else "outliers"
                if topic >= 0 and topic in tkw:
                    kw = tkw[topic][:2]
                    cname = "_".join(kw) if kw else f"Cluster_{topic}"
                else:
                    cname = "Outliers" if topic == -1 else f"Cluster_{topic}"

                c_entries = [eid for eid, c in self.entry_to_cluster.items() if c == cid]
                self.clusters[cid] = {
                    "clusterID": cid,
                    "cluster_name": cname,
                    "entry_ids": c_entries,
                    "created_manually": False
                }

            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize fine-tuning backend: {e}")
            return False

    # Reads
    # Lookup entry by entryID (canonical)
    def getEntry(self, entryID: str) -> Optional[Dict[str, Any]]:
        """Lookup by entryID (canonical)."""
        if not self.initialized or entryID not in self.entries:
            return None
        out = self.entries[entryID].copy()
        out["clusterID"] = self.entry_to_cluster.get(entryID)
        return out

    def getAllEntriesInCluster(self, clusterID: str) -> List[str]:
        if not self.initialized or clusterID not in self.clusters: return []
        return [self.entries[eid]["entry_text"] for eid in self.clusters[clusterID]["entry_ids"] if eid in self.entries]

    def getClusterName(self, clusterID: str) -> Optional[str]:
        if not self.initialized or clusterID not in self.clusters: return None
        return self.clusters[clusterID]["cluster_name"]

    def getAllClusters(self) -> Dict[str, Dict[str, Any]]:
        return {} if not self.initialized else self.clusters.copy()

    def getAllEntries(self) -> Dict[str, Dict[str, Any]]:
        if not self.initialized: return {}
        out = {}
        for eid, data in self.entries.items():
            c = data.copy(); c["clusterID"] = self.entry_to_cluster.get(eid); out[eid] = c
        return out

    def getClusterStatistics(self, clusterID: str) -> Optional[Dict[str, Any]]:
        if not self.initialized or clusterID not in self.clusters: return None
        eids = self.clusters[clusterID]["entry_ids"]
        probs, lens = [], []
        for eid in eids:
            if eid in self.entries:
                ent = self.entries[eid]
                probs.append(ent.get("probability", 0))
                lens.append(len(ent["entry_text"]))
        return {
            "clusterID": clusterID,
            "total_entries": len(eids),
            "avg_probability": (sum(probs)/len(probs)) if probs else 0,
            "avg_text_length": (sum(lens)/len(lens)) if lens else 0,
            "min_text_length": min(lens) if lens else 0,
            "max_text_length": max(lens) if lens else 0
        }

    # Writes
    def moveEntry(self, entryID: str, clusterID: str) -> Tuple[bool, str]:
        if not self.initialized: return False, "Fine-tuning backend not initialized"
        if entryID not in self.entries: return False, f"Entry {entryID} does not exist"
        if clusterID not in self.clusters: return False, f"Cluster {clusterID} does not exist"

        cur = self.entry_to_cluster.get(entryID)
        if cur == clusterID: return True, f"Entry {entryID} is already in cluster {clusterID}"
        if cur and cur in self.clusters:
            if entryID in self.clusters[cur]["entry_ids"]:
                self.clusters[cur]["entry_ids"].remove(entryID)
        self.clusters[clusterID]["entry_ids"].append(entryID)
        self.entry_to_cluster[entryID] = clusterID
        return True, f"Successfully moved entry {entryID} to cluster {clusterID}"

    def mergeClusters(self, c1: str, c2: str, new_name: str | None = None) -> Tuple[bool, str]:
        if not self.initialized: return False, "Fine-tuning backend not initialized"
        if c1 not in self.clusters: return False, f"Cluster {c1} does not exist"
        if c2 not in self.clusters: return False, f"Cluster {c2} does not exist"
        if c1 == c2: return False, "Cannot merge cluster with itself"

        new_id = f"merged_{self.next_cluster_id}"
        self.next_cluster_id += 1
        merged_entries = self.clusters[c1]["entry_ids"] + self.clusters[c2]["entry_ids"]

        self.clusters[new_id] = {
            "clusterID": new_id,
            "cluster_name": new_name or f"Merged_{c1}_{c2}",
            "entry_ids": merged_entries,
            "created_manually": True,
            "merged_from": [c1, c2]
        }
        for eid in merged_entries:
            self.entry_to_cluster[eid] = new_id
        del self.clusters[c1]; del self.clusters[c2]
        return True, new_id

    def changeClusterName(self, clusterID: str, newName: str) -> Tuple[bool, str]:
        if not self.initialized: return False, "Fine-tuning backend not initialized"
        if clusterID not in self.clusters: return False, f"Cluster {clusterID} does not exist"
        if not newName or not newName.strip(): return False, "New name cannot be empty"
        old = self.clusters[clusterID]["cluster_name"]
        self.clusters[clusterID]["cluster_name"] = newName.strip()
        return True, f"Successfully changed cluster name from '{old}' to '{newName}'"

    def createNewCluster(self, cluster_name: str) -> Tuple[bool, str]:
        if not self.initialized: return False, "Fine-tuning backend not initialized"
        if not cluster_name or not cluster_name.strip(): return False, "Cluster name cannot be empty"
        new_id = f"manual_{self.next_cluster_id}"; self.next_cluster_id += 1
        self.clusters[new_id] = {
            "clusterID": new_id,
            "cluster_name": cluster_name.strip(),
            "entry_ids": [],
            "created_manually": True
        }
        return True, new_id

    def deleteCluster(self, clusterID: str) -> Tuple[bool, str]:
        if not self.initialized: return False, "Fine-tuning backend not initialized"
        if clusterID not in self.clusters: return False, f"Cluster {clusterID} does not exist"

        if "outliers" not in self.clusters:
            self.clusters["outliers"] = {
                "clusterID": "outliers",
                "cluster_name": "Outliers",
                "entry_ids": [],
                "created_manually": False
            }

        moved = list(self.clusters[clusterID]["entry_ids"])
        for eid in moved:
            self.entry_to_cluster[eid] = "outliers"
            self.clusters["outliers"]["entry_ids"].append(eid)

        del self.clusters[clusterID]
        return True, f"Deleted cluster {clusterID} and moved {len(moved)} entries to outliers"

    # Export
    def exportFineTunedResults(self, original_data: pd.DataFrame, text_column: str, subject_id_column: str | None = None) -> pd.DataFrame:
        if not self.initialized: return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for eid, data in self.entries.items():
            cid = self.entry_to_cluster.get(eid, "unassigned")
            cname = self.getClusterName(cid) or "Unassigned"
            rows.append({
                "entryID": eid,
                "subjectID": data.get("subjectID"),
                "entry_text": data["entry_text"],
                "original_text": data.get("original_text", data["entry_text"]),
                "clusterID": cid,
                "cluster_name": cname,
                "probability": data.get("probability", 0),
                "manually_modified": cid.startswith("manual_") or cid.startswith("merged_")
            })
        return pd.DataFrame(rows)

    def getModificationSummary(self) -> Dict[str, Any]:
        if not self.initialized: return {}
        manual = [cid for cid, c in self.clusters.items() if c.get("created_manually", False)]
        merged = [cid for cid, c in self.clusters.items() if "merged_from" in c]
        total_entries = len(self.entries)
        entries_in_manual = sum(len(self.clusters[cid]["entry_ids"]) for cid in manual)
        return {
            "total_clusters": len(self.clusters),
            "manual_clusters_created": len(manual),
            "clusters_merged": len(merged),
            "total_entries": total_entries,
            "entries_in_manual_clusters": entries_in_manual,
            "modification_percentage": (entries_in_manual / total_entries * 100) if total_entries > 0 else 0
        }

# Optional singleton
_finetuning_backend = None
def get_finetuning_backend() -> FineTuningBackend:
    global _finetuning_backend
    if _finetuning_backend is None:
        _finetuning_backend = FineTuningBackend()
    return _finetuning_backend