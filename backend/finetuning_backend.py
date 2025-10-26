from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import streamlit as st

class FineTuningBackend:
    """
    Human-in-the-loop editing of the Entry-Cluster Dataset.

    The Entry-Cluster Dataset contains all text entries with their entryIDs,
    subjectIDs, and current cluster assignments (clusterIDs). This class
    provides high-level methods to manipulate the dataset.
    """
    def __init__(self):
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.clusters: Dict[str, Dict[str, Any]] = {}
        self.entry_to_cluster: Dict[str, str] = {}
        self.next_cluster_id = 0
        self.initialized = False
    
    def reset(self):
        """Reset the backend to uninitialized state."""
        self.entries.clear()
        self.clusters.clear()
        self.entry_to_cluster.clear()
        self.next_cluster_id = 0
        self.initialized = False

    def initialize_from_clustering_results(self, clustering_results: Dict[str, Any],
                                           original_data: pd.DataFrame,
                                           entry_column: str,
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
                entry_id = clean_ids[oidx] if oidx < len(clean_ids) else f"{oidx+1:03d}"

                subject_id = None
                if subject_id_column and subject_id_column in original_data.columns and oidx < len(original_data):
                    subject_id = str(original_data[subject_id_column].iloc[oidx])

                self.entries[entry_id] = {
                    "entryID": entry_id,
                    "subjectID": subject_id,
                    "entry_text": text,
                    "original_text": original_data[entry_column].iloc[oidx] if oidx < len(original_data) else text,
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

    # =========================================================================
    # INFORMATION FUNCTIONS
    # =========================================================================

    def getEntry(self, entryID: str) -> Optional[Dict[str, Any]]:
        """Lookup by entryID (canonical)."""
        if not self.initialized or entryID not in self.entries:
            return None
        out = self.entries[entryID].copy()
        out["clusterID"] = self.entry_to_cluster.get(entryID)
        return out

    def getAllEntriesInCluster(self, clusterID: str) -> List[str]:
        """Get all text entries in a specific cluster."""
        if not self.initialized or clusterID not in self.clusters: 
            return []
        return [self.entries[eid]["entry_text"] for eid in self.clusters[clusterID]["entry_ids"] if eid in self.entries]

    def getClusterName(self, clusterID: str) -> Optional[str]:
        """Get the display name of a cluster."""
        if not self.initialized or clusterID not in self.clusters: 
            return None
        return self.clusters[clusterID]["cluster_name"]

    def getAllClusters(self) -> Dict[str, Dict[str, Any]]:
        """Get all clusters with their metadata."""
        return {} if not self.initialized else self.clusters.copy()

    def getAllEntries(self) -> Dict[str, Dict[str, Any]]:
        """Get all entries with their current cluster assignments."""
        if not self.initialized: 
            return {}
        out = {}
        for eid, data in self.entries.items():
            c = data.copy()
            c["clusterID"] = self.entry_to_cluster.get(eid)
            out[eid] = c
        return out

    def getClusterStatistics(self, clusterID: str) -> Optional[Dict[str, Any]]:
        """Get statistical information about a cluster."""
        if not self.initialized or clusterID not in self.clusters: 
            return None
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

    def getEntriesByCluster(self, contains: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Return a mapping: clusterID -> [entryIDs], optionally filtered by substring.
        Filtering is case-insensitive and operates on 'entry_text'. Does not mutate state.
        """
        if not self.initialized:
            return {}

        needle = (contains or "").strip().lower()
        result: Dict[str, List[str]] = {}

        if not needle:
            # Fast path: return original lists (copy to avoid accidental mutation)
            for cid, c in self.clusters.items():
                result[cid] = list(c.get("entry_ids", []))
            return result

        for cid, c in self.clusters.items():
            filtered = []
            for eid in c.get("entry_ids", []):
                ent = self.entries.get(eid)
                if not ent:
                    continue
                txt = (ent.get("entry_text") or "").lower()
                if needle in txt:
                    filtered.append(eid)
            result[cid] = filtered
        return result

    def getEntriesByClusterFiltered(self, contains: Optional[str] = None,
                                confidence_level: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Return clusterID -> [entryIDs], filtered by:
        - contains: case-insensitive substring match on entry_text (optional)
        - confidence_level: one of {'low','medium','high'} (optional)

        Thresholds:
        Low    : prob < 0.3
        Medium : 0.3 <= prob < 0.7
        High   : prob >= 0.7

        This is read-only and does not mutate internal state.
        """
        if not self.initialized:
            return {}

        needle = (contains or "").strip().lower()
        want_level = (confidence_level or "").strip().lower()
        if want_level not in {"low", "medium", "high", ""}:
            want_level = ""

        def _level(p: float) -> str:
            if p is None:
                return "low"  # be conservative if missing
            return "high" if p >= 0.7 else ("medium" if p >= 0.3 else "low")

        out: Dict[str, List[str]] = {}
        for cid, c in self.clusters.items():
            selected: List[str] = []
            for eid in c.get("entry_ids", []):
                ent = self.entries.get(eid)
                if not ent:
                    continue

                # keyword filter
                if needle:
                    txt = (ent.get("entry_text") or "").lower()
                    if needle not in txt:
                        continue

                # confidence filter
                if want_level:
                    lvl = _level(ent.get("probability", 0.0))
                    if lvl != want_level:
                        continue

                selected.append(eid)
            out[cid] = selected
        return out


    # =========================================================================
    # MANIPULATION FUNCTIONS
    # =========================================================================

    def is_duplicate_name(self, name: str, exclude_cluster_id: str = None) -> bool:
        """Check if cluster name already exists (case-insensitive)."""
        if not self.initialized:
            return False
        
        name_lower = name.lower().strip()
        for cid, cluster in self.clusters.items():
            if cid != exclude_cluster_id and cluster["cluster_name"].lower() == name_lower:
                return True
        return False

    def moveEntry(self, entryID: str, target_clusterID: str) -> Tuple[bool, str]:
        """Move entry from one cluster to another."""
        if not self.initialized:
            return False, "Backend not initialized"
        
        if entryID not in self.entries:
            return False, f"Entry {entryID} not found"
        
        if target_clusterID not in self.clusters:
            return False, f"Cluster {target_clusterID} not found"
        
        # Get current cluster
        old_cluster_id = self.entry_to_cluster.get(entryID)
        
        # If already in target cluster, nothing to do
        if old_cluster_id == target_clusterID:
            return True, f"Entry already in {self.clusters[target_clusterID]['cluster_name']}"
        
        # Remove from old cluster
        if old_cluster_id and old_cluster_id in self.clusters:
            if entryID in self.clusters[old_cluster_id]["entry_ids"]:
                self.clusters[old_cluster_id]["entry_ids"].remove(entryID)
        
        # Add to new cluster
        self.entry_to_cluster[entryID] = target_clusterID
        if entryID not in self.clusters[target_clusterID]["entry_ids"]:
            self.clusters[target_clusterID]["entry_ids"].append(entryID)
        
        target_name = self.clusters[target_clusterID]['cluster_name']
        return True, f"Moved entry to {target_name}"

    def mergeClusters(self, clusterID1: str, clusterID2: str, new_name: str = None) -> Tuple[bool, str]:
        """Merge two clusters into one."""
        if not self.initialized:
            return False, "Backend not initialized"
        
        if clusterID1 not in self.clusters:
            return False, f"Cluster {clusterID1} not found"
        
        if clusterID2 not in self.clusters:
            return False, f"Cluster {clusterID2} not found"
        
        if clusterID1 == clusterID2:
            return False, "Cannot merge cluster with itself"
        
        # Create new merged cluster ID
        merged_id = f"merged_{self.next_cluster_id}"
        self.next_cluster_id += 1
        
        # Generate name for merged cluster
        name1 = self.clusters[clusterID1]["cluster_name"]
        name2 = self.clusters[clusterID2]["cluster_name"]
        
        if new_name and new_name.strip():
            final_name = new_name.strip()
        else:
            final_name = f"{name1}_{name2}"
        
        # Check for duplicate name
        if self.is_duplicate_name(final_name):
            final_name = f"{final_name}_{merged_id}"
        
        # Combine entries from both clusters
        all_entries = (self.clusters[clusterID1]["entry_ids"] + 
                      self.clusters[clusterID2]["entry_ids"])
        
        # Remove duplicates while preserving order
        unique_entries = []
        seen = set()
        for eid in all_entries:
            if eid not in seen:
                unique_entries.append(eid)
                seen.add(eid)
        
        # Create merged cluster
        self.clusters[merged_id] = {
            "clusterID": merged_id,
            "cluster_name": final_name,
            "entry_ids": unique_entries,
            "created_manually": True,
            "merged_from": [clusterID1, clusterID2]
        }
        
        # Update entry-to-cluster mappings
        for eid in unique_entries:
            self.entry_to_cluster[eid] = merged_id
        
        # Remove old clusters
        del self.clusters[clusterID1]
        del self.clusters[clusterID2]
        
        return True, final_name

    def changeClusterName(self, clusterID: str, new_name: str) -> Tuple[bool, str]:
        """Change the name of a cluster."""
        if not self.initialized:
            return False, "Backend not initialized"
        
        if clusterID not in self.clusters:
            return False, f"Cluster {clusterID} not found"
        
        if not new_name or not new_name.strip():
            return False, "Name cannot be empty"
        
        new_name = new_name.strip()
        
        # Check for duplicate name (excluding current cluster)
        if self.is_duplicate_name(new_name, clusterID):
            return False, f"Name '{new_name}' already exists"
        
        old_name = self.clusters[clusterID]["cluster_name"]
        self.clusters[clusterID]["cluster_name"] = new_name
        
        return True, f"Renamed '{old_name}' to '{new_name}'"

    def createNewCluster(self, name: str) -> Tuple[bool, str]:
        """Create a new empty cluster."""
        if not self.initialized:
            return False, "Backend not initialized"
        
        if not name or not name.strip():
            return False, "Name cannot be empty"
        
        name = name.strip()
        
        # Check for duplicate name
        if self.is_duplicate_name(name):
            return False, f"Name '{name}' already exists"
        
        # Generate new cluster ID
        new_id = f"manual_{self.next_cluster_id}"
        self.next_cluster_id += 1
        
        # Create cluster
        self.clusters[new_id] = {
            "clusterID": new_id,
            "cluster_name": name,
            "entry_ids": [],
            "created_manually": True
        }
        
        return True, name

    def deleteCluster(self, clusterID: str) -> Tuple[bool, str]:
        """Delete a cluster and move its entries to outliers."""
        if not self.initialized:
            return False, "Backend not initialized"
        
        if clusterID not in self.clusters:
            return False, f"Cluster {clusterID} not found"
        
        # Don't allow deleting outliers cluster
        if clusterID == "outliers":
            return False, "Cannot delete the Outliers cluster"
        
        # Ensure outliers cluster exists
        if "outliers" not in self.clusters:
            self.clusters["outliers"] = {
                "clusterID": "outliers",
                "cluster_name": "Outliers",
                "entry_ids": [],
                "created_manually": False
            }
        
        # Move all entries to outliers
        entries_to_move = self.clusters[clusterID]["entry_ids"][:]
        cluster_name = self.clusters[clusterID]["cluster_name"]
        
        for eid in entries_to_move:
            # Update mapping
            self.entry_to_cluster[eid] = "outliers"
            # Add to outliers if not already there
            if eid not in self.clusters["outliers"]["entry_ids"]:
                self.clusters["outliers"]["entry_ids"].append(eid)
        
        # Delete the cluster
        del self.clusters[clusterID]
        
        return True, f"Deleted '{cluster_name}' and moved {len(entries_to_move)} entries to Outliers"

    def recompute_confidence_with_final_assignments(self, clustering_backend) -> dict:
        """
        Recompute confidence scores AFTER manual edits, using the SAME metric
        as initial clustering:
        - Transform texts with the already-fitted vectorizer + reducer
        - Compute per-final-cluster centroid in reduced space
        - Confidence_i = max(0.1, 1 - dist_i / max_dist), Euclidean, normalized globally

        Updates self.entries[entryID]["probability"] in place and returns a summary.
        """
        import numpy as np

        if not self.initialized:
            return {"error": "Fine-tuning backend not initialized"}

        # Reuse fitted components (no refit!)
        model = getattr(clustering_backend, "model", None)
        if model is None or model.vectorizer is None or model.reducer is None:
            return {"error": "Clustering model/vectorizer/reducer not available. Run clustering first."}

        # 1) Collect texts and ids in a stable order
        entry_ids = list(self.entries.keys())
        texts = [self.entries[eid]["entry_text"] for eid in entry_ids]
        cluster_ids = [self.entry_to_cluster.get(eid) for eid in entry_ids]

        # 2) Embed -> reduce (fitted vectorizer/reducer)
        X = model.vectorizer.transform(texts)           # sparse
        Xr = model.reducer.transform(X)                 # (n_docs x n_components)

        # 3) Build centroids for final clusters (skip 'outliers')
        cid_to_indices = {}
        for i, cid in enumerate(cluster_ids):
            cid_to_indices.setdefault(cid, []).append(i)

        centroids = {}
        for cid, idxs in cid_to_indices.items():
            if cid == "outliers" or len(idxs) == 0:
                continue
            centroids[cid] = Xr[idxs].mean(axis=0)     # numpy vector

        # 4) Distances to own-cluster centroid
        dists = np.zeros(len(entry_ids), dtype=float)
        for i, cid in enumerate(cluster_ids):
            if cid in centroids:
                v = Xr[i]
                c = centroids[cid]
                # Euclidean distance in reduced space
                d = np.linalg.norm(v - c)
            else:
                # No centroid (e.g., outliers or empty cluster) => force low confidence
                d = np.inf
            dists[i] = d

        # Handle corner cases
        finite_mask = np.isfinite(dists)
        if not np.any(finite_mask):
            # No valid centroids; set all to minimum confidence
            for eid in entry_ids:
                self.entries[eid]["probability"] = 0.1
            return {
                "avg_confidence": 0.1,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": len(entry_ids)
            }

        # Normalize as before: 1 - d / d_max, clip at 0.1
        d_max = np.max(dists[finite_mask])
        if d_max <= 0:
            d_max = 1.0

        confidences = []
        for i, d in enumerate(dists):
            if np.isfinite(d):
                conf = max(0.1, 1.0 - (d / d_max))
            else:
                conf = 0.1
            self.entries[entry_ids[i]]["probability"] = float(conf)
            confidences.append(conf)

        # Summary (same level thresholds used in export)
        high = sum(1 for c in confidences if c >= 0.7)
        med  = sum(1 for c in confidences if 0.3 <= c < 0.7)
        low  = sum(1 for c in confidences if c < 0.3)
        avg  = float(np.mean(confidences)) if confidences else 0.0

        return {
            "avg_confidence": avg,
            "high_confidence": high,
            "medium_confidence": med,
            "low_confidence": low
        }

    # =========================================================================
    # EXPORT AND REPORTING
    # =========================================================================

    def exportFineTunedResults(self, original_data: pd.DataFrame, entry_column: str, subject_id_column: str | None = None) -> pd.DataFrame:
        """Export fine-tuned results as DataFrame."""
        if not self.initialized: 
            return pd.DataFrame()
        
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
        """Get summary of modifications made during fine-tuning."""
        if not self.initialized: 
            return {}
        
        manual_clusters = [cid for cid, c in self.clusters.items() if c.get("created_manually", False)]
        merged_clusters = [cid for cid, c in self.clusters.items() if "merged_from" in c]
        
        total_entries = len(self.entries)
        entries_in_manual = sum(len(self.clusters[cid]["entry_ids"]) for cid in manual_clusters)
        
        return {
            "total_clusters": len(self.clusters),
            "manual_clusters_created": len(manual_clusters),
            "clusters_merged": len(merged_clusters),
            "total_entries": total_entries,
            "entries_in_manual_clusters": entries_in_manual,
            "modification_percentage": (entries_in_manual / total_entries * 100) if total_entries > 0 else 0
        }


# =========================================================================
# SINGLETON ACCESSOR
# =========================================================================

# Optional singleton
_finetuning_backend = None

def get_finetuning_backend() -> FineTuningBackend:
    """Get the global fine-tuning backend instance."""
    import streamlit as st
    
    # Try to get from session state first (survives reruns)
    if 'finetuning_backend_instance' not in st.session_state:
        st.session_state.finetuning_backend_instance = FineTuningBackend()
    
    return st.session_state.finetuning_backend_instance