import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st


class FineTuningBackend:
    """Backend for fine-tuning clustering results with standardized API"""
    
    def __init__(self):
        self.entries = {}  # entryID -> entry data
        self.clusters = {}  # clusterID -> cluster data
        self.entry_to_cluster = {}  # entryID -> clusterID
        self.next_cluster_id = 0
        self.initialized = False
    
    def initialize_from_clustering_results(self, clustering_results: Dict[str, Any], 
                                         original_data: pd.DataFrame, 
                                         text_column: str, 
                                         subject_id_column: str = None) -> bool:
        """Initialize fine-tuning backend from clustering results
        
        Args:
            clustering_results: Results from clustering analysis
            original_data: Original dataframe
            text_column: Column name containing text entries
            subject_id_column: Column name containing subject IDs (optional)
        
        Returns:
            bool: Success status
        """
        try:
            # Clear existing data
            self.entries = {}
            self.clusters = {}
            self.entry_to_cluster = {}
            self.next_cluster_id = 0
            
            # Get clustering data
            topics = clustering_results["topics"]
            texts = clustering_results["texts"]
            probabilities = clustering_results["probabilities"]
            row_alignment = st.session_state.get('row_alignment', list(range(len(texts))))
            clean_ids = st.session_state.get('clean_ids', [])
            
            # Create entries mapping
            for proc_idx, (topic, text, prob) in enumerate(zip(topics, texts, probabilities)):
                # Get original row index
                orig_row_idx = row_alignment[proc_idx] if proc_idx < len(row_alignment) else proc_idx
                
                # Create unique entryID
                entry_id = clean_ids[orig_row_idx] if orig_row_idx < len(clean_ids) else f"ID_{orig_row_idx+1:03d}"
                
                # Get subjectID if column provided
                subject_id = None
                if subject_id_column and subject_id_column in original_data.columns:
                    if orig_row_idx < len(original_data):
                        subject_id = str(original_data[subject_id_column].iloc[orig_row_idx])
                
                # Store entry data
                self.entries[entry_id] = {
                    'entryID': entry_id,
                    'subjectID': subject_id,
                    'entry_text': text,
                    'original_text': original_data[text_column].iloc[orig_row_idx] if orig_row_idx < len(original_data) else text,
                    'probability': prob,
                    'original_row_index': orig_row_idx
                }
                
                # Map entry to cluster
                cluster_id = f"cluster_{topic}" if topic >= 0 else "outliers"
                self.entry_to_cluster[entry_id] = cluster_id
                
                # Update max cluster ID
                if topic >= 0:
                    self.next_cluster_id = max(self.next_cluster_id, topic + 1)
            
            # Create cluster data
            topic_keywords = clustering_results.get("metadata", {}).get("topic_keywords", {})
            unique_topics = set(topics)
            
            for topic in unique_topics:
                cluster_id = f"cluster_{topic}" if topic >= 0 else "outliers"
                
                # Get cluster name from keywords or default
                if topic >= 0 and topic in topic_keywords:
                    keywords = topic_keywords[topic][:2]
                    cluster_name = "_".join(keywords) if keywords else f"Cluster_{topic}"
                else:
                    cluster_name = "Outliers" if topic == -1 else f"Cluster_{topic}"
                
                # Get entries in this cluster
                cluster_entries = [eid for eid, cid in self.entry_to_cluster.items() if cid == cluster_id]
                
                self.clusters[cluster_id] = {
                    'clusterID': cluster_id,
                    'cluster_name': cluster_name,
                    'entry_ids': cluster_entries,
                    'created_manually': False
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
        """Returns specific entry with entry text and clusterID
        
        Args:
            entryID: Unique identifier for the entry
            
        Returns:
            Dict with entry data including clusterID or None if not found
        """
        if not self.initialized:
            return None
            
        if entryID not in self.entries:
            return None
        
        entry_data = self.entries[entryID].copy()
        entry_data['clusterID'] = self.entry_to_cluster.get(entryID)
        return entry_data
    
    def getAllEntriesInCluster(self, clusterID: str) -> List[str]:
        """Returns list of all entry texts in a cluster
        
        Args:
            clusterID: Unique identifier for the cluster
            
        Returns:
            List of entry texts (strings)
        """
        if not self.initialized or clusterID not in self.clusters:
            return []
        
        entry_ids = self.clusters[clusterID]['entry_ids']
        entry_texts = []
        
        for entry_id in entry_ids:
            if entry_id in self.entries:
                entry_texts.append(self.entries[entry_id]['entry_text'])
        
        return entry_texts
    
    def getClusterName(self, clusterID: str) -> Optional[str]:
        """Gets name/label of cluster
        
        Args:
            clusterID: Unique identifier for the cluster
            
        Returns:
            String with cluster name or None if cluster doesn't exist
        """
        if not self.initialized or clusterID not in self.clusters:
            return None
        
        return self.clusters[clusterID]['cluster_name']
    
    def getAllClusters(self) -> Dict[str, Dict[str, Any]]:
        """Returns all clusters with their metadata
        
        Returns:
            Dictionary of clusterID -> cluster data
        """
        if not self.initialized:
            return {}
        
        return self.clusters.copy()
    
    def getAllEntries(self) -> Dict[str, Dict[str, Any]]:
        """Returns all entries with their data
        
        Returns:
            Dictionary of entryID -> entry data
        """
        if not self.initialized:
            return {}
        
        # Add clusterID to each entry
        entries_with_clusters = {}
        for entry_id, entry_data in self.entries.items():
            entry_copy = entry_data.copy()
            entry_copy['clusterID'] = self.entry_to_cluster.get(entry_id)
            entries_with_clusters[entry_id] = entry_copy
        
        return entries_with_clusters
    
    def getClusterStatistics(self, clusterID: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific cluster
        
        Args:
            clusterID: Unique identifier for the cluster
            
        Returns:
            Dictionary with cluster statistics or None if cluster doesn't exist
        """
        if not self.initialized or clusterID not in self.clusters:
            return None
        
        entry_ids = self.clusters[clusterID]['entry_ids']
        
        # Calculate statistics
        total_entries = len(entry_ids)
        probabilities = []
        text_lengths = []
        
        for entry_id in entry_ids:
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                probabilities.append(entry.get('probability', 0))
                text_lengths.append(len(entry['entry_text']))
        
        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        return {
            'clusterID': clusterID,
            'total_entries': total_entries,
            'avg_probability': avg_probability,
            'avg_text_length': avg_text_length,
            'min_text_length': min(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0
        }
    
    # =========================================================================
    # MANIPULATION FUNCTIONS
    # =========================================================================
    
    def moveEntry(self, entryID: str, clusterID: str) -> Tuple[bool, str]:
        """Move a specific entry from one cluster to another
        
        Args:
            entryID: Unique identifier for the entry
            clusterID: Target cluster ID
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.initialized:
            return False, "Fine-tuning backend not initialized"
        
        # Check if entry exists
        if entryID not in self.entries:
            return False, f"Entry {entryID} does not exist"
        
        # Check if target cluster exists
        if clusterID not in self.clusters:
            return False, f"Cluster {clusterID} does not exist"
        
        # Get current cluster
        current_cluster_id = self.entry_to_cluster.get(entryID)
        
        if current_cluster_id == clusterID:
            return True, f"Entry {entryID} is already in cluster {clusterID}"
        
        # Remove from current cluster
        if current_cluster_id and current_cluster_id in self.clusters:
            self.clusters[current_cluster_id]['entry_ids'].remove(entryID)
        
        # Add to new cluster
        self.clusters[clusterID]['entry_ids'].append(entryID)
        self.entry_to_cluster[entryID] = clusterID
        
        return True, f"Successfully moved entry {entryID} to cluster {clusterID}"
    
    def mergeClusters(self, clusterID1: str, clusterID2: str, 
                     new_name: str = None) -> Tuple[bool, str]:
        """Merges two clusters into one
        
        Args:
            clusterID1: First cluster ID
            clusterID2: Second cluster ID  
            new_name: Optional name for merged cluster
            
        Returns:
            Tuple of (success: bool, result: str)
            If successful, result is new cluster ID
            If failed, result is error description
        """
        if not self.initialized:
            return False, "Fine-tuning backend not initialized"
        
        # Check if both clusters exist
        if clusterID1 not in self.clusters:
            return False, f"Cluster {clusterID1} does not exist"
        
        if clusterID2 not in self.clusters:
            return False, f"Cluster {clusterID2} does not exist"
        
        if clusterID1 == clusterID2:
            return False, "Cannot merge cluster with itself"
        
        # Create new cluster ID
        new_cluster_id = f"merged_{self.next_cluster_id}"
        self.next_cluster_id += 1
        
        # Combine entries from both clusters
        entries1 = self.clusters[clusterID1]['entry_ids']
        entries2 = self.clusters[clusterID2]['entry_ids']
        merged_entries = entries1 + entries2
        
        # Create merged cluster
        merged_name = new_name or f"Merged_{clusterID1}_{clusterID2}"
        
        self.clusters[new_cluster_id] = {
            'clusterID': new_cluster_id,
            'cluster_name': merged_name,
            'entry_ids': merged_entries,
            'created_manually': True,
            'merged_from': [clusterID1, clusterID2]
        }
        
        # Update entry mappings
        for entry_id in merged_entries:
            self.entry_to_cluster[entry_id] = new_cluster_id
        
        # Remove old clusters
        del self.clusters[clusterID1]
        del self.clusters[clusterID2]
        
        return True, new_cluster_id
    
    def changeClusterName(self, clusterID: str, newName: str) -> Tuple[bool, str]:
        """Change the name of a cluster
        
        Args:
            clusterID: Unique identifier for the cluster
            newName: New name for the cluster
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.initialized:
            return False, "Fine-tuning backend not initialized"
        
        if clusterID not in self.clusters:
            return False, f"Cluster {clusterID} does not exist"
        
        if not newName or not newName.strip():
            return False, "New name cannot be empty"
        
        old_name = self.clusters[clusterID]['cluster_name']
        self.clusters[clusterID]['cluster_name'] = newName.strip()
        
        return True, f"Successfully changed cluster name from '{old_name}' to '{newName}'"
    
    def createNewCluster(self, cluster_name: str) -> Tuple[bool, str]:
        """Create a new empty cluster
        
        Args:
            cluster_name: Name for the new cluster
            
        Returns:
            Tuple of (success: bool, result: str)
            If successful, result is new cluster ID
        """
        if not self.initialized:
            return False, "Fine-tuning backend not initialized"
        
        if not cluster_name or not cluster_name.strip():
            return False, "Cluster name cannot be empty"
        
        new_cluster_id = f"manual_{self.next_cluster_id}"
        self.next_cluster_id += 1
        
        self.clusters[new_cluster_id] = {
            'clusterID': new_cluster_id,
            'cluster_name': cluster_name.strip(),
            'entry_ids': [],
            'created_manually': True
        }
        
        return True, new_cluster_id
    
    def deleteCluster(self, clusterID: str) -> Tuple[bool, str]:
        """Delete a cluster (moves entries to outliers)
        
        Args:
            clusterID: Unique identifier for the cluster to delete
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.initialized:
            return False, "Fine-tuning backend not initialized"
        
        if clusterID not in self.clusters:
            return False, f"Cluster {clusterID} does not exist"
        
        # Move all entries to outliers cluster
        entries_to_move = self.clusters[clusterID]['entry_ids'].copy()
        
        # Ensure outliers cluster exists
        if "outliers" not in self.clusters:
            self.clusters["outliers"] = {
                'clusterID': "outliers",
                'cluster_name': "Outliers",
                'entry_ids': [],
                'created_manually': False
            }
        
        # Move entries
        for entry_id in entries_to_move:
            self.entry_to_cluster[entry_id] = "outliers"
            self.clusters["outliers"]['entry_ids'].append(entry_id)
        
        # Delete the cluster
        del self.clusters[clusterID]
        
        return True, f"Deleted cluster {clusterID} and moved {len(entries_to_move)} entries to outliers"
    
    # =========================================================================
    # EXPORT FUNCTIONS
    # =========================================================================
    
    def exportFineTunedResults(self, original_data: pd.DataFrame, 
                              text_column: str, 
                              subject_id_column: str = None) -> pd.DataFrame:
        """Export fine-tuned results as DataFrame
        
        Args:
            original_data: Original dataset
            text_column: Name of text column  
            subject_id_column: Name of subject ID column
            
        Returns:
            DataFrame with fine-tuned clustering results
        """
        if not self.initialized:
            return pd.DataFrame()
        
        export_data = []
        
        for entry_id, entry_data in self.entries.items():
            cluster_id = self.entry_to_cluster.get(entry_id, "unassigned")
            cluster_name = self.getClusterName(cluster_id) or "Unassigned"
            
            row_data = {
                'entryID': entry_id,
                'subjectID': entry_data.get('subjectID'),
                'entry_text': entry_data['entry_text'],
                'original_text': entry_data.get('original_text', entry_data['entry_text']),
                'clusterID': cluster_id,
                'cluster_name': cluster_name,
                'probability': entry_data.get('probability', 0),
                'manually_modified': cluster_id.startswith('manual_') or cluster_id.startswith('merged_')
            }
            
            export_data.append(row_data)
        
        return pd.DataFrame(export_data)
    
    def getModificationSummary(self) -> Dict[str, Any]:
        """Get summary of modifications made during fine-tuning
        
        Returns:
            Dictionary with modification statistics
        """
        if not self.initialized:
            return {}
        
        manual_clusters = [cid for cid, cluster in self.clusters.items() 
                          if cluster.get('created_manually', False)]
        
        merged_clusters = [cid for cid, cluster in self.clusters.items() 
                          if 'merged_from' in cluster]
        
        total_entries = len(self.entries)
        entries_in_manual_clusters = sum(len(self.clusters[cid]['entry_ids']) 
                                       for cid in manual_clusters)
        
        return {
            'total_clusters': len(self.clusters),
            'manual_clusters_created': len(manual_clusters),
            'clusters_merged': len(merged_clusters),
            'total_entries': total_entries,
            'entries_in_manual_clusters': entries_in_manual_clusters,
            'modification_percentage': (entries_in_manual_clusters / total_entries * 100) if total_entries > 0 else 0
        }

# Global instance
_finetuning_backend = None

def get_finetuning_backend() -> FineTuningBackend:
    """Get global fine-tuning backend instance"""
    global _finetuning_backend
    if _finetuning_backend is None:
        _finetuning_backend = FineTuningBackend()
    return _finetuning_backend