# tabs/finetuning.py - Updated to use proper backend API

import streamlit as st
from backend_finetuning import get_finetuning_backend
import pandas as pd

def tab_e_finetuning(backend_available):
    """Tab E: Human-in-the-Loop Fine-tuning with proper backend API"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "finetuning"})
    
    st.header("Fine-tuning: Manual Cluster Adjustment")
    st.caption("Manually adjust your clustering results using standardized backend functions.")

    # Check prerequisites
    if not st.session_state.get('clustering_results') or not st.session_state.clustering_results.get("success", False):
        st.error("Please complete Clustering first!")
        st.info("Go to the Clustering tab and run the clustering analysis to see results here.")
        return
    
    # Initialize backend if needed
    if not _initialize_backend():
        st.error("Failed to initialize fine-tuning backend")
        return
    
    backend = get_finetuning_backend()
    
    # Show summary statistics
    show_finetuning_summary(backend)
    
    # Main interface
    show_cluster_management_interface(backend)
    
    # Entry manipulation interface  
    show_entry_management_interface(backend)
    
    # Export section
    show_export_interface(backend)

def _initialize_backend() -> bool:
    """Initialize fine-tuning backend from clustering results"""
    
    if 'finetuning_initialized' in st.session_state:
        return True
    
    backend = get_finetuning_backend()
    
    # Get required data
    clustering_results = st.session_state.clustering_results
    df = st.session_state.df
    text_column = st.session_state.text_column
    
    # Get subject ID column from user selections
    user_selections = st.session_state.get('user_selections', {})
    subject_id_column = None
    
    if not user_selections.get('id_is_auto_generated', True):
        subject_id_column = user_selections.get('id_column_choice')
    
    # Initialize backend
    success = backend.initialize_from_clustering_results(
        clustering_results, df, text_column, subject_id_column
    )
    
    if success:
        st.session_state.finetuning_initialized = True
        return True
    
    return False

def show_finetuning_summary(backend):
    """Show summary of current clustering state"""
    
    with st.expander("Clustering Summary", expanded=True):
        all_clusters = backend.getAllClusters()
        all_entries = backend.getAllEntries()
        modification_summary = backend.getModificationSummary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clusters", len(all_clusters))
        with col2:
            st.metric("Total Entries", len(all_entries))
        with col3:
            st.metric("Manual Clusters", modification_summary.get('manual_clusters_created', 0))
        with col4:
            mod_pct = modification_summary.get('modification_percentage', 0)
            st.metric("Modified %", f"{mod_pct:.1f}%")

def show_cluster_management_interface(backend):
    """Interface for managing clusters"""
    
    st.subheader("Cluster Management")
    
    all_clusters = backend.getAllClusters()
    
    # Cluster operations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Create New Cluster**")
        new_cluster_name = st.text_input("New cluster name", placeholder="Enter cluster name")
        if st.button("Create Cluster") and new_cluster_name.strip():
            success, result = backend.createNewCluster(new_cluster_name.strip())
            if success:
                st.success(f"Created cluster: {result}")
                st.rerun()
            else:
                st.error(result)
    
    with col2:
        st.markdown("**Merge Clusters**")
        cluster_ids = list(all_clusters.keys())
        if len(cluster_ids) >= 2:
            cluster1 = st.selectbox("First cluster", cluster_ids, key="merge_cluster1")
            cluster2 = st.selectbox("Second cluster", cluster_ids, key="merge_cluster2")
            merge_name = st.text_input("Merged cluster name (optional)", key="merge_name")
            
            if st.button("Merge Clusters") and cluster1 != cluster2:
                success, result = backend.mergeClusters(cluster1, cluster2, merge_name or None)
                if success:
                    st.success(f"Merged into cluster: {result}")
                    st.rerun()
                else:
                    st.error(result)
    
    # Display clusters
    st.markdown("**Current Clusters**")
    
    for cluster_id, cluster_data in all_clusters.items():
        with st.expander(f"🗂️ {cluster_data['cluster_name']} ({len(cluster_data['entry_ids'])} entries)", expanded=False):
            
            # Cluster name editing
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                new_name = st.text_input(
                    "Cluster name", 
                    value=cluster_data['cluster_name'],
                    key=f"name_{cluster_id}"
                )
                
                if new_name != cluster_data['cluster_name']:
                    if st.button(f"Update Name", key=f"update_name_{cluster_id}"):
                        success, message = backend.changeClusterName(cluster_id, new_name)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            with col2:
                # Show cluster statistics
                stats = backend.getClusterStatistics(cluster_id)
                if stats:
                    st.metric("Avg Confidence", f"{stats['avg_probability']:.2f}")
            
            with col3:
                # Delete cluster option
                if st.button(f"Delete", key=f"delete_{cluster_id}"):
                    success, message = backend.deleteCluster(cluster_id)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            # Show entries in cluster
            entries_text = backend.getAllEntriesInCluster(cluster_id)
            if entries_text:
                st.markdown("**Entries:**")
                for i, text in enumerate(entries_text[:5]):  # Show first 5
                    st.text(f"• {text[:100]}{'...' if len(text) > 100 else ''}")
                
                if len(entries_text) > 5:
                    st.caption(f"... and {len(entries_text) - 5} more entries")

def show_entry_management_interface(backend):
    """Interface for managing individual entries"""
    
    st.subheader("Entry Management")
    
    all_entries = backend.getAllEntries()
    all_clusters = backend.getAllClusters()
    
    if not all_entries:
        st.info("No entries available")
        return
    
    # Entry search and selection
    entry_ids = list(all_entries.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Find Entry**")
        
        # Search by text content
        search_text = st.text_input("Search in entry text", placeholder="Type to search...")
        
        if search_text:
            matching_entries = []
            for entry_id, entry_data in all_entries.items():
                if search_text.lower() in entry_data['entry_text'].lower():
                    matching_entries.append(entry_id)
            
            if matching_entries:
                selected_entry = st.selectbox("Matching entries", matching_entries, format_func=lambda x: f"{x}: {all_entries[x]['entry_text'][:50]}...")
            else:
                st.info("No entries match your search")
                selected_entry = None
        else:
            # Show all entries
            selected_entry = st.selectbox("Select entry", entry_ids[:50], format_func=lambda x: f"{x}: {all_entries[x]['entry_text'][:50]}...")  # Limit to first 50 for performance
    
    with col2:
        if 'selected_entry' in locals() and selected_entry:
            st.markdown("**Entry Details**")
            
            entry_data = backend.getEntry(selected_entry)
            if entry_data:
                st.text(f"Entry ID: {entry_data['entryID']}")
                st.text(f"Subject ID: {entry_data.get('subjectID', 'N/A')}")
                st.text(f"Current Cluster: {entry_data.get('clusterID', 'Unassigned')}")
                st.text(f"Confidence: {entry_data.get('probability', 0):.2f}")
                
                st.markdown("**Text:**")
                st.text_area("", value=entry_data['entry_text'], height=100, disabled=True, key=f"text_{selected_entry}")
                
                # Move entry
                st.markdown("**Move Entry**")
                cluster_options = list(all_clusters.keys())
                current_cluster = entry_data.get('clusterID')
                
                if current_cluster in cluster_options:
                    current_index = cluster_options.index(current_cluster)
                else:
                    current_index = 0
                
                target_cluster = st.selectbox(
                    "Move to cluster", 
                    cluster_options,
                    index=current_index,
                    format_func=lambda x: f"{all_clusters[x]['cluster_name']} ({len(all_clusters[x]['entry_ids'])} entries)",
                    key=f"move_{selected_entry}"
                )
                
                if target_cluster != current_cluster:
                    if st.button(f"Move to {all_clusters[target_cluster]['cluster_name']}", key=f"move_btn_{selected_entry}"):
                        success, message = backend.moveEntry(selected_entry, target_cluster)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

def show_export_interface(backend):
    """Interface for exporting fine-tuned results"""
    
    st.subheader("Export Fine-tuned Results")
    
    modification_summary = backend.getModificationSummary()
    
    # Show modification summary
    with st.expander("Modification Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Manual Clusters Created", modification_summary.get('manual_clusters_created', 0))
        with col2:
            st.metric("Clusters Merged", modification_summary.get('clusters_merged', 0))
        with col3:
            st.metric("Entries Modified", modification_summary.get('entries_in_manual_clusters', 0))
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV", use_container_width=True):
            df = st.session_state.df
            text_column = st.session_state.text_column
            user_selections = st.session_state.get('user_selections', {})
            subject_id_column = None
            
            if not user_selections.get('id_is_auto_generated', True):
                subject_id_column = user_selections.get('id_column_choice')
            
            export_df = backend.exportFineTunedResults(df, text_column, subject_id_column)
            
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "Download Fine-tuned Results CSV",
                csv_data,
                "finetuned_clustering_results.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("📋 Export Summary Report", use_container_width=True):
            report = create_finetuning_report(backend)
            st.download_button(
                "Download Summary Report",
                report,
                "finetuning_summary_report.txt",
                "text/plain",
                use_container_width=True
            )
    
    with col3:
        if st.button("🔄 Reset to Original", use_container_width=True):
            if st.session_state.get('confirm_reset'):
                # Actually reset
                st.session_state.finetuning_initialized = False
                del st.session_state['confirm_reset']
                st.success("Reset to original clustering results")
                st.rerun()
            else:
                # Ask for confirmation
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")

def create_finetuning_report(backend) -> str:
    """Create a detailed text report of fine-tuning changes"""
    
    all_clusters = backend.getAllClusters()
    modification_summary = backend.getModificationSummary()
    
    report = f"""
FINE-TUNING SUMMARY REPORT
=========================

Generated: {st.session_state.get('session_id', 'unknown')}

OVERVIEW:
- Total Clusters: {len(all_clusters)}
- Manual Clusters Created: {modification_summary.get('manual_clusters_created', 0)}
- Clusters Merged: {modification_summary.get('clusters_merged', 0)}
- Total Entries: {modification_summary.get('total_entries', 0)}
- Entries in Manual Clusters: {modification_summary.get('entries_in_manual_clusters', 0)}
- Modification Percentage: {modification_summary.get('modification_percentage', 0):.1f}%

CLUSTER DETAILS:
"""
    
    for cluster_id, cluster_data in all_clusters.items():
        cluster_name = cluster_data['cluster_name']
        entry_count = len(cluster_data['entry_ids'])
        is_manual = cluster_data.get('created_manually', False)
        
        report += f"\n{cluster_name} (ID: {cluster_id})\n"
        report += f"  - Entries: {entry_count}\n"
        report += f"  - Type: {'Manual' if is_manual else 'Original'}\n"
        
        if 'merged_from' in cluster_data:
            report += f"  - Merged from: {', '.join(cluster_data['merged_from'])}\n"
        
        # Show statistics
        stats = backend.getClusterStatistics(cluster_id)
        if stats:
            report += f"  - Avg Confidence: {stats['avg_probability']:.3f}\n"
            report += f"  - Avg Text Length: {stats['avg_text_length']:.1f} chars\n"
        
        # Show sample entries
        entries_text = backend.getAllEntriesInCluster(cluster_id)
        if entries_text:
            report += f"  - Sample entries:\n"
            for i, text in enumerate(entries_text[:3]):
                report += f"    • {text[:100]}{'...' if len(text) > 100 else ''}\n"
            if len(entries_text) > 3:
                report += f"    ... and {len(entries_text) - 3} more\n"
    
    report += "\n" + "="*50
    report += "\nGenerated by Clustery Fine-tuning Module\n"
    
    return report

# Function testing interface for debugging
def show_function_testing_interface(backend):
    """Debug interface for testing backend functions"""
    
    st.subheader("🔧 Function Testing (Debug)")
    
    with st.expander("Test Backend Functions", expanded=False):
        test_type = st.selectbox(
            "Function to test",
            [
                "getEntry",
                "getAllEntriesInCluster", 
                "getClusterName",
                "moveEntry",
                "changeClusterName",
                "createNewCluster"
            ]
        )
        
        if test_type == "getEntry":
            all_entries = backend.getAllEntries()
            if all_entries:
                entry_id = st.selectbox("Entry ID", list(all_entries.keys())[:10])
                if st.button("Test getEntry"):
                    result = backend.getEntry(entry_id)
                    st.json(result)
        
        elif test_type == "getAllEntriesInCluster":
            all_clusters = backend.getAllClusters()
            if all_clusters:
                cluster_id = st.selectbox("Cluster ID", list(all_clusters.keys()))
                if st.button("Test getAllEntriesInCluster"):
                    result = backend.getAllEntriesInCluster(cluster_id)
                    st.write(f"Found {len(result)} entries:")
                    for i, text in enumerate(result[:5]):
                        st.text(f"{i+1}. {text[:100]}...")
        
        elif test_type == "getClusterName":
            all_clusters = backend.getAllClusters()
            if all_clusters:
                cluster_id = st.selectbox("Cluster ID", list(all_clusters.keys()))
                if st.button("Test getClusterName"):
                    result = backend.getClusterName(cluster_id)
                    st.write(f"Cluster name: {result}")
        
        elif test_type == "moveEntry":
            all_entries = backend.getAllEntries()
            all_clusters = backend.getAllClusters()
            
            if all_entries and all_clusters:
                col1, col2 = st.columns(2)
                with col1:
                    entry_id = st.selectbox("Entry ID", list(all_entries.keys())[:10])
                with col2:
                    target_cluster = st.selectbox("Target Cluster", list(all_clusters.keys()))
                
                if st.button("Test moveEntry"):
                    success, message = backend.moveEntry(entry_id, target_cluster)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        elif test_type == "changeClusterName":
            all_clusters = backend.getAllClusters()
            if all_clusters:
                col1, col2 = st.columns(2)
                with col1:
                    cluster_id = st.selectbox("Cluster ID", list(all_clusters.keys()))
                with col2:
                    new_name = st.text_input("New Name", value=f"Test_{cluster_id}")
                
                if st.button("Test changeClusterName"):
                    success, message = backend.changeClusterName(cluster_id, new_name)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        elif test_type == "createNewCluster":
            new_cluster_name = st.text_input("New Cluster Name", value="Test_Cluster")
            if st.button("Test createNewCluster"):
                success, result = backend.createNewCluster(new_cluster_name)
                if success:
                    st.success(f"Created cluster with ID: {result}")
                else:
                    st.error(result)
            