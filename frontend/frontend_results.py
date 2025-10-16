import streamlit as st

def tab_results(backend_available):
    """Tab: Results Visualization and Export using backend services"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(
            st.session_state.session_id,
            "tab_visit",
            {"tab_name": "results"}
        )
    
    # Check prerequisites first
    if not st.session_state.get('tab_data_loading_complete', False):
        st.error("Please complete Data Loading first!")
        st.info("Go to the Data Loading tab to load and configure your data.")
        return
    
    if not st.session_state.get('tab_preprocessing_complete', False):
        st.error("Please complete Preprocessing first!")
        st.info("Go to the Preprocessing tab to process your text data.")
        return
    
    # Check if clustering is complete
    if not st.session_state.get('clustering_results') or not st.session_state.clustering_results.get("success"):
        st.error("Please complete Clustering first!")
        st.info("Go to the Clustering tab and run the clustering analysis to see results here.")
        return
    
    if not backend_available:
        st.error("Backend services not available. Please check backend installation.")
        return

    # Prefer Fine-tuning snapshot if available; else use original clustering results
    results = st.session_state.get("finetuning_results") or st.session_state.clustering_results
    stats = results["statistics"]

    # confidence = results["confidence_analysis"]  # COMMENTED OUT: No longer showing confidence scores
    performance = results["performance"]
    
    # Results overview
    st.subheader("📈 Overview")
    
    # Calculate per-cluster confidence
    def calculate_per_cluster_confidence(results_data):
        """Calculate average confidence per cluster from results data."""
        topics = results_data.get("topics", [])
        probabilities = results_data.get("probabilities", [])
        
        if not topics or not probabilities:
            return 0.0
        
        # Group probabilities by cluster
        cluster_confidences = {}
        for topic, prob in zip(topics, probabilities):
            if topic not in cluster_confidences:
                cluster_confidences[topic] = []
            cluster_confidences[topic].append(prob)
        
        # Calculate average confidence per cluster
        if not cluster_confidences:
            return 0.0
        
        avg_confidences = [sum(probs) / len(probs) for probs in cluster_confidences.values()]
        return sum(avg_confidences) / len(avg_confidences) if avg_confidences else 0.0
    
    per_cluster_confidence = calculate_per_cluster_confidence(results)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🗂️ Total Non-Empty Clusters", stats['n_clusters'])
    with col2:
        st.metric("✅ Clustered Texts", stats['clustered'])
    with col3:
        st.metric("❓ Outliers", stats['outliers'])
    with col4:
        st.metric("📈 Success Rate", f"{stats['success_rate']:.1f}%")
    
    """
    if st.session_state.get('finetuning_results'):
        from backend.finetuning_backend import get_finetuning_backend
        backend = get_finetuning_backend()
        
        if backend.initialized:
            modification_summary = backend.getModificationSummary()
            
            st.subheader("🧩 Fine-tuning Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                manual_clusters = modification_summary.get("manual_clusters_created", 0)
                st.metric("🆕 Manual Clusters Created", manual_clusters)
            
            with col2:
                merged_clusters = modification_summary.get("clusters_merged", 0)
                st.metric("🔄 Clusters Merged", merged_clusters)
            
            with col3:
                modification_pct = modification_summary.get("modification_percentage", 0)
                st.metric("📝 Entries Modified", f"{modification_pct:.1f}%")
    """
    #######################################################
    #st.subheader("📝 Cluster Details")
    #######################################################

    cluster_details = st.session_state.backend.get_cluster_details(results, st.session_state.session_id)

    def _render_cluster(cid: int, d: dict):
        keywords = d.get("keywords") or []
        name = keywords[0] if keywords else (f"cluster_{cid}" if cid != -1 else "Outliers")
        size = d.get("size", 0)

        with st.expander(f"📁 {name} ({size} entries)"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**📄 Sample Text Entries:**")
                sample_texts = d.get("top_texts", [])
                if sample_texts:
                    for i, (text, _) in enumerate(sample_texts[:5], 1):  # Ignore confidence score
                        short = text[:150] + ("..." if len(text) > 150 else "")
                        st.write(f"**{i}.** {short}")
                else:
                    # Fallback to all_texts if top_texts not available
                    all_texts = d.get("all_texts", [])
                    for i, text in enumerate(all_texts[:5], 1):
                        short = text[:150] + ("..." if len(text) > 150 else "")
                        st.write(f"**{i}.** {short}")

            with col2:
                # Show per-cluster confidence 
                avg_confidence = d.get('avg_confidence', 0.0)
                st.metric("Cluster Confidence", f"{avg_confidence:.2f}")
                # st.metric("High Confidence", d.get("high_confidence_count", 0))
                
                # Show keywords if available
                #if keywords:
                #    st.write("**Keywords:**")
                #    for keyword in keywords[:3]:
                #        st.write(f"• {keyword}")

    regular_ids = sorted([cid for cid in cluster_details.keys() if cid != -1])
    for cid in regular_ids:
        _render_cluster(cid, cluster_details[cid])

    if -1 in cluster_details and cluster_details[-1].get("size", 0) > 0:
        _render_cluster(-1, cluster_details[-1])
    
    # Export section
    st.markdown("---")
    st.subheader("Export Results")

    export_view = st.radio(
        "Choose export view:",
        ["Summary View (Essential columns only)", "Detailed View (All columns)"],
        horizontal=True
    )

    if export_view == "Summary View (Essential columns only)":
        results_df = st.session_state.backend.create_essential_export(
            results,
            st.session_state.df,
            st.session_state.entry_column,  
            st.session_state.session_id
        )
        export_type = "summary"
        filename_suffix = "_summary"
    else:
        results_df = st.session_state.backend.create_detailed_export(
            results,
            st.session_state.df,
            st.session_state.entry_column,   
            st.session_state.session_id
        )
        export_type = "detailed"
        filename_suffix = "_detailed"

    st.write(f"**Preview of {export_type}-view export data:**")
    st.dataframe(results_df, width="stretch", hide_index=True, height=400)

    with st.expander("Results Columns Information"):
        if export_type == "summary":
            st.write("**Summary View Columns:**")
            st.write("• entryID: Row numbers from your original dataset")
            st.write("• original_text: Raw text from dataset")
            st.write("• cluster_id: Assigned cluster number (-1 = outlier)")
            st.write("• cluster_label: Descriptive cluster name based on keywords")
        else:
            st.write("**Detailed Export Columns:**")
            st.write("• entryID: Row numbers from your original dataset")
            st.write("• subjectID: Subject identifier chosen by user")
            st.write("• original_text: Raw text from dataset")
            st.write("• processed_text: Text after preprocessing steps")
            st.write("• cluster_id: Assigned cluster number (-1 = outlier)")
            st.write("• cluster_label: Descriptive cluster name based on keywords")
            
            # COMMENTED OUT: Confidence score descriptions
            st.write("• confidence_score: Confidence score of cluster assignment (0-1)")
            st.write("• confidence_level: High, Medium, or Low based on confidence score")
            
            # Note about confidence scores being temporarily hidden
            #st.info("ℹ️ **Note:** Confidence scores are temporarily hidden as they reflect original clustering and don't account for fine-tuning modifications.")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = results_df.to_csv(index=False)
        if st.download_button(
            label=f"⬇️ Download {export_type.title()} Results CSV",
            data=csv_data,
            file_name=f"clustering_results{filename_suffix}.csv",
            mime="text/csv",
            width="stretch"
        ):
            st.session_state.backend.track_activity(
                st.session_state.session_id,
                "export",
                {
                    "export_type": f"csv_{export_type}",
                    "export_info": {
                        "rows": len(results_df),
                        "format": "csv",
                        "view": export_type
                    }
                }
            )

    with col2:
        summary_report = st.session_state.backend.create_summary_report(
            results,
            st.session_state.preprocessing_settings,
            st.session_state.session_id
        )
        
        if st.download_button(
            "ℹ️ Download Clustery Report",
            summary_report,
            "clustering_summary.txt",
            "text/plain",
            width="stretch"
        ):
            st.session_state.backend.track_activity(st.session_state.session_id, "export", {
                "export_type": "summary_report",
                "export_info": {"format": "text"}
            })

    with col3:
        if st.button("🔄 Start New Analysis", width="stretch"):
            from utils.session_state import reset_analysis
            reset_analysis()
            st.success("Ready for new analysis! Go to Data Loading tab.")
            st.rerun()
    
    
    # Performance metrics
    st.markdown("---")
    with st.expander("⚡ Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{performance['total_time']:.2f}s")
        with col2:
            st.metric("Setup Time", f"{performance['setup_time']:.2f}s")
        with col3:
            st.metric("Clustering Time", f"{performance['clustering_time']:.2f}s")