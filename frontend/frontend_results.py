import streamlit as st

def tab_results(backend_available):
    """Tab: Results Visualization and Export using backend services"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "results"})
    
    #st.header("📊 Clustering Results")

    
    
   # Add this at the beginning of tab_results function, after the track_activity call:

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
    

    # only get clutering results
    #results = st.session_state.clustering_results

    
    # Prefer Fine-tuning snapshot if available; else use original clustering results
    results = st.session_state.get("finetuning_results") or st.session_state.clustering_results

    

    stats = results["statistics"]
    confidence = results["confidence_analysis"]
    performance = results["performance"]
    
    # Results overview
    st.subheader("📈 Overview")

    st.markdown("---")
    st.caption(f"Source: {'Fine-tuning' if st.session_state.get('finetuning_results') else 'Clustering'}")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🗂️ Total Clusters", stats['n_clusters'])
    with col2:
        st.metric("✅ Clustered Texts", stats['clustered'])
    with col3:
        st.metric("❓ Outliers", stats['outliers'])
    with col4:
        st.metric("📈 Success Rate", f"{stats['success_rate']:.1f}%")
    
    # Performance metrics
    with st.expander("⚡ Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{performance['total_time']:.2f}s")
        with col2:
            st.metric("Setup Time", f"{performance['setup_time']:.2f}s")
        with col3:
            st.metric("Clustering Time", f"{performance['clustering_time']:.2f}s")
    
    # Confidence analysis
    st.subheader("🎯 Confidence Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        high_pct = (confidence['high_confidence'] / stats['total_texts']) * 100
        st.metric("🟢 High Confidence", f"{confidence['high_confidence']}", f"{high_pct:.1f}%")
        st.caption("Probability ≥ 0.7")
    
    with col2:
        med_pct = (confidence['medium_confidence'] / stats['total_texts']) * 100
        st.metric("🟡 Medium Confidence", f"{confidence['medium_confidence']}", f"{med_pct:.1f}%")
        st.caption("Probability 0.3-0.7")
    
    with col3:
        low_pct = (confidence['low_confidence'] / stats['total_texts']) * 100
        st.metric("🔴 Low Confidence", f"{confidence['low_confidence']}", f"{low_pct:.1f}%")
        st.caption("Probability < 0.3")
    
    
    
    #######################################################
    st.subheader("📝 Cluster Details")
    #######################################################

    # Cluster details (single list: folder-style header + details content)
    cluster_details = st.session_state.backend.get_cluster_details(results, st.session_state.session_id)

    def _render_cluster(cid: int, d: dict):
        keywords = d.get("keywords") or []
        name = keywords[0] if keywords else (f"cluster_{cid}" if cid != -1 else "Outliers")
        size = d.get("size", 0)

        with st.expander(f"📁 {name} ({size} entries)"):
            col1, col2 = st.columns([2, 1])

            with col1:
                #if keywords:
                #    st.write("**🔤 Top Keywords:**")
                #    st.write(", ".join(keywords))

                st.write("**📄 Sample Texts:**")
                for text, prob in d.get("top_texts", [])[:5]:
                    badge = "🟢" if prob >= 0.7 else ("🟡" if prob >= 0.3 else "🔴")
                    short = text[:150] + ("..." if len(text) > 150 else "")
                    st.write(f"{badge} {short} *(conf: {prob:.2f})*")

            with col2:
                st.metric("Avg Confidence", f"{d.get('avg_confidence', 0.0):.2f}")
                st.metric("High Confidence", d.get("high_confidence_count", 0))
                st.metric("Cluster Size", size)

    regular_ids = sorted([cid for cid in cluster_details.keys() if cid != -1])
    for cid in regular_ids:
        _render_cluster(cid, cluster_details[cid])

    if -1 in cluster_details and cluster_details[-1].get("size", 0) > 0:
        _render_cluster(-1, cluster_details[-1])


    

    
   # Export section with dual views
    st.markdown("---")
    st.subheader("Export Results")

    # Export view selector
    export_view = st.radio(
        "Choose export view:",
        ["Summary View (Essential columns only)", "Detailed View (All columns)"],
        horizontal=True
    )

    # Get the appropriate dataframe
    if export_view == "Summary View (Essential columns only)":
        # Create summary-view export
        results_df = st.session_state.backend.create_essential_export(
            results,
            st.session_state.df,
            st.session_state.text_column,
            st.session_state.session_id
        )
        export_type = "summary"
        filename_suffix = "_summary"
    else:
        # Create detailed-view export
        results_df = st.session_state.backend.create_detailed_export(
            results,
            st.session_state.df,
            st.session_state.text_column,
            st.session_state.session_id
        )
        export_type = "detailed"
        filename_suffix = "_detailed"

    # Show preview of selected export
    st.write(f"**Preview of {export_type}-view export data:**")
    st.dataframe(results_df, use_container_width=True, hide_index=True, height=400)

    # Show column information
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
            st.write("• preprocessed_text: Text after preprocessing steps")
            st.write("• cluster_id: Assigned cluster number (-1 = outlier)")
            st.write("• cluster_label: Descriptive cluster name based on keywords")
            st.write("• confidence_score: Confidence score of cluster assignment (0-1)")
            st.write("• confidence_level: High, Medium, or Low based on confidence score")


    # Export buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = results_df.to_csv(index=False)
        if st.download_button(
            f"Download {export_type.title()} Results CSV",
            csv_data,
            f"clustering_results{filename_suffix}.csv",
            "text/csv",
            use_container_width=True
        ):
            # Track export
            st.session_state.backend.track_activity(st.session_state.session_id, "export", {
                "export_type": f"csv_{export_type}",
                "export_info": {"rows": len(results_df), "format": "csv", "view": export_type}
            })

    with col2:
        # generate a clustery report
        summary_report = st.session_state.backend.create_summary_report(
            results,
            st.session_state.preprocessing_settings,
            st.session_state.session_id
        )
        
        if st.download_button(
            "Download Clustery Report",
            summary_report,
            "clustering_summary.txt",
            "text/plain",
            use_container_width=True
        ):
            st.session_state.backend.track_activity(st.session_state.session_id, "export", {
                "export_type": "summary_report",
                "export_info": {"format": "text"}
            })

    with col3:
        if st.button("Start New Analysis", use_container_width=True):
            from utils.session_state import reset_analysis
            reset_analysis()
            st.success("Ready for new analysis! Go to Data Loading tab.")
            st.rerun()

