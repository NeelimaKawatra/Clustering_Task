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
    

    results = st.session_state.clustering_results
    stats = results["statistics"]
    confidence = results["confidence_analysis"]
    performance = results["performance"]
    
    # Results overview
    st.subheader("📈 Overview")
    
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
    
    # Get detailed cluster information from backend
    cluster_details = st.session_state.backend.get_cluster_details(results, st.session_state.session_id)
    
    # Cluster details
    st.subheader("📝 Cluster Details")
    
    # Display each cluster
    for cluster_id, details in cluster_details.items():
        if cluster_id == -1:  # Outliers
            if details['size'] > 0:
                with st.expander(f"❓ **Outliers** ({details['size']} texts)"):
                    st.write("**Texts that didn't fit into any cluster:**")
                    for i, text in enumerate(details['texts'][:10]):
                        st.write(f"• {text[:150]}{'...' if len(text) > 150 else ''}")
                    if details['size'] > 10:
                        st.write(f"... and {details['size'] - 10} more")
        else:  # Regular clusters
            keywords = ", ".join(details['keywords'])
            with st.expander(f"📋 **Cluster {cluster_id}** ({details['size']} texts) - {keywords}"):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**🔤 Top Keywords:**")
                    st.write(keywords)
                    
                    st.write("**📄 Sample Texts:**")
                    for text, prob in details['top_texts'][:5]:
                        confidence_emoji = "🟢" if prob >= 0.7 else "🟡" if prob >= 0.3 else "🔴"
                        st.write(f"{confidence_emoji} {text[:150]}{'...' if len(text) > 150 else ''} *(conf: {prob:.2f})*")
                
                with col2:
                    st.metric("Avg Confidence", f"{details['avg_confidence']:.2f}")
                    st.metric("High Confidence", details['high_confidence_count'])
                    st.metric("Cluster Size", details['size'])
    
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
        # Create summary export
        results_df = st.session_state.backend.create_summary_export(
            results,
            st.session_state.df,
            st.session_state.text_column,
            st.session_state.respondent_id_column,
            st.session_state.session_id
        )
        export_type = "summary"
        filename_suffix = "_summary"
    else:
        # Create detailed export
        results_df = st.session_state.backend.export_results(
            results,
            st.session_state.df,
            st.session_state.text_column,
            st.session_state.respondent_id_column,
            st.session_state.session_id
        )
        export_type = "detailed"
        filename_suffix = "_detailed"

    # Show preview of selected export
    st.write(f"**Preview of {export_type} export data:**")
    st.dataframe(results_df.head(10), use_container_width=True)
    st.caption(f"Showing first 10 rows of {len(results_df)} total rows")

    # Show column information
    with st.expander("Column Information"):
        if export_type == "summary":
            st.write("**Summary Export Columns:**")
            for col in results_df.columns:
                if col == 'auto_generated_id':
                    st.write(f"• **{col}**: System-generated unique identifier")
                elif col.startswith('user_id_'):
                    st.write(f"• **{col}**: Your original ID column")
                elif col.startswith('original_'):
                    st.write(f"• **{col}**: Your original text data")
                elif col == 'cluster_id':
                    st.write(f"• **{col}**: Assigned cluster number (-1 = outlier)")
                elif col == 'confidence_score':
                    st.write(f"• **{col}**: Clustering confidence (0-1)")
                elif col == 'confidence_level':
                    st.write(f"• **{col}**: High/Medium/Low confidence category")
                elif col == 'cluster_label':
                    st.write(f"• **{col}**: Descriptive cluster name based on keywords")
        else:
            st.write("**Detailed Export includes all columns:**")
            st.write("• All summary columns (above)")
            st.write("• **processed_text**: Cleaned text used for clustering")
            st.write("• Additional metadata columns")

    # Export buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv_data = results_df.to_csv(index=False)
        if st.download_button(
            f"Download {export_type.title()} CSV",
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
        # Summary report
        summary_report = st.session_state.backend.create_summary_report(
            results,
            st.session_state.preprocessing_settings,
            st.session_state.session_id
        )
        
        if st.download_button(
            "Download Summary Report",
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

