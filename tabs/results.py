import streamlit as st

def tab_d_results(backend_available):
    """Tab D: Results Visualization and Export using backend services"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "results"})
    
    st.header("📊 Clustering Results")
    
    # Check if clustering is complete
    if not st.session_state.get('clustering_results') or not st.session_state.clustering_results.get("success"):
        st.info("💡 Complete clustering first to see results here.")
        return
    
    if not backend_available:
        st.error("❌ Backend services not available. Please check backend installation.")
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
    
    # Export section
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    # Create results dataframe using backend
    results_df = st.session_state.backend.export_results(
        results, 
        st.session_state.df, 
        st.session_state.text_column, 
        st.session_state.respondent_id_column, 
        st.session_state.session_id
    )
    
    # Show preview
    st.write("**Preview of export data:**")
    st.dataframe(results_df.head(), use_container_width=True)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = results_df.to_csv(index=False)
        if st.download_button(
            "📥 Download CSV",
            csv_data,
            "clustering_results.csv",
            "text/csv",
            use_container_width=True
        ):
            # Track export
            st.session_state.backend.track_activity(st.session_state.session_id, "export", {
                "export_type": "csv",
                "export_info": {"rows": len(results_df), "format": "csv"}
            })
    
    with col2:
        # Create summary report using backend
        summary_report = st.session_state.backend.create_summary_report(
            results, 
            st.session_state.preprocessing_settings, 
            st.session_state.session_id
        )
        
        if st.download_button(
            "📄 Download Summary Report",
            summary_report,
            "clustering_summary.txt",
            "text/plain",
            use_container_width=True
        ):
            # Track export
            st.session_state.backend.track_activity(st.session_state.session_id, "export", {
                "export_type": "summary",
                "export_info": {"format": "text"}
            })
    
    with col3:
        if st.button("🔄 **New Analysis**", use_container_width=True):
            # Get session analytics before clearing
            session_summary = st.session_state.backend.get_session_analytics(st.session_state.session_id)
            
            # Clear all session state to start over
            for key in list(st.session_state.keys()):
                if key.startswith(('df', 'processed_', 'clustering_', 'tab_')):
                    del st.session_state[key]
            
            # Show session summary
            with st.expander("📊 Session Analytics"):
                st.write("**Your Session Summary:**")
                st.write(f"• Duration: {session_summary.get('duration_seconds', 0):.0f} seconds")
                st.write(f"• Completion: {session_summary.get('completion_percentage', 0):.0f}%")
                st.write(f"• Activities: {session_summary.get('total_activities', 0)}")
                
                activity_counts = session_summary.get('activity_counts', {})
                if activity_counts:
                    st.write("**Activity Breakdown:**")
                    for activity, count in activity_counts.items():
                        st.write(f"  - {activity.replace('_', ' ').title()}: {count}")
            
            st.success("✅ Ready for new analysis! Go to Data Loading tab.")
            st.rerun()