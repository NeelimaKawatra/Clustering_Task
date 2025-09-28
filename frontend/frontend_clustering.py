# frontend/frontend_clustering.py - Complete clustering interface
import streamlit as st
import pandas as pd
import time

def tab_clustering(backend_available):
    """Tab: Clustering Configuration and Execution"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(
            st.session_state.session_id, "tab_visit", {"tab_name": "clustering"}
        )
    
    # Check prerequisites
    if not st.session_state.get('tab_preprocessing_complete', False):
        st.error("Please complete Preprocessing first!")
        st.info("Go to the Preprocessing tab to clean and prepare your text data.")
        return
    
    if not backend_available:
        st.error("Backend services not available. Please check backend installation.")
        return

    # Get processed data
    processed_texts = st.session_state.get('processed_texts')
    if not processed_texts:
        st.error("No preprocessed texts found. Please complete Preprocessing first.")
        return
    
    st.subheader("Clustering Configuration")
    
    # Show data status
    st.success(f"Ready to cluster {len(processed_texts)} preprocessed text entries")
    
    # Get parameter recommendations
    try:
        recommended_params = st.session_state.backend.get_clustering_parameters(
            len(processed_texts), st.session_state.session_id
        )
    except Exception as e:
        st.warning(f"Could not get parameter recommendations: {e}")
        recommended_params = {
            "min_topic_size": 5,
            "min_samples": 3,
            "n_neighbors": 15,
            "n_components": 5,
            "metric": "cosine",
            "random_state": 42
        }
    
    # Parameter configuration
    with st.expander("Clustering Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Parameters**")
            min_topic_size = st.slider(
                "Number of Clusters (K)", 
                min_value=2, 
                max_value=min(20, len(processed_texts)//3), 
                value=recommended_params.get("min_topic_size", 5),
                help="Target number of clusters to create"
            )
            
            n_components = st.slider(
                "Dimensionality Reduction Components",
                min_value=2,
                max_value=min(20, len(processed_texts)//2),
                value=recommended_params.get("n_components", 5),
                help="Number of components for dimensionality reduction"
            )
        
        with col2:
            st.markdown("**Advanced Parameters**")
            random_state = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=999999,
                value=recommended_params.get("random_state", 42),
                help="Seed for reproducible results"
            )
            
            metric = st.selectbox(
                "Distance Metric",
                options=["cosine", "euclidean", "manhattan"],
                index=0 if recommended_params.get("metric") == "cosine" else 0,
                help="Metric for measuring text similarity"
            )
    
    # Parameter summary
    clustering_params = {
        "min_topic_size": min_topic_size,
        "n_components": n_components,
        "random_state": random_state,
        "metric": metric,
        "min_samples": recommended_params.get("min_samples", 3),
        "n_neighbors": recommended_params.get("n_neighbors", 15)
    }
    
    # Show parameter summary in user-friendly format
    with st.expander("Clustering Configuration Summary", expanded=False):
        st.markdown("**Your Selected Settings:**")
        
        # User-friendly parameter descriptions
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"• **Target Clusters:** {clustering_params.get('min_topic_size', 'N/A')}")
            st.write(f"• **Complexity Reduction:** {clustering_params.get('n_components', 'N/A')} dimensions")
            
        with col2:
            st.write(f"• **Similarity Measure:** {clustering_params.get('metric', 'cosine').title()}")
            st.write(f"• **Random Seed:** {clustering_params.get('random_state', 42)}")
        
        st.caption("These settings control how the algorithm groups your text entries into clusters.")
        
    # Run clustering button
    st.markdown("---")
    
    if st.button("🔍 Run Clustering Analysis", type="primary", use_container_width=True):
        # Check if re-clustering will affect fine-tuning work
        if st.session_state.get('finetuning_initialized'):
            st.warning("🔄 Running new clustering will reset your fine-tuning work!")
            # Reset finetuning state only (clustering doesn't affect its own permanent progress)
            for key in list(st.session_state.keys()):
                if key.startswith('finetuning_'):
                    del st.session_state[key]
        
        run_clustering_analysis(processed_texts, clustering_params, backend_available)
    
    # Show results if available
    clustering_results = st.session_state.get('clustering_results')
    if clustering_results and clustering_results.get("success"):
        show_clustering_results()

def run_clustering_analysis(processed_texts, clustering_params, backend_available):
    """Execute the clustering analysis"""
    
    with st.spinner("Running clustering analysis..."):
        # Add progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Setting up clustering model...")
        progress_bar.progress(0.2)
        time.sleep(0.5)
        
        status_text.text("Processing text embeddings...")
        progress_bar.progress(0.4)
        time.sleep(0.5)
        
        status_text.text("Finding optimal clusters...")
        progress_bar.progress(0.7)
        time.sleep(0.5)
        
        try:
            # Run clustering using backend
            clustering_results = st.session_state.backend.run_clustering(
                processed_texts, 
                clustering_params, 
                st.session_state.session_id
            )
            
            progress_bar.progress(1.0)
            status_text.text("Clustering complete!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if clustering_results.get("success"):
                # Store results
                st.session_state.clustering_results = clustering_results
                
                st.success("🎉 Clustering analysis completed successfully!")
                
                # Show quick summary
                stats = clustering_results.get("statistics", {})
                st.info(f"Found {stats.get('n_clusters', 'unknown')} clusters from {stats.get('total_texts', len(processed_texts))} texts")
                
                # ✅ CRITICAL: Refresh sidebar immediately to show green button
                st.rerun()
                
            else:
                error_msg = clustering_results.get("error", "Unknown error occurred")
                st.error(f"❌ {error_msg}")
                
                # Show helpful suggestions
                suggestions = clustering_results.get("suggestions", [])
                if suggestions:
                    st.markdown("**💡 Try these solutions:**")
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
                
                # Show technical error in debug section
                technical_error = clustering_results.get("technical_error")
                debug_info = clustering_results.get("debug_info", {})
                
                if technical_error or debug_info:
                    with st.expander("🔧 Technical Details (for debugging)"):
                        if technical_error:
                            st.code(technical_error)
                            st.caption("Original error message for technical support.")
                        if debug_info:
                            st.json(debug_info)
                            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Clustering analysis failed: {str(e)}")
            
            # Provide troubleshooting tips
            with st.expander("Troubleshooting Tips"):
                st.markdown("""
                **Common solutions:**
                - Try reducing the number of clusters
                - Check if you have enough text data (minimum 10 texts recommended)
                - Ensure preprocessing was successful
                - Try different parameter settings
                - Check backend logs for detailed error information
                """)

def show_clustering_results():
    """Display comprehensive clustering results"""
    
    results = st.session_state.get('clustering_results')
    if not results:
        return
    
    st.markdown("---")
    st.subheader("Clustering Results")
    
    # Overall statistics
    stats = results.get("statistics", {})
    confidence = results.get("confidence_analysis", {})
    performance = results.get("performance", {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Clusters Found", stats.get("n_clusters", "N/A"))
    with col2:
        st.metric("Text Entries Clustered", stats.get("clustered", "N/A"))
    with col3:
        success_rate = stats.get("success_rate", 0)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        avg_confidence = confidence.get("avg_confidence", 0)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Confidence distribution
    st.markdown("**Confidence Distribution**")
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    
    with conf_col1:
        high_conf = confidence.get("high_confidence", 0)
        st.metric("High Confidence (≥0.7)", high_conf)
    with conf_col2:
        med_conf = confidence.get("medium_confidence", 0)
        st.metric("Medium Confidence (0.3-0.7)", med_conf)
    with conf_col3:
        low_conf = confidence.get("low_confidence", 0)
        st.metric("Low Confidence (<0.3)", low_conf)
    
    # Cluster details
    st.markdown("**Cluster Analysis**")
    
    try:
        cluster_details = st.session_state.backend.get_cluster_details(
            results, st.session_state.session_id
        )
        
        if cluster_details and not cluster_details.get("error"):
            show_cluster_details_table(cluster_details, results)
        else:
            st.warning("Could not generate detailed cluster analysis")
            
    except Exception as e:
        st.warning(f"Error generating cluster details: {e}")
    
    # Performance metrics
    with st.expander("Performance Metrics"):
        st.write(f"**Total Processing Time:** {performance.get('total_time', 0):.2f} seconds")
        st.write(f"**Setup Time:** {performance.get('setup_time', 0):.2f} seconds") 
        st.write(f"**Clustering Time:** {performance.get('clustering_time', 0):.2f} seconds")
        
        params_used = results.get("parameters_used", {})
        if params_used:
            st.write("**Parameters Used:**")
            st.json(params_used)
    
    # Next steps
    st.markdown("---")
    st.success("Clustering Complete!")
    st.info("Proceed to **Fine-tuning** to explore options to manually adjust cluster assignments.")

def show_cluster_details_table(cluster_details, results):
    """Show detailed cluster information in a table format"""
    
    # Build cluster summary table
    cluster_data = []
    topics = results.get("topics", [])
    texts = results.get("texts", [])
    probabilities = results.get("probabilities", [])
    
    for cluster_id, details in cluster_details.items():
        if cluster_id == -1:  # Skip outliers for main table
            continue
            
        cluster_data.append({
            "Cluster ID": cluster_id,
            "Size": details.get("size", 0),
            "Avg Confidence": f"{details.get('avg_confidence', 0):.2f}",
            "High Conf Count": details.get("high_confidence_count", 0),
            "Top Keywords": ", ".join(details.get("keywords", [])[:3])
        })
    
    if cluster_data:
        cluster_df = pd.DataFrame(cluster_data)
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)
    
    # Detailed cluster inspection
    with st.expander("Detailed Cluster Inspection", expanded=False):
        selected_cluster = st.selectbox(
            "Select cluster to inspect:",
            options=list(cluster_details.keys()),
            format_func=lambda x: f"Cluster {x}" if x != -1 else "Outliers"
        )
        
        if selected_cluster in cluster_details:
            cluster_info = cluster_details[selected_cluster]
            
            st.write(f"**Cluster {selected_cluster} Details:**")
            st.write(f"- Size: {cluster_info.get('size', 0)} texts")
            st.write(f"- Average Confidence: {cluster_info.get('avg_confidence', 0):.3f}")
            st.write(f"- Keywords: {', '.join(cluster_info.get('keywords', []))}")
            
            # Show top texts
            top_texts = cluster_info.get("top_texts", [])
            if top_texts:
                st.write("**Representative Texts (by confidence):**")
                for i, (text, confidence) in enumerate(top_texts[:5], 1):
                    st.write(f"{i}. **({confidence:.2f})** {text[:200]}{'...' if len(text) > 200 else ''}")
    
    # Outliers section
    if -1 in cluster_details:
        outlier_info = cluster_details[-1]
        with st.expander(f"Outliers ({outlier_info.get('size', 0)} texts)", expanded=False):
            outlier_texts = outlier_info.get("texts", [])
            outlier_confs = outlier_info.get("confidences", [])
            
            st.write("**Outlier texts (low clustering confidence):**")
            for i, (text, conf) in enumerate(zip(outlier_texts[:10], outlier_confs[:10]), 1):
                st.write(f"{i}. **({conf:.2f})** {text[:150]}{'...' if len(text) > 150 else ''}")
            
            if len(outlier_texts) > 10:
                st.caption(f"... and {len(outlier_texts) - 10} more outliers")

def show_clustering_tips():
    """Show tips for better clustering results"""
    
    with st.expander("Tips for Better Clustering Results"):
        st.markdown("""
        **Parameter Tuning:**
        - **More clusters**: Increase K if you see overly broad clusters
        - **Fewer clusters**: Decrease K if you see very small, similar clusters
        - **Different seed**: Try different random seeds for consistency
        
        **Data Quality:**
        - Ensure texts are sufficiently different to form meaningful clusters
        - Consider more aggressive preprocessing if clusters seem noisy
        - Remove very short texts (< 10 words) that might be outliers
        
        **Interpreting Results:**
        - High confidence (>0.7): Text clearly belongs to its cluster
        - Medium confidence (0.3-0.7): Text could belong to multiple clusters
        - Low confidence (<0.3): Text might be an outlier or misclassified
        
        **Next Steps:**
        - Use Fine-tuning to manually adjust cluster assignments
        - Review cluster keywords to understand themes
        - Export results for further analysis
        """)

# Additional utility functions
def export_clustering_results():
    """Export clustering results to downloadable format"""
    
    results = st.session_state.get('clustering_results')
    if not results:
        return None
    
    processed_texts = st.session_state.get('processed_texts', [])
    
    # Build export data
    export_data = []
    topics = results.get("topics", [])
    probabilities = results.get("probabilities", [])
    
    for i, (text, topic, prob) in enumerate(zip(processed_texts, topics, probabilities)):
        export_data.append({
            "text_id": i + 1,
            "text": text,
            "cluster_id": topic,
            "confidence": prob
        })
    
    return pd.DataFrame(export_data)

def reset_clustering_results():
    """Reset clustering results and clear downstream data"""
    
    if 'clustering_results' in st.session_state:
        del st.session_state['clustering_results']
    
    # Clear any downstream processing
    for key in list(st.session_state.keys()):
        if key.startswith('finetuning_'):
            del st.session_state[key]
    
    st.success("Clustering results cleared. You can run clustering again with different parameters.")
    st.rerun()
