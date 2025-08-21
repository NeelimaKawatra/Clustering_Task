import streamlit as st
import time

def show_clustering_approach_info():
    """Display information about the progressive clustering approach"""
    
    with st.expander("Understanding Our Clustering Approach"):
        st.markdown("### How We Find the Best Clustering Method for Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Our Progressive Strategy**")
            st.write("1. **Advanced UMAP Method** - Highest quality results")
            st.write("   - Uses sophisticated dimensionality reduction")
            st.write("   - Best for finding subtle patterns")
            st.write("   - May fail with certain dataset sizes")
            st.write("")
            st.write("2. **PCA Fallback** - Good alternative")
            st.write("   - Reliable dimensionality reduction")
            st.write("   - Works with more datasets")
            st.write("   - Slightly less sophisticated")
            st.write("")
            st.write("3. **Basic Clustering** - Direct approach")
            st.write("   - No dimensionality reduction")
            st.write("   - Fast and reliable")
            st.write("   - Works with most datasets")
            st.write("")
            st.write("4. **Minimal Configuration** - Maximum compatibility")
            st.write("   - Simplest possible setup")
            st.write("   - Guaranteed to work")
            st.write("   - Basic but functional results")
        
        with col2:
            st.markdown("**Why This Approach?**")
            st.info("""
            **Dataset Size Matters**: Different clustering methods work better with different dataset sizes. 
            Your dataset has specific mathematical constraints that we automatically detect and handle.
            """)
            
            st.markdown("**What You'll See**")
            st.write("- Real-time updates as we try each method")
            st.write("- Clear explanations when methods fail")
            st.write("- Final summary of what worked")
            st.write("- Honest assessment of result quality")
            
            st.markdown("**Quality Levels**")
            st.write("🟢 **High**: Advanced UMAP clustering")
            st.write("🟡 **Medium**: PCA or Basic clustering") 
            st.write("🟠 **Low**: Minimal configuration")
            
            st.markdown("**Remember**")
            st.write("Even 'Low' quality results are often very useful for understanding your data patterns!")

def show_setup_summary(clustering_results):
    """Display the setup summary from clustering results"""
    
    if clustering_results and clustering_results.get("success"):
        metadata = clustering_results.get("metadata", {})
        setup_summary = metadata.get("setup_summary", "")
        model_config = metadata.get("model_config", {})
        
        if setup_summary:
            with st.expander("Clustering Setup Summary - What Actually Happened"):
                st.markdown(setup_summary)
                
                if model_config:
                    st.markdown("### Final Configuration Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Method Used**: {model_config.get('method', 'Unknown')}")
                        st.write(f"**Sophistication**: {model_config.get('sophistication', 'Unknown')}")
                        
                    with col2:
                        st.write(f"**Dimensionality Reduction**: {model_config.get('dimensionality_reduction', 'Unknown')}")
                        st.write(f"**Probabilities**: {'Yes' if model_config.get('has_probabilities', False) else 'Synthetic'}")
                
                # Quality assessment
                sophistication = model_config.get('sophistication', 'Unknown')
                if sophistication == 'High':
                    st.success("Excellent: Using advanced clustering with full feature set")
                elif sophistication == 'Medium':
                    st.info("Good: Using reliable clustering with some limitations")
                elif sophistication == 'Low':
                    st.warning("Basic: Using simplified clustering for maximum compatibility")

def tab_c_clustering(backend_available):
    """Tab C: Enhanced Clustering with Progressive Fallback Display"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "clustering"})
    
    st.header("Clustering Configuration")
    
    # Check if preprocessing is complete
    if st.session_state.processed_texts is None:
        st.error("No processed text found. Please complete preprocessing first.")
        return
    
    if not backend_available:
        st.error("Backend services not available. Please check backend installation.")
        return
    
    texts = st.session_state.processed_texts
    st.success(f"Ready to cluster {len(texts)} processed texts")
    
    # Show preprocessing summary
    with st.expander("Preprocessing Summary"):
        settings = st.session_state.preprocessing_settings
        st.write(f"**Method:** {settings['method']}")
        st.write(f"**Details:** {settings['details']}")
        st.write(f"**Texts ready:** {len(texts)}")
    
    # Show clustering approach information
    show_clustering_approach_info()
    
    st.subheader("Clustering Parameters")
    
    # Get optimal parameters from backend
    optimal_params = st.session_state.backend.get_clustering_parameters(len(texts), st.session_state.session_id)
    
    # Parameter selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Recommended parameters for your dataset:**")
        for key, value in optimal_params.items():
            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
    
    with col2:
        use_optimal = st.radio(
            "Parameter choice:",
            ["Use recommended", "Customize"],
            help="Recommended settings work best for most datasets"
        )
    
    # Custom parameters if selected
    if use_optimal == "Customize":
        st.subheader("Custom Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            min_cluster_size = st.slider("Min Cluster Size", 2, 20, optimal_params['min_cluster_size'],
                                       help="Minimum number of texts required to form a cluster")
            n_neighbors = st.slider("UMAP Neighbors", 2, 15, optimal_params['n_neighbors'],
                                  help="Number of neighbors for UMAP (if used)")
            embedding_model = st.selectbox("Embedding Model",
                                         ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                                         index=0,
                                         help="Model for converting text to vectors")
        
        with col2:
            min_samples = st.slider("Min Samples", 1, 10, optimal_params['min_samples'],
                                  help="Minimum samples for core points in clustering")
            n_components = st.slider("UMAP Components", 2, 8, optimal_params['n_components'],
                                   help="Number of dimensions for UMAP (if used)")
        
        params = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'n_neighbors': n_neighbors,
            'n_components': n_components,
            'embedding_model': embedding_model
        }
    else:
        params = optimal_params
    
    # Current configuration summary
    st.info(f"""
    **Strategy**: We'll try advanced methods first, then fall back to simpler approaches if needed.
    Your dataset ({len(texts)} texts) will be tested with multiple clustering configurations automatically.
    """)
    
    # Debug information expander
    with st.expander("Debug Information"):
        st.write(f"**Texts count:** {len(texts)}")
        st.write(f"**Sample text:** {texts[0][:100] if texts else 'None'}...")
        st.write(f"**Text lengths:** {[len(str(t)) for t in texts[:5]]}")
        st.write(f"**Empty texts:** {sum(1 for t in texts if not t or not str(t).strip())}")
        st.write(f"**Parameters:** {params}")
        st.write(f"**Backend available:** {backend_available}")
        if hasattr(st.session_state, 'backend') and hasattr(st.session_state.backend, 'clustering_model'):
            st.write(f"**Clustering ready:** {getattr(st.session_state.backend.clustering_model, 'clustering_ready', False)}")
    
    # Clustering execution
    st.markdown("---")
    st.subheader("Start Clustering")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Run Progressive Clustering", type="primary", use_container_width=True):
            
            # Pre-clustering validation
            if not texts:
                st.error("No texts available for clustering")
                return
            
            valid_texts = [t for t in texts if t and str(t).strip()]
            if len(valid_texts) < 5:
                st.error(f"Need at least 5 valid texts for clustering. Found {len(valid_texts)} valid texts out of {len(texts)}")
                return
            
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a container for real-time updates
            updates_container = st.empty()
            
            try:
                status_text.text("Validating data...")
                progress_bar.progress(10)
                
                status_text.text("Starting progressive clustering approach...")
                progress_bar.progress(20)
                
                # Show what's happening
                with updates_container.container():
                    st.info("🔄 **Progressive Clustering in Progress**")
                    st.write("We're trying different clustering methods to find the best approach for your data...")
                
                # Run clustering using backend with enhanced error handling
                clustering_results = st.session_state.backend.run_clustering(
                    texts, params, st.session_state.session_id
                )
                
                progress_bar.progress(90)
                
                # Clear the updates container
                updates_container.empty()
                
                # Check results more thoroughly
                if not clustering_results:
                    raise Exception("No results returned from clustering service")
                
                if not clustering_results.get("success"):
                    error_msg = clustering_results.get('error', 'Unknown clustering error')
                    debug_info = clustering_results.get('debug_info', {})
                    
                    st.error(f"All clustering methods failed: {error_msg}")
                    
                    # Show what was attempted
                    attempts = debug_info.get('attempts_made', [])
                    if attempts:
                        with st.expander("What We Tried"):
                            for attempt in attempts:
                                status_icon = "✅" if attempt['result'] == 'Success' else "❌"
                                st.write(f"{status_icon} **{attempt['method']}**: {attempt['result']}")
                                st.write(f"   {attempt['details']}")
                    
                    # Provide guidance
                    st.info("Consider adjusting your text preprocessing or trying different parameters.")
                    return
                
                # Validate specific result fields
                required_fields = ['predictions', 'topics', 'probabilities', 'statistics']
                missing_fields = [field for field in required_fields if field not in clustering_results]
                
                if missing_fields:
                    st.error(f"Clustering results missing required fields: {missing_fields}")
                    return
                
                # Check for empty predictions/topics
                if not clustering_results.get('predictions') and not clustering_results.get('topics'):
                    st.error("No prediction data was generated")
                    return
                
                progress_bar.progress(100)
                status_text.text("Clustering completed successfully!")
                
                # Store results
                st.session_state.clustering_results = clustering_results
                st.session_state.tab_c_complete = True
                
                # Show what actually happened
                show_setup_summary(clustering_results)
                
                # Track clustering completion
                st.session_state.backend.track_activity(st.session_state.session_id, "clustering", {
                    "parameters": params,
                    "results": {
                        "clusters": clustering_results["statistics"]["n_clusters"],
                        "success_rate": clustering_results["statistics"]["success_rate"],
                        "processing_time": clustering_results["performance"]["total_time"],
                        "final_method": clustering_results["metadata"].get("model_config", {}).get("method", "unknown")
                    }
                })
                
                st.balloons()
                st.success("Clustering successful! Check the Results tab to see your clusters.")
                
                # Clean up progress indicators
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                # Switch to results tab
                st.session_state.current_page = "results"
                st.rerun()
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("Clustering failed!")
                updates_container.empty()
                
                error_msg = str(e)
                st.error(f"Clustering error: {error_msg}")
                
                # Log error for debugging
                if hasattr(st.session_state, 'backend'):
                    st.session_state.backend.track_activity(st.session_state.session_id, "clustering_error", {
                        "error": error_msg,
                        "parameters": params,
                        "text_count": len(texts)
                    })
                
                # Provide helpful suggestions
                st.info("Try adjusting your preprocessing method or clustering parameters.")
    
    # Show previous clustering results if available
    if st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success"):
        results = st.session_state.clustering_results
        stats = results["statistics"]
        
        st.markdown("---")
        st.subheader("Quick Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clusters Found", stats['n_clusters'])
        with col2:
            st.metric("Clustered Texts", stats['clustered'])
        with col3:
            st.metric("Outliers", stats['outliers'])
        with col4:
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
        
        # Show the setup summary again for easy reference
        show_setup_summary(results)
        
        st.info("**View detailed results in the Results tab!**")