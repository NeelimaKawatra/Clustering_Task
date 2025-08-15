import streamlit as st

def tab_c_clustering(backend_available):
    """Tab C: Clustering Configuration and Execution using backend services"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "clustering"})
    
    st.header("⚙️ Clustering Configuration")
    
    # Check if preprocessing is complete
    if st.session_state.processed_texts is None:
        st.error("❌ No processed text found. Please complete preprocessing first.")
        return
    
    if not backend_available:
        st.error("❌ Backend services not available. Please check backend installation.")
        return
    
    texts = st.session_state.processed_texts
    st.success(f"✅ Ready to cluster {len(texts)} processed texts")
    
    # Show preprocessing summary
    with st.expander("📋 Preprocessing Summary"):
        settings = st.session_state.preprocessing_settings
        st.write(f"**Method:** {settings['method']}")
        st.write(f"**Details:** {settings['details']}")
        st.write(f"**Texts ready:** {len(texts)}")
    
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
            n_neighbors = st.slider("UMAP Neighbors", 5, 30, optimal_params['n_neighbors'],
                                  help="Number of neighbors for UMAP dimensionality reduction")
            embedding_model = st.selectbox("Embedding Model",
                                         ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                                         index=0,
                                         help="Model for converting text to vectors")
        
        with col2:
            min_samples = st.slider("Min Samples", 1, 10, optimal_params['min_samples'],
                                  help="Minimum samples for core points in clustering")
            n_components = st.slider("UMAP Components", 2, 20, optimal_params['n_components'],
                                   help="Number of dimensions for UMAP reduction")
        
        params = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'n_neighbors': n_neighbors,
            'n_components': n_components,
            'embedding_model': embedding_model
        }
    else:
        params = optimal_params
    
    # Clustering execution
    st.markdown("---")
    st.subheader("🚀 Start Clustering")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Run BERTopic Clustering", type="primary", use_container_width=True):
            
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing clustering...")
            progress_bar.progress(20)
            
            # Run clustering using backend
            clustering_results = st.session_state.backend.run_clustering(
                texts, params, st.session_state.session_id
            )
            
            progress_bar.progress(100)
            
            if clustering_results.get("success"):
                status_text.text("✅ Clustering completed!")
                
                # Store results
                st.session_state.clustering_results = clustering_results
                st.session_state.tab_c_complete = True
                
                # Track clustering completion
                st.session_state.backend.track_activity(st.session_state.session_id, "clustering", {
                    "parameters": params,
                    "results": {
                        "clusters": clustering_results["statistics"]["n_clusters"],
                        "success_rate": clustering_results["statistics"]["success_rate"],
                        "processing_time": clustering_results["performance"]["total_time"]
                    }
                })
                
                st.balloons()
                st.success("🎉 **Clustering successful!** Check the Results tab to see your clusters.")
                # Switch to results tab
                st.query_params.tab = "results"
                st.rerun()
            else:
                status_text.text("❌ Clustering failed!")
                st.error(f"❌ Clustering failed: {clustering_results.get('error', 'Unknown error')}")
    
    # Show quick results if clustering is complete
    if st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success"):
        results = st.session_state.clustering_results
        stats = results["statistics"]
        
        st.markdown("---")
        st.subheader("📊 Quick Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🗂️ Clusters Found", stats['n_clusters'])
        with col2:
            st.metric("✅ Clustered Texts", stats['clustered'])
        with col3:
            st.metric("❓ Outliers", stats['outliers'])
        with col4:
            st.metric("📈 Success Rate", f"{stats['success_rate']:.1f}%")
        
        st.info("📊 **View detailed results in the Results tab!**")