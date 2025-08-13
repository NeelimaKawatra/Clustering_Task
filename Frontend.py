import streamlit as st
import pandas as pd
import os
import time
from collections import Counter

# Import the backend
try:
    from backend import ClusteryBackend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    st.error("❌ Backend not found! Please ensure backend.py is in your project directory.")

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize session state variables and backend"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'respondent_id_column' not in st.session_state:
        st.session_state.respondent_id_column = None
    if 'text_column' not in st.session_state:
        st.session_state.text_column = None
    if 'tab_a_complete' not in st.session_state:
        st.session_state.tab_a_complete = False
    if 'processed_texts' not in st.session_state:
        st.session_state.processed_texts = None
    if 'preprocessing_settings' not in st.session_state:
        st.session_state.preprocessing_settings = {}
    if 'tab_b_complete' not in st.session_state:
        st.session_state.tab_b_complete = False
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None
    
    # Initialize backend and session
    if 'backend' not in st.session_state and BACKEND_AVAILABLE:
        st.session_state.backend = ClusteryBackend()
        st.session_state.session_id = f"user_{int(time.time())}"
        st.session_state.backend.start_session(st.session_state.session_id)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_file_from_upload(uploaded_file):
    """Convert uploaded file to temporary file path for backend processing"""
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    return None

# ============================================================================
# TAB A: DATA LOADING
# ============================================================================

def tab_a_data_loading():
    """Tab A: Data Loading using backend services"""
    initialize_session_state()
    
    # Track tab visit
    if BACKEND_AVAILABLE:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "data_loading"})
    
    st.header("📁 Data Loading")
    
    # File upload section
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your survey data or text file for clustering"
    )
    
    if uploaded_file is not None:
        if not BACKEND_AVAILABLE:
            st.error("❌ Backend services not available. Please check backend installation.")
            return
        
        try:
            # Convert uploaded file to temporary path
            temp_file_path = get_file_from_upload(uploaded_file)
            
            # Use backend to load and validate file
            success, df, message = st.session_state.backend.load_data(temp_file_path, st.session_state.session_id)
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            if not success:
                st.error(f"❌ {message}")
                return
            
            st.success(f"✅ {message}")
            
            # Store dataframe
            st.session_state.df = df
            
            # Show file metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Rows", len(df))
            with col2:
                st.metric("📋 Columns", len(df.columns))
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024
                st.metric("💾 Size", f"{memory_usage:.1f} KB")
            
            # Data preview
            with st.expander("👀 Preview Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # Column selection section
            st.subheader("Select Columns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🆔 Respondent ID Column (Optional)**")
                id_options = ["Auto-generate IDs"] + list(df.columns)
                selected_id = st.selectbox(
                    "Choose ID column:",
                    id_options,
                    help="Select a column to track individual responses"
                )
                
                if selected_id == "Auto-generate IDs":
                    st.session_state.respondent_id_column = None
                    st.info("💡 Will create sequential IDs: ID_001, ID_002, etc.")
                else:
                    st.session_state.respondent_id_column = selected_id
            
            with col2:
                st.markdown("**📝 Text Column for Clustering**")
                
                # Get text column suggestions from backend
                text_columns = st.session_state.backend.get_text_column_suggestions(df, st.session_state.session_id)
                
                if text_columns:
                    st.info(f"💡 Detected text columns: {', '.join(text_columns[:3])}")
                
                text_column = st.selectbox(
                    "Select text column:",
                    df.columns,
                    help="Choose the column with text you want to cluster"
                )
                
                if text_column:
                    st.session_state.text_column = text_column
            
            # Validate columns using backend
            if st.session_state.text_column:
                validation_result = st.session_state.backend.validate_columns(
                    df, 
                    st.session_state.text_column, 
                    st.session_state.respondent_id_column, 
                    st.session_state.session_id
                )
                
                if validation_result["text_column_valid"]:
                    st.success(f"✅ {validation_result['text_column_message']}")
                    
                    # Show text quality metrics
                    stats = validation_result["text_quality"]
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Valid texts", stats['total_texts'] - stats['empty_texts'])
                    with col_b:
                        st.metric("Avg length", f"{stats['avg_length']:.0f}")
                    with col_c:
                        st.metric("Avg words", f"{stats['avg_words']:.1f}")
                    with col_d:
                        st.metric("Unique", stats['unique_texts'])
                    
                    # Sample texts
                    with st.expander("📖 Sample texts", expanded=False):
                        sample_texts = df[text_column].dropna().head(5)
                        for i, text in enumerate(sample_texts, 1):
                            st.write(f"**{i}.** {str(text)[:150]}{'...' if len(str(text)) > 150 else ''}")
                    
                    # Show ID column analysis
                    id_analysis = validation_result["id_column_analysis"]
                    if id_analysis["status"] == "perfect":
                        st.success(f"✅ {id_analysis['message']}")
                    elif id_analysis["status"] in ["duplicates", "missing"]:
                        st.warning(f"⚠️ {id_analysis['message']}")
                    
                    # Ready to proceed
                    st.markdown("---")
                    st.subheader("✅ Ready to Proceed")
                    
                    # Summary display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**📊 Data Ready:**")
                        st.write(f"• {len(df)} rows")
                        st.write(f"• {stats['total_texts'] - stats['empty_texts']} valid texts")
                    
                    with col2:
                        st.markdown("**📝 Text Column:**")
                        st.write(f"• {st.session_state.text_column}")
                        st.write(f"• Avg: {stats['avg_length']:.0f} chars")
                    
                    with col3:
                        st.markdown("**🆔 ID Setup:**")
                        if st.session_state.respondent_id_column:
                            st.write(f"• {st.session_state.respondent_id_column}")
                            st.write(f"• {id_analysis['status']}")
                        else:
                            st.write("• Auto-generating")
                    
                    # Proceed button
                    if st.button("🚀 Proceed to Preprocessing", type="primary", use_container_width=True):
                        st.session_state.tab_a_complete = True
                        
                        # Track completion
                        st.session_state.backend.track_activity(st.session_state.session_id, "data_upload", {
                            "filename": uploaded_file.name,
                            "rows": len(df),
                            "columns": len(df.columns),
                            "text_column": st.session_state.text_column,
                            "id_column": st.session_state.respondent_id_column
                        })
                        
                        st.success("✅ Data loading complete! Moving to preprocessing...")
                        st.balloons()
                        # Switch to preprocessing tab
                        st.query_params.tab = "preprocessing"
                        st.rerun()
                
                else:
                    st.error(f"❌ {validation_result['text_column_message']}")
                    st.session_state.text_column = None
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.write("Please check that your file is a valid CSV or Excel format.")

# ============================================================================
# TAB B: PREPROCESSING
# ============================================================================

def tab_b_preprocessing():
    """Tab B: Text Preprocessing using backend services"""
    
    # Track tab visit
    if BACKEND_AVAILABLE:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "preprocessing"})
    
    st.header("🔧 Text Preprocessing")
    
    # Get data from previous tab
    df = st.session_state.df
    text_column = st.session_state.text_column
    
    if df is None or text_column is None:
        st.error("❌ No data found. Please complete Data Loading first.")
        return
    
    if not BACKEND_AVAILABLE:
        st.error("❌ Backend services not available. Please check backend installation.")
        return
    
    st.subheader("Choose Preprocessing Level")
    
    # Get preprocessing recommendations from backend
    original_texts = df[text_column].dropna().tolist()
    recommendations = st.session_state.backend.get_preprocessing_recommendations(
        original_texts, st.session_state.session_id
    )
    
    # Show recommendations
    with st.expander("💡 Preprocessing Recommendations"):
        st.write(f"**Suggested method:** {recommendations['suggested_method']}")
        for reason in recommendations['reasons']:
            st.write(f"• {reason}")
    
    # Preprocessing options
    preprocessing_option = st.radio(
        "Select preprocessing approach:",
        [
            "No preprocessing (use original text)",
            "Basic cleaning (URLs, emails, whitespace)",
            "Advanced cleaning (+ stopwords, punctuation)",
            "Custom preprocessing"
        ],
        help="Different levels of text cleaning to improve clustering quality",
        index=1 if recommendations['suggested_method'] == 'basic' else 2
    )
    
    # Custom preprocessing options
    custom_settings = {}
    if preprocessing_option == "Custom preprocessing":
        st.subheader("Custom Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            custom_settings['remove_stopwords'] = st.checkbox("Remove stopwords", value=True)
            custom_settings['remove_punctuation'] = st.checkbox("Remove punctuation", value=True)
        with col2:
            custom_settings['min_length'] = st.slider("Minimum word length", 1, 5, 2)
            custom_settings['remove_numbers'] = st.checkbox("Remove numbers", value=True)
    
    # Process text using backend
    if st.button("🔄 Process Text", type="primary"):
        with st.spinner("Processing text..."):
            
            # Map preprocessing option to backend method
            method_mapping = {
                "No preprocessing (use original text)": "none",
                "Basic cleaning (URLs, emails, whitespace)": "basic",
                "Advanced cleaning (+ stopwords, punctuation)": "advanced",
                "Custom preprocessing": "custom"
            }
            
            method = method_mapping[preprocessing_option]
            
            # Use backend preprocessing service
            processed_texts, metadata = st.session_state.backend.preprocess_texts(
                original_texts, method, custom_settings, st.session_state.session_id
            )
            
            # Store results
            st.session_state.processed_texts = processed_texts
            st.session_state.preprocessing_settings = metadata
            
            st.success(f"✅ Processing complete! {len(processed_texts)} texts ready for clustering.")
    
    # Show results if processing is complete
    if st.session_state.processed_texts is not None:
        metadata = st.session_state.preprocessing_settings
        
        st.subheader("📊 Before/After Analysis")
        
        # Get statistics from metadata
        original_stats = metadata["original_stats"]
        processed_stats = metadata["processed_stats"]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original texts", original_stats['total_texts'])
            st.metric("Processed texts", len(st.session_state.processed_texts), 
                     f"{len(st.session_state.processed_texts) - original_stats['total_texts']:+d}")
        
        with col2:
            st.metric("Original avg length", f"{original_stats['avg_length']:.0f}")
            st.metric("Processed avg length", f"{processed_stats['avg_length']:.0f}",
                     f"{processed_stats['avg_length'] - original_stats['avg_length']:+.0f}")
        
        with col3:
            st.metric("Original avg words", f"{original_stats['avg_words']:.1f}")
            st.metric("Processed avg words", f"{processed_stats['avg_words']:.1f}",
                     f"{processed_stats['avg_words'] - original_stats['avg_words']:+.1f}")
        
        with col4:
            st.metric("Original unique", original_stats['unique_texts'])
            st.metric("Processed unique", processed_stats['unique_texts'],
                     f"{processed_stats['unique_texts'] - original_stats['unique_texts']:+d}")
        
        # Before/After comparison
        st.subheader("🔍 Text Comparison")
        
        # Create aligned original/processed pairs for display
        original_texts_all = df[text_column].dropna().tolist()
        
        # Align original and processed texts
        aligned_pairs = []
        processed_index = 0
        
        for i, original_text in enumerate(original_texts_all):
            if processed_index < len(st.session_state.processed_texts):
                # Apply the same filtering logic to see if this original text would be kept
                if original_text and str(original_text).strip() and len(str(original_text).strip()) > 2 and not str(original_text).strip().isspace():
                    processed_text = st.session_state.processed_texts[processed_index]
                    aligned_pairs.append((original_text, processed_text))
                    processed_index += 1
            
            if len(aligned_pairs) >= 5:  # Only show first 5
                break
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Texts:**")
            if aligned_pairs:
                for i, (original, _) in enumerate(aligned_pairs, 1):
                    st.write(f"**{i}.** {str(original)[:150]}{'...' if len(str(original)) > 150 else ''}")
            else:
                st.write("No matching texts found")
        
        with col2:
            st.markdown("**Processed Texts:**")
            if aligned_pairs:
                for i, (_, processed) in enumerate(aligned_pairs, 1):
                    st.write(f"**{i}.** {processed[:150]}{'...' if len(processed) > 150 else ''}")
            else:
                st.write("No processed texts to show")
        
        # Show if any texts were filtered out
        if len(original_texts_all) > len(st.session_state.processed_texts):
            filtered_out = len(original_texts_all) - len(st.session_state.processed_texts)
            st.info(f"ℹ️ {filtered_out} texts were filtered out (empty, too short, or only whitespace)")
        
        # Word frequency analysis
        if st.checkbox("📈 Show word frequency analysis"):
            st.subheader("Word Frequency Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Text - Top Words:**")
                original_words = []
                for text in df[text_column].dropna():
                    original_words.extend(str(text).lower().split())
                original_freq = Counter(original_words).most_common(10)
                for word, count in original_freq:
                    st.write(f"• {word}: {count}")
            
            with col2:
                st.markdown("**Processed Text - Top Words:**")
                processed_words = []
                for text in st.session_state.processed_texts:
                    processed_words.extend(text.split())
                processed_freq = Counter(processed_words).most_common(10)
                for word, count in processed_freq:
                    st.write(f"• {word}: {count}")
        
        # Processing summary
        with st.expander("ℹ️ Processing Summary"):
            st.write(f"**Method:** {metadata['method']}")
            st.write(f"**Details:** {metadata['details']}")
            st.write(f"**Processing time:** {metadata['processing_time']:.2f} seconds")
            st.write(f"**Results:** {metadata['filtered_count']} valid texts from {metadata['original_count']} original")
            if metadata.get('custom_settings'):
                st.write(f"**Custom settings:** {metadata['custom_settings']}")
        
        # Check if we have enough texts for clustering
        if len(st.session_state.processed_texts) >= 10:
            st.markdown("---")
            st.subheader("✅ Ready for Clustering")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Proceed to Clustering", type="primary", use_container_width=True):
                    st.session_state.tab_b_complete = True
                    
                    # Track preprocessing completion
                    st.session_state.backend.track_activity(st.session_state.session_id, "preprocessing", {
                        "method": metadata['method'],
                        "settings": metadata.get('custom_settings', {}),
                        "original_count": metadata['original_count'],
                        "final_count": metadata['filtered_count'],
                        "processing_time": metadata['processing_time']
                    })
                    
                    st.success("✅ Preprocessing complete! Moving to clustering setup...")
                    st.balloons()
                    # Switch to clustering tab
                    st.query_params.tab = "clustering"
                    st.rerun()
        else:
            st.error(f"❌ Need at least 10 texts for clustering. Current: {len(st.session_state.processed_texts)}")
            st.write("Try using less aggressive preprocessing or check your data quality.")

# ============================================================================
# TAB C: CLUSTERING
# ============================================================================

def tab_c_clustering():
    """Tab C: Clustering Configuration and Execution using backend services"""
    
    # Track tab visit
    if BACKEND_AVAILABLE:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "clustering"})
    
    st.header("⚙️ Clustering Configuration")
    
    # Check if preprocessing is complete
    if st.session_state.processed_texts is None:
        st.error("❌ No processed text found. Please complete preprocessing first.")
        return
    
    if not BACKEND_AVAILABLE:
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

# ============================================================================
# TAB D: RESULTS
# ============================================================================

def tab_d_results():
    """Tab D: Results Visualization and Export using backend services"""
    
    # Track tab visit
    if BACKEND_AVAILABLE:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "results"})
    
    st.header("📊 Clustering Results")
    
    # Check if clustering is complete
    if not st.session_state.get('clustering_results') or not st.session_state.clustering_results.get("success"):
        st.info("💡 Complete clustering first to see results here.")
        return
    
    if not BACKEND_AVAILABLE:
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

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main app with clean, app-like header"""
    st.set_page_config(
        page_title="Clustery - Text Clustering Tool",
        page_icon="🔍",
        layout="wide"
    )
    
    # Custom CSS for clean app header
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 12px 20px;
        margin: -1rem -1rem 0.5rem -1rem;
        border-radius: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-content {
        display: flex;
        justify-content: center;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    .logo {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
        margin-top: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0px 24px;
        background-color: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #667eea !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: 600;
    }
    div.block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Simple app header
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <h1 class="logo">🔍 Clustery</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show backend status
    if not BACKEND_AVAILABLE:
        st.error("❌ **Backend not available!** Please ensure backend.py is in your project directory.")
        st.stop()
    
    # Create tabs - these are the functional navigation
    tabs = st.tabs(["📁 Data Loading", "🔧 Preprocessing", "⚙️ Clustering", "📊 Results"])
    
    # TAB A: DATA LOADING
    with tabs[0]:
        tab_a_data_loading()
    
    # TAB B: PREPROCESSING
    with tabs[1]:
        if st.session_state.get('tab_a_complete', False):
            tab_b_preprocessing()
        else:
            st.info("💡 Complete Data Loading first to unlock this tab.")
    
    # TAB C: CLUSTERING
    with tabs[2]:
        if st.session_state.get('tab_b_complete', False):
            tab_c_clustering()
        else:
            st.info("💡 Complete Data Loading and Preprocessing first.")
    
    # TAB D: RESULTS
    with tabs[3]:
        if st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success"):
            tab_d_results()
        else:
            st.info("💡 Complete clustering to see results.")

# ============================================================================
# OPTIONAL: ANALYTICS SIDEBAR FOR DEVELOPMENT
# ============================================================================

def show_analytics_sidebar():
    """Optional analytics sidebar for development/admin"""
    if BACKEND_AVAILABLE and st.session_state.get('backend'):
        with st.sidebar:
            st.subheader("📊 Session Analytics")
            
            if st.button("Show Analytics"):
                session_summary = st.session_state.backend.get_session_analytics(st.session_state.session_id)
                
                st.write(f"**Session ID:** {st.session_state.session_id}")
                st.write(f"**Duration:** {session_summary.get('duration_seconds', 0):.0f}s")
                st.write(f"**Completion:** {session_summary.get('completion_percentage', 0):.0f}%")
                
                activity_counts = session_summary.get('activity_counts', {})
                if activity_counts:
                    st.write("**Activities:**")
                    for activity, count in activity_counts.items():
                        st.write(f"• {activity}: {count}")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize session state first
    initialize_session_state()
    
    # Run main app
    main()
    
    # Optional: Show analytics in sidebar during development
    # Uncomment the line below to enable analytics sidebar
    # show_analytics_sidebar()