import streamlit as st
from collections import Counter

def tab_b_preprocessing(backend_available):
    """Tab B: Text Preprocessing using backend services"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "preprocessing"})
    
    st.header("🔧 Text Preprocessing")
    
    # Get data from previous tab
    df = st.session_state.df
    text_column = st.session_state.text_column
    
    if df is None or text_column is None:
        st.error("❌ No data found. Please complete Data Loading first.")
        return
    
    if not backend_available:
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
        
        """
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
        """
                    
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