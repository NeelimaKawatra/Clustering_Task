import streamlit as st
from collections import Counter
import pandas as pd

def tab_b_preprocessing(backend_available):
    """Tab B: Text Preprocessing with automatic completion and cascading"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "preprocessing"})
    
    # Check prerequisites first
    if not st.session_state.get('tab_a_complete', False):
        st.error("Please complete Data Loading first!")
        st.info("Go to the Data Loading tab to load and configure your data.")
        return
    
    if not backend_available:
        st.error("Backend services not available. Please check backend installation.")
        return

    # Get data from previous tab
    df = st.session_state.df
    text_column = st.session_state.text_column
    
    if df is None or text_column is None:
        st.error("No data found. Please complete Data Loading first.")
        st.info("Go to the Data Loading tab to load and configure your data.")
        return
    
    st.subheader("Choose Preprocessing Level")
    
    # Get preprocessing recommendations from backend
    original_texts = df[text_column].dropna().tolist()
    recommendations = st.session_state.backend.get_preprocessing_recommendations(
        original_texts, st.session_state.session_id
    )

    # Expander to explain the different preprocessing options
    with st.expander("Understanding what each option does"):
        st.write("- No Preprocessing: use your original text")
        st.write("- Basic Preprocessing: remove URLs, email addresses, normalize case and whitespace")
        st.write("- Advanced Preprocessing: Basic Preprocessing + more powerful tokenization + remove stopwords, punctuation, numbers, short words, etc.")
        st.write("- Custom Preprocessing: choose your own settings")

    # Preprocessing options
    preprocessing_option = st.radio(
        label="",
        options=[
                "No Preprocessing",
                "Basic Preprocessing",
                "Advanced Preprocessing",
                "Custom Preprocessing"
            ],
        index=1 if recommendations['suggested_method'] == 'basic' else 2
    )

    # Custom preprocessing options
    custom_settings = {}
    if preprocessing_option == "Custom Preprocessing":
        st.subheader("Custom Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            custom_settings['remove_stopwords'] = st.checkbox("Remove stopwords", value=True)
            custom_settings['remove_punctuation'] = st.checkbox("Remove punctuation", value=True)
        with col2:
            custom_settings['min_length'] = st.slider("Minimum word length", 1, 5, 2)
            custom_settings['remove_numbers'] = st.checkbox("Remove numbers", value=True)
    
    # Process text using backend
    if st.button("Process Text", type="primary"):
        with st.spinner("Processing text..."):
            
            # Map preprocessing option to backend method
            method_mapping = {
                "No Preprocessing": "none",
                "Basic Preprocessing": "basic",
                "Advanced Preprocessing": "advanced",
                "Custom Preprocessing": "custom"
            }
            
            method = method_mapping[preprocessing_option]
            
            # Use backend preprocessing service
            processed_texts, metadata = st.session_state.backend.preprocess_texts(
                original_texts, method, custom_settings, st.session_state.session_id
            )
            
            # Store results - ensure original_texts matches processed_texts length
            # Filter original texts to match what the backend kept
            filtered_original_texts = []
            processed_index = 0
            
            for original_text in original_texts:
                # Apply same filtering logic as backend
                if (original_text and str(original_text).strip() and 
                    len(str(original_text).strip()) > 2 and 
                    processed_index < len(processed_texts)):
                    filtered_original_texts.append(original_text)
                    processed_index += 1
            
            # Store aligned arrays
            st.session_state.original_texts = filtered_original_texts
            st.session_state.processed_texts = processed_texts
            st.session_state.preprocessing_settings = metadata
            
            st.success(f"Processing complete! {len(processed_texts)} texts ready for clustering.")
            
            # Auto-complete if conditions are met
            if len(processed_texts) >= 10:
                st.session_state.tab_b_complete = True
                st.success("Preprocessing completed automatically!")
                st.balloons()
            
            st.rerun()
    
    # Show results if processing is complete
    if st.session_state.processed_texts is not None:
        metadata = st.session_state.preprocessing_settings
        
        st.subheader("Before/After Analysis")
        
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
        
        # Text comparison
        st.subheader("Text Comparison")
        with st.expander("Detailed Review", expanded=True):
            # Ensure arrays are same length by matching valid texts
            original_texts_filtered = []
            processed_texts_display = st.session_state.processed_texts
            
            # Get the original texts that weren't filtered out
            valid_original_texts = [text for text in st.session_state.original_texts 
                                  if text and str(text).strip() and len(str(text).strip()) > 2]
            
            # Take only the same number as processed texts
            min_length = min(len(valid_original_texts), len(processed_texts_display))
            
            comparison_df = pd.DataFrame({
                "Original Text": valid_original_texts[:min_length],
                "Pre-processed Text": processed_texts_display[:min_length]
            })
            
            if len(st.session_state.original_texts) > len(processed_texts_display):
                filtered_count = len(st.session_state.original_texts) - len(processed_texts_display)
                st.info(f"Note: {filtered_count} texts were filtered out during processing (empty, too short, or whitespace only)")
            
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
                    
        # Processing summary
        with st.expander("Preprocessing Summary"):
            st.write(f"**Method:** {metadata['method']}")
            st.write(f"**Details:** {metadata['details']}")
            st.write(f"**Processing time:** {metadata['processing_time']:.2f} seconds")
            st.write(f"**Results:** {metadata['filtered_count']} valid texts from {metadata['original_count']} original")
            if metadata.get('custom_settings'):
                st.write(f"**Custom settings:** {metadata['custom_settings']}")
        
        # Step status with celebration message
        st.markdown("---")
        st.subheader("Step Status")
        
        if len(st.session_state.processed_texts) >= 10:
            if not st.session_state.get('tab_b_complete', False):
                # Auto-complete preprocessing
                st.session_state.tab_b_complete = True
                
                # Track preprocessing completion
                st.session_state.backend.track_activity(st.session_state.session_id, "preprocessing", {
                    "method": metadata['method'],
                    "settings": metadata.get('custom_settings', {}),
                    "original_count": metadata['original_count'],
                    "final_count": metadata['filtered_count'],
                    "processing_time": metadata['processing_time']
                })
                
                st.balloons()
                st.success("Text Preprocessing Complete!")
                st.info("Your texts are cleaned and ready! Navigate to the **Clustering** tab to discover patterns and group similar texts together.")
            else:
                # Already completed - show status
                st.success("Text Preprocessing Complete")
                st.info("Your texts are processed and ready for clustering. Head to the **Clustering** tab to continue.")
                
            # Option to reprocess
            if st.button("Reprocess with Different Settings", 
                        help="Change preprocessing method"):
                from utils.session_state import cascade_from_preprocessing
                cascade_from_preprocessing()
                st.session_state.tab_b_complete = False
                st.session_state.processed_texts = None
                st.session_state.preprocessing_settings = {}
                st.info("Preprocessing reset. You can now modify settings above.")
                st.rerun()
        else:
            st.error(f"Need at least 10 texts for clustering. Current: {len(st.session_state.processed_texts)}")
            st.write("Try using less aggressive preprocessing or check your data quality.")
            
            # Reset completion if insufficient texts
            if st.session_state.get('tab_b_complete', False):
                st.session_state.tab_b_complete = False