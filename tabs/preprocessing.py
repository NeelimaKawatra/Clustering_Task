# tabs/preprocessing.py - Fixed version with simplified data alignment
import streamlit as st
import pandas as pd

def tab_b_preprocessing(backend_available):
    """Tab B: Simplified Text Preprocessing with clear data alignment"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(
            st.session_state.session_id, "tab_visit", {"tab_name": "preprocessing"}
        )
    
    # Check prerequisites
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
        return
    
    st.subheader("Choose Preprocessing Level")
    
    # Get and store original texts consistently
    original_texts = df[text_column].fillna("").astype(str).tolist()
    st.session_state.original_texts = original_texts
    
    # Get recommendations
    try:
        recommendations = st.session_state.backend.get_preprocessing_recommendations(
            original_texts, st.session_state.session_id
        )
    except Exception as e:
        st.warning(f"Could not get recommendations: {e}")
        recommendations = {'suggested_method': 'basic'}

    # Show current data status
    st.info(f"Ready to process {len(original_texts)} texts from your data")

    # Preprocessing options
    with st.expander("Understanding Preprocessing Options"):
        st.write("- **No Preprocessing**: Use your text exactly as uploaded")
        st.write("- **Basic Preprocessing**: Remove URLs, emails, normalize whitespace") 
        st.write("- **Advanced Preprocessing**: Basic + remove stopwords, punctuation, numbers")
        st.write("- **Custom Preprocessing**: Choose specific settings")

    preprocessing_option = st.radio(
        "Select preprocessing level:",
        [
            "No Preprocessing",
            "Basic Preprocessing", 
            "Advanced Preprocessing",
            "Custom Preprocessing"
        ],
        index=1 if recommendations.get('suggested_method') == 'basic' else 2
    )

    # Custom settings
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

    # Process button
    if st.button("Process Text", type="primary"):
        with st.spinner("Processing text..."):
            
            # Map options to methods
            method_mapping = {
                "No Preprocessing": "none",
                "Basic Preprocessing": "basic", 
                "Advanced Preprocessing": "advanced",
                "Custom Preprocessing": "custom"
            }
            
            method = method_mapping[preprocessing_option]
            
            try:
                # Process texts using backend
                processed_texts, metadata = st.session_state.backend.preprocess_texts(
                    original_texts, method, custom_settings, st.session_state.session_id
                )
                
                # Store results with simplified alignment approach
                st.session_state.processed_texts = processed_texts
                st.session_state.preprocessing_metadata = metadata
                st.session_state.preprocessing_settings = {
                    'method': method,
                    'details': f"{preprocessing_option} applied",
                    'custom_settings': custom_settings if method == "custom" else {}
                }
                
                # Store simplified row alignment - just the valid indices
                st.session_state.row_alignment = metadata.get('valid_row_indices', list(range(len(processed_texts))))
                
                st.success(f"Processing complete! {len(processed_texts)} texts ready for clustering.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                return

    # Show results if processing is complete
    if st.session_state.processed_texts is not None:
        show_processing_results()

def show_processing_results():
    """Display processing results in a clear format"""
    
    processed_texts = st.session_state.processed_texts
    original_texts = st.session_state.original_texts
    row_alignment = st.session_state.get('row_alignment', [])
    
    st.subheader("Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Rows", len(original_texts))
    with col2:
        st.metric("Valid Rows After Preprocessing", len(processed_texts))
    with col3:
        filtered_count = len(original_texts) - len(processed_texts)
        st.metric("Filtered Out Rows", filtered_count)
    with col4:
        success_rate = (len(processed_texts) / len(original_texts)) * 100
        st.metric("Kept Rows Percentage", f"{success_rate:.1f}%")
    
    # Before/After comparison
    st.subheader("Before / After Comparison")

    with st.expander("Sample Comparisons", expanded=True):
        comparison_data = []
        
        # Show all comparisons using the alignment, filtering out empty texts
        for i in range(len(processed_texts)):
            original_idx = row_alignment[i] if i < len(row_alignment) else i
            
            # Ensure we don't go out of bounds
            if original_idx < len(original_texts):
                original_text = original_texts[original_idx]
                processed_text = processed_texts[i]
                
                # Only include non-empty texts
                if original_text.strip() and processed_text.strip():
                    comparison_data.append({
                        'Row': original_idx + 1,
                        'Original Text': original_text,
                        'Processed Text': processed_text
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=400)
            
            st.caption(f"Showing all {len(comparison_data)} non-empty text comparisons")
        else:
            st.warning("No valid comparisons to show")

        """ st.table to left align the text
        def show_comparisons(df, align="left"):
            assert align in ("left", "center")
            styler = (
                df.style
                .set_properties(**{"text-align": align})
                .set_table_styles([{"selector": "th", "props": [("text-align", align)]}])
            )
            st.table(styler)
        show_comparisons(comparison_df, align="left") 
        """


    # Quality check and completion
    if len(processed_texts) >= 10:
        st.success("Processing Complete! Ready for clustering.")
        
        # Auto-complete
        if not st.session_state.get('tab_b_complete', False):
            st.session_state.tab_b_complete = True
            # AUTO-NAVIGATE
            from utils.session_state import auto_navigate_to_next_available
            auto_navigate_to_next_available()
            #st.balloons()
            
            # Track completion
            if hasattr(st.session_state, 'backend') and st.session_state.backend:
                st.session_state.backend.track_activity(
                    st.session_state.session_id, "preprocessing", {
                        "method": st.session_state.preprocessing_settings['method'],
                        "original_count": len(original_texts),
                        "final_count": len(processed_texts)
                    }
                )
        
        st.info("Proceed to the **Clustering** tab to analyze your processed texts.")
        
        # Option to reprocess
        if st.button("Redo Preprocessing with Different Settings"):
            # Clear preprocessing results
            st.session_state.processed_texts = None
            st.session_state.preprocessing_metadata = {}
            st.session_state.row_alignment = []
            st.session_state.tab_b_complete = False
            # Cascade to clustering
            from utils.session_state import cascade_from_preprocessing
            cascade_from_preprocessing()
            st.rerun()
            
    else:
        st.error(f"Need at least 10 valid texts for clustering. Current: {len(processed_texts)}")
        st.info("Try using less aggressive preprocessing settings or check your data quality.")
        st.session_state.tab_b_complete = False
        
    # Show processing details
    with st.expander("Preprocessing Details"):
        settings = st.session_state.preprocessing_settings
        st.write(f"**Method**: {settings['method']}")
        st.write(f"**Description**: {settings['details']}")
        if settings.get('custom_settings'):
            st.write("**Custom Settings**:")
            for key, value in settings['custom_settings'].items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        metadata = st.session_state.preprocessing_metadata
        if metadata:
            st.write(f"**Processing Time**: {metadata.get('processing_time', 'N/A'):.2f} seconds")
            st.write(f"**Texts Removed**: {metadata.get('texts_removed', 0)}")