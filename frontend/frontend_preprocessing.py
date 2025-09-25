# tabs/preprocessing.py - Clean version without debug code
import streamlit as st
import pandas as pd

def reset_downstream_from_preprocessing():
    """Reset everything downstream from preprocessing"""
    # Reset current completion status
    if 'clustering_results' in st.session_state:
        del st.session_state['clustering_results']
    
    # Reset permanent progress for downstream steps
    if 'permanent_progress' in st.session_state:
        st.session_state.permanent_progress['clustering'] = False
    
    # Clear finetuning
    for key in list(st.session_state.keys()):
        if key.startswith('finetuning_'):
            del st.session_state[key]

def tab_preprocessing(backend_available):
    """Tab: Simplified Text Preprocessing with clear data alignment"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(
            st.session_state.session_id, "tab_visit", {"tab_name": "preprocessing"}
        )
    
    # Check prerequisites
    if not st.session_state.get('tab_data_loading_complete', False):
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
    st.success(f"Ready to preprocess {len(original_texts)} texts from your data")

    # Preprocessing options
    with st.expander("Understanding Preprocessing Options"):
        st.write("- **No Preprocessing**: Use your text exactly as uploaded")
        st.write("- **Basic Preprocessing**: Remove URLs, emails, normalize whitespace") 
        st.write("- **Advanced Preprocessing**: Basic Preprocessing + remove stopwords, punctuation, numbers")
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

    # Detect preprocessing method changes
    current_method = f"{preprocessing_option}_{str(custom_settings) if preprocessing_option == 'Custom Preprocessing' else ''}"
    previous_method = st.session_state.get('last_preprocessing_method', '')

    # Process button
    if st.button("Preprocess Text", type="primary"):
        # Check if method changed and there's downstream work
        method_changed = (current_method != previous_method and 
                         previous_method != '' and
                         st.session_state.get('clustering_results'))
        
        if method_changed:
            st.warning("🔄 Preprocessing method changed - resetting clustering results!")
            reset_downstream_from_preprocessing()
        
        # Store current method for future comparison
        st.session_state.last_preprocessing_method = current_method
        
        # Check if reprocessing will affect downstream work
        if st.session_state.get('clustering_results'):
            st.warning("🔄 Reprocessing text will reset your clustering results!")
            reset_downstream_from_preprocessing()
        
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
    processed_texts = st.session_state.get('processed_texts')
    if processed_texts is not None:
        show_processing_results()

def show_processing_results():
    """Display processing results in a clear format"""
    
    processed_texts = st.session_state.get('processed_texts', [])
    original_texts = st.session_state.get('original_texts', [])
    row_alignment = st.session_state.get('row_alignment', [])
    
    st.subheader("Preprocessing Results")
    
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
                    # Get the actual subject ID value from df using the selected ID column
                    df = st.session_state.df
                    subject_id_column = st.session_state.subjectID
                    actual_subject_id = df.iloc[original_idx][subject_id_column] if original_idx < len(df) else f"Row_{original_idx}"
                    
                    comparison_data.append({
                        'Subject ID': actual_subject_id,
                        'Original Text': original_text,
                        'Processed Text': processed_text
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=400)
            
            st.caption(f"Showing all {len(comparison_data)} non-empty text comparisons")
        else:
            st.warning("No valid comparisons to show")

    # Quality check and completion
    if len(processed_texts) >= 10:
        st.success("Preprocessing Complete!")
        
        # Check previous states to determine if we need to rerun
        was_tab_complete_before = st.session_state.get('tab_preprocessing_complete', False)
        was_permanent_complete_before = st.session_state.get('permanent_progress', {}).get('preprocessing', False)
        
        # Set completion flags
        st.session_state.tab_preprocessing_complete = True
        
        # Update permanent progress
        if 'permanent_progress' in st.session_state:
            st.session_state.permanent_progress['preprocessing'] = True
        
        # Track completion only once
        if not st.session_state.get('preprocessing_tracked', False):
            if hasattr(st.session_state, 'backend') and st.session_state.backend:
                st.session_state.backend.track_activity(
                    st.session_state.session_id, "preprocessing", {
                        "method": st.session_state.get('preprocessing_settings', {}).get('method', 'unknown'),
                        "original_count": len(original_texts),
                        "final_count": len(processed_texts)
                    }
                )
            st.session_state.preprocessing_tracked = True
        
        # Trigger rerun if EITHER flag changed from False to True (to update sidebar)
        should_rerun = (not was_tab_complete_before) or (not was_permanent_complete_before)
        
        if should_rerun:
            st.rerun()
        
        st.info("Proceed to the **Clustering** tab to analyze your preprocessed texts.")
        
    else:
        st.error(f"Need at least 10 valid texts for clustering. Current: {len(processed_texts)}")
        st.info("Try using less aggressive preprocessing settings or check your data quality.")
        st.session_state.tab_preprocessing_complete = False
        
    # Show processing details
    with st.expander("Preprocessing Details", expanded=False):
        settings = st.session_state.get('preprocessing_settings', {})
        st.write(f"**Method**: {settings.get('method', 'N/A')}")
        st.write(f"**Description**: {settings.get('details', 'N/A')}")
        if settings.get('custom_settings'):
            st.write("**Custom Settings**:")
            for key, value in settings['custom_settings'].items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        metadata = st.session_state.get('preprocessing_metadata', {})
        if metadata:
            st.write(f"**Processing Time**: {metadata.get('processing_time', 'N/A'):.2f} seconds")
            st.write(f"**Texts Entries Filtered Out**: {metadata.get('texts_removed', 0)}")
        
        # Collapsed dropdown of filtered-out rows
        try:
            original_texts = st.session_state.original_texts or []
            row_alignment = st.session_state.row_alignment or []
        except Exception:
            original_texts, row_alignment = [], []

        # Indices kept by preprocessing -> complement are filtered-out
        filtered_idx = sorted(set(range(len(original_texts))) - set(row_alignment))
        with st.expander("Filtered-out entries", expanded=False):
            if filtered_idx:
                df_src = st.session_state.get("df")
                id_col = st.session_state.get("subjectID", "entryID")

                rows = []
                for i in filtered_idx:
                    # Subject ID (safe fallback)
                    if isinstance(df_src, pd.DataFrame) and i < len(df_src) and id_col in df_src.columns:
                        sid = df_src.iloc[i][id_col]
                    else:
                        sid = i + 1  # 1-based row number fallback

                    # Original text -> display "None" if empty/NaN/whitespace
                    val = original_texts[i] if i < len(original_texts) else None
                    if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                        display_text = "None"
                    else:
                        display_text = str(val)

                    rows.append({"Subject ID": sid, "Original Text": display_text})

                filt_df = pd.DataFrame(rows)
                st.dataframe(filt_df, use_container_width=True, hide_index=True, height=300)
            else:
                st.caption("No rows were filtered out.")