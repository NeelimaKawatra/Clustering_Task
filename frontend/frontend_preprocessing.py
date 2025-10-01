# frontend/frontend_preprocessing.py - Proper fix with correct reset timing
import streamlit as st
import pandas as pd
from utils.reset_manager import reset_from_preprocessing_change

def tab_preprocessing(backend_available):
    """Tab: Simplified Text Preprocessing with proper reset timing"""
    
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
    entry_column = st.session_state.entry_column
    
    if df is None or entry_column is None:
        st.error("No data found. Please complete Data Loading first.")
        return
    
    st.subheader("Choose Preprocessing Level")
    st.markdown("Clean and prepare your text entries for optimal clustering results.")
    
    # Get and store original texts consistently
    original_texts = df[entry_column].fillna("").astype(str).tolist()
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
    st.success(f"Ready to preprocess {len(original_texts)} text entries from your data")

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

    # Create method signature for change detection
    current_method = f"{preprocessing_option}_{str(custom_settings) if preprocessing_option == 'Custom Preprocessing' else ''}"
    previous_method = st.session_state.get('last_preprocessing_method', '')
    
    # Check what will be affected by reprocessing
    method_changed = (current_method != previous_method and previous_method != '')
    has_clustering = bool(st.session_state.get('clustering_results'))
    has_finetuning = bool(st.session_state.get('finetuning_results'))
    
    # Show impact warning BEFORE the button
    if method_changed and (has_clustering or has_finetuning):
        affected_steps = []
        if has_clustering:
            affected_steps.append("clustering results")
        if has_finetuning:
            affected_steps.append("fine-tuning work")
        
        st.warning(f"⚠️ **Changing preprocessing method will reset:** {', '.join(affected_steps)}")
        st.info("💡 The reset will happen AFTER the new preprocessing completes successfully.")

    # Process button
    if st.button("Preprocess Text", type="primary"):
        
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
                # STEP 1: Process texts using backend
                processed_texts, metadata = st.session_state.backend.preprocess_texts(
                    original_texts, method, custom_settings, st.session_state.session_id
                )
                
                # STEP 2: Store ALL new results first (before any resets)
                st.session_state.processed_texts = processed_texts
                st.session_state.preprocessing_metadata = metadata
                st.session_state.preprocessing_settings = {
                    'method': method,
                    'details': f"{preprocessing_option} applied",
                    'custom_settings': custom_settings if method == "custom" else {}
                }
                st.session_state.row_alignment = metadata.get('valid_row_indices', list(range(len(processed_texts))))
                st.session_state.last_preprocessing_method = current_method
                
                # STEP 3: Mark preprocessing as complete
                st.session_state.tab_preprocessing_complete = True
                if 'permanent_progress' in st.session_state:
                    st.session_state.permanent_progress['preprocessing'] = True
                
                # STEP 4: NOW reset downstream if method changed or there were existing results
                if method_changed or has_clustering or has_finetuning:
                    # Clear downstream results manually (more reliable than unified reset during processing)
                    downstream_cleared = []
                    
                    if 'clustering_results' in st.session_state:
                        del st.session_state['clustering_results']
                        downstream_cleared.append("clustering results")
                    
                    # Clear all finetuning keys
                    for key in list(st.session_state.keys()):
                        if key.startswith('finetuning_'):
                            del st.session_state[key]
                            if "fine-tuning" not in downstream_cleared:
                                downstream_cleared.append("fine-tuning data")
                    
                    # Update permanent progress
                    if 'permanent_progress' in st.session_state:
                        st.session_state.permanent_progress['clustering'] = False
                    
                    if downstream_cleared:
                        st.info(f"✅ Cleared: {', '.join(downstream_cleared)}")
                
                # STEP 5: Track completion
                if not st.session_state.get('preprocessing_tracked', False):
                    if hasattr(st.session_state, 'backend') and st.session_state.backend:
                        st.session_state.backend.track_activity(
                            st.session_state.session_id, "preprocessing", {
                                "method": method,
                                "original_count": len(original_texts),
                                "final_count": len(processed_texts),
                                "downstream_reset": method_changed or has_clustering or has_finetuning
                            }
                        )
                    st.session_state.preprocessing_tracked = True
                
                # STEP 6: Show success message
                st.success(f"✅ Processing complete! {len(processed_texts)} preprocessed text entries ready for clustering.")
                
                # STEP 7: Single rerun to update UI
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
            st.dataframe(comparison_df, width="stretch", hide_index=True, height=400)
            
            st.caption(f"Showing all {len(comparison_data)} non-empty text comparisons")
        else:
            st.warning("No valid comparisons to show")

    # Quality check and completion status
    if len(processed_texts) >= 0:
        st.success("Preprocessing Complete!")
        st.info("Proceed to the **Clustering** tab to analyze your preprocessed text entries.")
        
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
            st.write(f"**Text Entries Filtered Out**: {metadata.get('texts_removed', 0)}")
        
        # Show what was reset (if anything)
        current_method = st.session_state.get('last_preprocessing_method', '')
        if current_method:
            st.write(f"**Current Method Signature**: {current_method}")
        
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
                st.dataframe(filt_df, width="stretch", hide_index=True, height=300)
            else:
                st.caption("No rows were filtered out.")