# tabs/data_loading.py - Complete version with enhanced change detection
import streamlit as st
import os
import re
from utils.helpers import get_file_from_upload
from utils.session_state import reset_analysis
import pandas as pd

def get_safe_index(options_list, value, default=0):
    """Safely get index of value in list, return default if not found"""
    try:
        if value is not None and value in options_list:
            return options_list.index(value)
        else:
            return default
    except (ValueError, TypeError):
        return default

def handle_column_selection_change(new_selection, current_selection, selection_type):
    """Handle column selection changes and trigger appropriate resets"""
    
    # Define what constitutes a meaningful change
    prompt_values = {
        "id_column_prompt": ["-- Select any column --", None, "Auto-generate IDs"],
        "text_column_prompt": ["-- Select a text column for clustering --", None]
    }
    
    # Check if this is a meaningful change
    is_meaningful_change = (
        new_selection != current_selection and
        current_selection not in prompt_values.get(selection_type, []) and
        new_selection not in prompt_values.get(selection_type, []) and
        (st.session_state.get('tab_preprocessing_complete') or st.session_state.get('clustering_results'))
    )
    
    if is_meaningful_change:
        # Show warning about what will be reset
        st.warning(f"⚠️ {selection_type.title()} column changed from '{current_selection}' to '{new_selection}'")
        
        with st.expander("What will be reset?", expanded=True):
            reset_items = []
            if st.session_state.get('clustering_results'):
                reset_items.append("🔍 Clustering results")
            if st.session_state.get('tab_preprocessing_complete'):
                reset_items.append("🧹 Text preprocessing")
            
            if reset_items:
                st.write("The following will be reset due to this change:")
                for item in reset_items:
                    st.write(f"• {item}")
                st.write("You'll need to run these steps again with your new column selection.")
        
        # Perform the cascade reset
        from utils.session_state import cascade_from_data_loading
        cascade_from_data_loading()
        st.success("✅ Reset complete. Your new column selection is saved.")
        
        # Mark that changes were made
        st.session_state.data_loading_changes_made = True

def tab_data_loading(backend_available):
    """Tab: Data Loading with persistent selections and smart change detection"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "data_loading"})
    
    # Introduction section
    st.markdown("""
    Welcome to Clustery! Start by uploading your data file containing text you want to cluster.
    
    **Supported formats:** CSV, Excel (.xlsx, .xls)  
    **Requirements:** At least 10 rows of text data
    
    **Note:** An `entryID` column (row numbers) will be automatically added to your data for tracking purposes.
    """)
    
    # Check if data already exists in session state
    data_already_loaded = 'df' in st.session_state and st.session_state.df is not None
    
    # File upload section
    st.subheader("Upload Your File")
    upload_key = st.session_state.get('file_uploader_key', 'data_file_uploader')
    
    # Show message if file uploader was recently reset
    if 'file_uploader_reset' in st.session_state and st.session_state.file_uploader_reset:
        st.info("📁 File cleared. Please upload a new file to restart the analysis.")
        st.session_state.file_uploader_reset = False
    

    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls"],
        help="Upload your survey data or text file for clustering",
        key=upload_key,
        label_visibility="collapsed"
    )

    # Check if a new file was uploaded (different from previous)
    current_file_key = None
    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
    
    # Detect file change and reset analysis if needed
    file_changed = False
    if 'previous_file_key' in st.session_state:
        if current_file_key != st.session_state.previous_file_key and current_file_key is not None:
            # New file detected - this should reset everything
            st.warning("🔄 New file detected - resetting entire analysis")
            reset_analysis()
            file_changed = True
    else:
        # First time loading
        st.session_state.previous_file_key = current_file_key
        file_changed = bool(current_file_key)

    
    # Process file upload if provided and changed
    if uploaded_file is not None and file_changed:
        if not backend_available:
            st.error("Backend services not available. Please check backend installation.")
            return
        
        try:
            temp_file_path = get_file_from_upload(uploaded_file)
            
            with st.spinner("Loading and validating file..."):
                success, df, message = st.session_state.backend.load_data(temp_file_path, st.session_state.session_id)
            
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            if not success:
                st.error(f"{message}")
                return
            
            # Add entryID column with original row numbers (now df is original df + "entryID" column)
            df = df.copy()
            df['entryID'] = range(1, len(df) + 1)
            
            # Store dataframe
            st.session_state.df = df
            st.session_state.previous_file_key = current_file_key
            st.session_state.tab_data_loading_complete = False
            
            st.success(f"{message}")
            #st.balloons()
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
            # Provide helpful error guidance
            with st.expander("Troubleshooting Help"):
                st.markdown("""
                **Common issues and solutions:**
                
                - **File format:** Ensure your file is a valid CSV or Excel format
                - **Encoding:** Try saving your CSV with UTF-8 encoding
                - **File size:** Large files (>10MB) may take longer to process
                - **Column names:** Avoid special characters in column headers
                - **Data quality:** Ensure your file isn't corrupted
                
                **Need help?** Check that your file:
                1. Opens correctly in Excel or a text editor
                2. Has clear column headers
                3. Contains the text data you want to analyze
                """)
            return
    
    # If no file uploaded and no data exists, return
    if not data_already_loaded and uploaded_file is None:
        return
    
    # From here, we have data loaded - display all configuration sections
    df = st.session_state.df
    
    # Ensure entryID exists even if df came from an older session
    if 'entryID' not in df.columns:
        df = df.copy()
        df.insert(0, 'entryID', range(1, len(df) + 1))
        st.session_state.df = df

    # Ensure dataframe is valid before proceeding
    if df is None or df.empty:
        st.error("No data loaded. Please upload a file first.")
        return
    

    # File Overview Section
    st.subheader("File Overview")
    
    # Show file metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        file_name = getattr(uploaded_file, 'name', 'Loaded Data') if uploaded_file else 'Loaded Data'
        st.metric("File Name", file_name)
    with metric_col2:
        st.metric("Total Rows", len(df))
    with metric_col3:
        st.metric("Total Columns", len(df.columns))
    with metric_col4:
        # Count text columns - object dtype columns that aren't entryID
        text_cols = sum(1 for col in df.columns 
                       if col != 'entryID' and df[col].dtype == 'object')
        st.metric("Text Columns", text_cols)
    
    
    # Data Overview Section
    with st.expander("Data Preview", expanded=True):
        st.markdown("**Your Loaded Data (first 300 rows):**")
        cols = ['entryID'] + [c for c in df.columns if c != 'entryID']
        st.dataframe(df[cols], use_container_width=True, hide_index=True)


        # Column Statistics
        st.markdown("**Column Statistics:**")
        
        # Create statistics with columns as columns and metrics as rows
        stats_data = {}
        
        # Initialize the stats dictionary with each column
        for col in df.columns:
            total_rows = len(df)
            empty_rows = int(df[col].isna().sum() + (df[col] == '').sum())
            non_empty_rows = int(total_rows - empty_rows)
            
            # Special handling for entryID column
            if col == 'entryID':
                col_type = 'Entry ID (Auto-generated)'
            else:
                col_type = 'Text' if df[col].dtype == 'object' else 'Non-Text'
            
            stats_data[col] = {
                'Total Rows': total_rows,
                'Empty Rows': empty_rows,
                'Non-Empty Rows': non_empty_rows,
                'Column Type': col_type
            }
        
        # Create DataFrame with metrics as index and columns as columns
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    


    # Column Selection Section
    st.subheader("Column Selection")

    # subject id column selection
    st.markdown("**Step 1: Choose a column for subject identification**")
    auto_option = " use entryID (row numbers) as subject IDs"
    prompt_option = "-- Select a subject ID column--"

    # Include entryID as a selectable option
    available_columns = [col for col in df.columns if col != 'entryID']
    id_options = [prompt_option, auto_option] + available_columns
    
    # Ensure we have valid options
    if not id_options or len(id_options) == 0:
        st.error("No valid options available for ID selection.")
        return
    
    # get the current subject id selection
    current_id_selection = st.session_state.get('subjectID', prompt_option) or prompt_option

    # Display "entryID" as the auto label in the UI
    if current_id_selection == "entryID":
        current_id_selection = auto_option
    if current_id_selection not in id_options:
        current_id_selection = prompt_option

    # get the index of the current subject id selection
    id_column_index = get_safe_index(id_options, current_id_selection, 0)

    # selectbox for the subject id column
    selected_id = st.selectbox(
        label="Choose a column for ID:",
        label_visibility="collapsed",
        options=id_options,
        index=id_column_index,
        help="Select a column to track individual responses"
    )

    # Map any auto label to the real column
    if selected_id == auto_option:
        selected_id = "entryID"

    # get the previous subject id selection
    prev = st.session_state.get('subjectID')
    if selected_id != prev:
        handle_column_selection_change(selected_id, prev, "ID")

    # save the selected subject id to the session state 
    st.session_state.subjectID = selected_id

    # Show sample subject IDs
    if selected_id and selected_id != prompt_option:
        try:
            col_to_show = "entryID" if selected_id == "entryID" else selected_id
            sample_ids = df[col_to_show].head(10).tolist()
            if sample_ids:
                formatted = ", ".join([f'"{str(x)}"' for x in sample_ids])
                st.caption(f"**Sample subject IDs: {{{formatted}}}**")
        except (KeyError, AttributeError):
            st.caption("Selected ID column not accessible")


    # Show ID column preview and validation
    if selected_id and selected_id not in [auto_option, prompt_option]:
        try:
            id_column_data = df[selected_id]
            
            if selected_id == 'entryID':
                pass
                #st.success(f"Using entryID column: {selected_id} (auto-generated row numbers)")
            else:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(id_column_data):
                    #st.success(f"Numeric ID column selected: {selected_id}")
                    pass
                else:
                    st.warning(f"Non-numeric column selected. ID columns work best with numbers.")
                
        except (KeyError, TypeError):
            st.error(f"Column '{selected_id}' not found in data")
            return
    elif selected_id == auto_option:
        pass
        #st.info("Will provide you with your row numbers as your subject IDs.")
    

    # Text Column Section with Enhanced Change Detection
    st.markdown("**Step 2: Choose a Text Column for Clustering**")
    prompt_option_text = "-- Select a text column --"
    
    # Get text column suggestions first to filter options
    text_columns = []
    if backend_available:
        try:
            text_columns = st.session_state.backend.get_text_column_suggestions(df, st.session_state.session_id)
        except AttributeError:
            # Simple fallback - check for text-like columns, excluding entryID
            text_columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) and col != 'entryID']
    else:
        # Fallback when backend not available, excluding entryID
        text_columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) and col != 'entryID']
    
    # Create filtered options - only show text columns plus prompt
    text_options = [prompt_option_text] + text_columns
    
    # FIXED: Preserve existing text column selection
    current_text_selection = st.session_state.get('text_column', prompt_option_text)
    
    # Only reset if no previous selection or selection is invalid
    if current_text_selection is None:
        current_text_selection = prompt_option_text
    
    # Ensure the current selection is valid for current data
    if current_text_selection not in text_options:
        current_text_selection = prompt_option_text
    
    text_column_index = get_safe_index(text_options, current_text_selection, 0)
    
    selected_text_column = st.selectbox(
        label="Choose a text column:",
        label_visibility="collapsed",
        options=text_options,
        index=text_column_index,
        help="Choose the column with text you want to cluster",
        key="text_selector"
    )
    
    # Enhanced change detection for text column
    if selected_text_column != st.session_state.get('text_column'):
        handle_column_selection_change(selected_text_column, st.session_state.get('text_column'), "text")
        st.session_state.text_column = selected_text_column
    
    # Store user selections for output structure
    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {}

    # Only update user selections if valid columns are selected
    if selected_text_column != prompt_option_text and selected_id != prompt_option:
        st.session_state.user_selections.update({
            'id_column_choice': selected_id,
            'text_column_choice': selected_text_column,
            'id_is_auto_generated': selected_id == "entryID", 
            'subjectID': selected_id,
            'original_columns': [selected_id, selected_text_column] if selected_id != 'entryID' else [selected_text_column]
        })


    # Show feedback about text column detection
    if selected_text_column and selected_text_column != prompt_option_text:
        if text_columns:
            #st.success(f"Text column selected: {selected_text_column}" )
            # telling how many text columns were detected
            #st.success(f"Text column selected from {len(text_columns)} text columns")
            pass
        else:
            st.warning("No obvious text columns detected. Please verify your selection.")
    
    # Show 10 sample texts with improved formatting and safety
    if selected_text_column and selected_text_column != prompt_option_text:
        try:
            sample_texts = df[selected_text_column].dropna().head(10).tolist()
            if sample_texts:
                formatted_samples = ", ".join([f'"{str(text)[:100] + "..." if len(str(text)) > 100 else str(text)}"' for text in sample_texts])
                st.caption(f"**Sample texts: {{{formatted_samples}}}**")
        except (KeyError, AttributeError):
            st.caption("Selected text column not accessible")

    # Validation and Quality Analysis
    if (selected_text_column and selected_text_column != prompt_option_text and 
        selected_text_column in df.columns):
        st.subheader("Text Column Quality Analysis")
        
        with st.spinner("Analyzing data quality..."):
            validation_result = st.session_state.backend.validate_columns(
                df, 
                selected_text_column, 
                selected_id if selected_id != prompt_option else None, 
                st.session_state.session_id
            )
        
        if validation_result["text_column_valid"]:
            st.success(f"{validation_result['text_column_message']}")
            
            # Show text quality metrics in an attractive layout
            stats = validation_result["text_quality"]
            
            st.markdown("**Text Quality Metrics**")
            quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
            
            with quality_col1:
                st.metric(
                    "Valid Texts", 
                    stats['total_texts'] - stats['empty_texts'],
                    delta=f"of {stats['total_texts']} total"
                )
            with quality_col2:
                st.metric(
                    "Avg Length", 
                    f"{stats['avg_length']:.0f} chars",
                    delta=f"Sample size: {stats['sample_size']}"
                )
            with quality_col3:
                st.metric(
                    "Avg Words", 
                    f"{stats['avg_words']:.1f}",
                    delta=f"Sample analysis"
                )
            with quality_col4:
                st.metric(
                    "Unique Texts", 
                    stats['unique_texts'],
                    delta=f"No duplicates/NA"
                )
            
            # Sample texts in a nice format
            with st.expander("Sample Texts Analysis", expanded=False):
                st.markdown("**Representative samples from your text data:**")
                sample_texts = df[selected_text_column].dropna().head(5)
                
                for i, text in enumerate(sample_texts, 1):
                    text_str = str(text)
                    word_count = len(text_str.split())
                    char_count = len(text_str)
                    
                    with st.container():
                        col_text, col_stats = st.columns([4, 1])
                        
                        with col_text:
                            st.markdown(f"**Sample {i}:**")
                            display_text = text_str[:300] + "..." if len(text_str) > 300 else text_str
                            st.markdown(f"*{display_text}*")
                        
                        with col_stats:
                            st.caption(f"Words: {word_count}")
                            st.caption(f"Chars: {char_count}")
                        
                        st.markdown("---")

            
            # Ready to proceed section
            st.markdown("---")
            st.subheader("Ready to Proceed")
            
            # Summary display with better styling
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown("**Data Summary:**")
                st.write(f"• **Total rows:** {len(df):,}")
                st.write(f"• **Valid texts:** {stats['total_texts'] - stats['empty_texts']:,}")
                st.write(f"• **Data quality:** {'Excellent' if stats['avg_length'] > 50 else 'Good' if stats['avg_length'] > 20 else 'Fair'}")
            
            with summary_col3:
                st.markdown("**Text Configuration:**")
                st.write(f"• **Column:** {selected_text_column}")
                st.write(f"• **Avg length:** {stats['avg_length']:.0f} characters")
                st.write(f"• **Avg words:** {stats['avg_words']:.1f} words")
            
            with summary_col2:
                st.markdown("**Subject ID Configuration:**")
                if selected_id and selected_id != prompt_option:
                    if selected_id == 'entryID':
                        st.write("• **Source:** entryID (row numbers)")
                        st.write("• **Type:** Auto-generated")
                        st.write("• **Format:** 1, 2, 3, 4...")
                    else:
                        st.write(f"• **Column:** {selected_id}")
                        col_type = 'Numeric' if pd.api.types.is_numeric_dtype(df[selected_id]) else 'Text'
                        st.write(f"• **Type:** {col_type}")
                        #st.write(f"• **Status:** {id_analysis['status'].title()}")
                        #st.write(f"• **Unique:** {id_analysis['unique']:,} IDs")
                else:
                    st.write("• **Type:** Auto-generated")
                    st.write("• **Format:** ID_001, ID_002...")
                    st.write("• **Count:** Sequential numbering")
            
            # Auto-completion with celebration message
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Check if step should auto-complete
            if not st.session_state.get('tab_data_loading_complete', False):
                # Auto-complete when validation passes
                st.session_state.tab_data_loading_complete = True

            
                
                # Track completion
                if backend_available:
                    st.session_state.backend.track_activity(st.session_state.session_id, "data_upload", {
                        "filename": "loaded_data",
                        "rows": len(df),
                        "columns": len(df.columns),
                        "text_column": selected_text_column,
                        "id_column": selected_id,
                        "text_quality": stats
                    })
                            
                # AUTO-NAVIGATE: Add this here
                from utils.session_state import auto_navigate_to_next_available
                auto_navigate_to_next_available()
                
                #st.balloons()
                # Show completion message
                st.success("Data Loading Complete!")
                st.info("Your data is ready! Head over to the **Preprocessing** tab to clean and prepare your text data for clustering.")
                
            else:
                # Already completed - just show status
                st.success("Data Loading Complete!")
                st.info("Your data configuration is saved. You can proceed to **Preprocessing** or modify settings above to trigger automatic reset.")
            
            # Show feedback if changes were made during this session
            if st.session_state.get('data_loading_changes_made'):
                st.info("💡 **Tip:** Your changes have been saved and downstream processing has been reset. Navigate to the next tab to continue.")
                # Clear the flag
                del st.session_state['data_loading_changes_made']
            
            # Additional tips
            st.markdown("---")
            with st.expander("Tips for Better Results", expanded=False):
                st.markdown("""
                **For optimal clustering results:**
                
                - Text length: Texts with 20+ words work best
                - Data quality: Remove or fix obviously corrupted entries
                - Language: Ensure all texts are in the same language
                - Relevance: All texts should be about similar topics
                - Volume: 50+ texts recommended for meaningful clusters
                
                **What happens next:**
                1. **Preprocessing:** Clean and prepare your texts
                2. **Clustering:** Run advanced algorithms to find patterns
                3. **Results:** Explore and export your findings
                """)
        
        else:
            st.error(f"{validation_result['text_column_message']}")
            # Provide helpful suggestions
            st.markdown("**Suggestions to fix this issue:**")
            st.markdown("""
            - Choose a different column that contains longer text
            - Ensure the column has meaningful sentences, not just single words
            - Check that the column isn't mostly empty or contains mostly numbers/codes
            - Look for columns with survey responses, comments, or descriptions
            """)
            
            # Reset completion if validation fails
            if st.session_state.get('tab_data_loading_complete', False):
                st.session_state.tab_data_loading_complete = False
                st.rerun()

            # In your data loading tab, when completion happens:
            st.session_state.tab_data_loading_complete = True
            from utils.session_state import auto_navigate_to_next_available
            auto_navigate_to_next_available()
            st.rerun()