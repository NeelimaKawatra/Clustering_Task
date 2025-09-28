# frontend/frontend_data_loading.py - Complete version with unified reset system
import streamlit as st
import os
import re
from utils import session_state
from utils.helpers import get_file_from_upload
from utils.reset_manager import reset_from_file_change, reset_from_column_change
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

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
    """Handle column selection changes using unified reset system"""
    
    # Define what constitutes a meaningful change
    prompt_values = {
        "ID": ["-- Select a subject ID column--", None, "use entryID (row numbers) as subject IDs"],
        "entry": ["-- Select an entry column --", None]
    }
    
    # Check if this is a meaningful change (simplified logic)
    is_meaningful_change = (
        new_selection != current_selection and
        current_selection not in prompt_values.get(selection_type, []) and
        new_selection not in prompt_values.get(selection_type, [])
    )
    
    # Also trigger if there's any downstream work that would be affected
    has_downstream_work = (
        st.session_state.get('tab_preprocessing_complete') or 
        st.session_state.get('clustering_results') or
        st.session_state.get('processed_texts')
    )
    
    if is_meaningful_change and has_downstream_work:
        # Use unified reset system
        reset_summary = reset_from_column_change(
            changed_column=selection_type,
            show_message=True
        )
        
        # Reset the OTHER column selection
        if selection_type == "ID":
            st.session_state.entry_column = "-- Select an entry column --"
        elif selection_type == "entry":
            st.session_state.subjectID = "-- Select a subject ID column--"
        
        st.success("✅ Reset complete. Please reselect your columns.")
        
        # Force immediate UI refresh
        st.rerun()

def tab_data_loading(backend_available):
    """Tab: Data Loading with unified reset system"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "data_loading"})
    
    # Introduction section
    st.markdown("""
    Welcome to Clustery! Start by uploading your data file containing text entries you want to cluster.
    
    **Supported formats:** CSV, Excel (.xlsx, .xls)  
    **Requirements:** Any number of rows with text entries
    
    **Note:** An `entryID` column (row numbers) will be automatically added to your data for tracking purposes.
    """)
    
    # --- Handle explicit 'Start New Analysis' first ---
    if (st.session_state.get("file_uploader_reset")
        and st.session_state.get("file_reset_reason") == "start_new_analysis"):
        st.info("📁 File cleared. Please upload a new file to restart the analysis.")
        st.session_state["file_uploader_reset"] = False
        st.session_state["file_reset_reason"] = None
        st.session_state["data_loading_alerts"] = []  

    # --- Persistent alerts (survive reruns) ---
    alerts = st.session_state.get("data_loading_alerts", [])
    for kind, text in alerts:
        if kind == "warning":
            st.warning(text)
        elif kind == "success":
            st.success(text)
        else:
            st.info(text)

    # Check if data already exists in session state
    data_already_loaded = 'df' in st.session_state and st.session_state.df is not None
    
    # File upload section
    st.subheader("Upload Your File")
    upload_key = st.session_state.get('file_uploader_key', 'data_file_uploader')
    
    # Show message if file uploader was recently reset
    if st.session_state.get('file_uploader_reset') and st.session_state.get('file_reset_reason') == "start_new_analysis":
        st.info("📁 File cleared. Please upload a new file to restart the analysis.")
        st.session_state.file_uploader_reset = False
        st.session_state.file_reset_reason = None
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls"],
        help="Upload your survey data or file containing text entries for clustering",
        key=upload_key,
        label_visibility="collapsed"
    )

    # Check if a new file was uploaded (different from previous)
    current_file_key = None
    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
    
    # Detect file change and reset analysis if needed using unified reset system
    file_changed = False
    if current_file_key != st.session_state.get('previous_file_key') and current_file_key is not None:
        # New file detected - warn user and reset everything
        if (st.session_state.get('tab_preprocessing_complete') or 
            st.session_state.get('clustering_results') or 
            st.session_state.get('finetuning_initialized')):
            st.warning("🔄 New file uploaded! This will reset all your previous work.")
        
        # Use unified reset system for file change
        reset_summary = reset_from_file_change(show_message=True)
        
        file_changed = True
        # clear previous alerts when a new file is uploaded
        st.session_state["data_loading_alerts"] = []

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
                success, df, message = st.session_state.backend.load_data(
                    temp_file_path, st.session_state.session_id
                )

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            if not success:
                st.error(f"{message}")
                return

            # Ensure entryID is present and clean (1..N, int)
            df = df.copy()
            if "entryID" not in df.columns:
                df.insert(0, "entryID", range(1, len(df) + 1))
            else:
                # Recreate to guarantee consistent integer IDs starting at 1
                df["entryID"] = range(1, len(df) + 1)

            # Store dataframe and reset progress flags
            st.session_state.df = df
            st.session_state.previous_file_key = current_file_key
            st.session_state.tab_data_loading_complete = False
            st.session_state.uploaded_filename = uploaded_file.name

            # If you track overall progress, also reset here
            if "permanent_progress" in st.session_state:
                st.session_state.permanent_progress["data_loading"] = False
                st.session_state.permanent_progress["preprocessing"] = False
                st.session_state.permanent_progress["clustering"] = False

            # Build persistent alerts
            alerts = []
            if "truncated to 300" in (message or "").lower():
                # Make a yellow warning + green success
                # Extract the truncation phrase if present
                try:
                    # message looks like: "File loaded successfully (truncated to 300 rows from X)"
                    trunc = message.split("File loaded successfully", 1)[1].strip()
                    trunc = trunc.strip("()")
                except Exception:
                    trunc = "File truncated to 300 rows"
                alerts.append(("warning", f"{trunc.capitalize()}." if not trunc.endswith(".") else trunc))
                alerts.append(("success", "File uploaded successfully."))
            else:
                alerts.append(("success", message or "File uploaded successfully."))

            st.session_state["data_loading_alerts"] = alerts
            # Rerun so sidebar status updates; alerts will re-render on next run
            st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
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
                3. Contains the text entries you want to analyze
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
    
    st.markdown("---")

    # File Overview Section
    st.subheader("File Overview")
    
    # Show file metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        file_name = st.session_state.get("uploaded_filename", "Loaded Data")
        st.metric("File Name", file_name)
    with metric_col2:
        st.metric("Total Rows", len(df))
    with metric_col3:
        st.metric("Total Columns", len(df.columns))
    with metric_col4:
        # Count text-like columns excluding entryID
        text_cols = sum(1 for col in df.columns
                        if col != 'entryID' and (is_object_dtype(df[col]) or is_string_dtype(df[col]))
                    )
        st.metric("Total Text Columns", text_cols)
    
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
            empty_rows = int(df[col].isna().sum())
            if is_object_dtype(df[col]) or is_string_dtype(df[col]):
                empty_rows += int((df[col] == '').sum())
            non_empty_rows = int(len(df) - empty_rows)
            
            # calculate column type (special handling for entryID column)
            if col == 'entryID':
                col_type = '(Auto-generated)'
            else:
                is_text_like = is_object_dtype(df[col]) or is_string_dtype(df[col])
                col_type = 'Text' if is_text_like else 'Non-Text'

            stats_data[col] = {
                'Total Rows': total_rows,
                'Empty Rows': empty_rows,
                'Non-Empty Rows': non_empty_rows,
                'Column Type': col_type
            }
        
        # display the column stats (entryID first, then the rest)
        ordered_cols = ['entryID'] + [c for c in stats_data.keys() if c != 'entryID']
        stats_df = pd.DataFrame(stats_data)[ordered_cols]

        st.dataframe(stats_df, use_container_width=True)

    # Column Selection Section
    st.markdown("---")
    st.subheader("Column Selection:")

    # subject id column selection section
    st.markdown("Step 1: Choose a column for subject identification (Subject IDs)")
    auto_option = "use entryID (row numbers) as subject IDs"
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

    # Display "entryID" as the auto_option label in the UI
    if current_id_selection == "entryID":
        current_id_selection = auto_option
    if current_id_selection not in id_options:
        current_id_selection = prompt_option

    # get the index of the current subject id selection
    id_column_index = get_safe_index(id_options, current_id_selection, 0)

    # selectbox for the subject id column
    selected_id = st.selectbox(
        label="SubjectID Column:",
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

    # fallback: save entryID (row numbers) as subjectID, if the selected subject id is the prompt option or not set
    if st.session_state.subjectID == prompt_option or not st.session_state.subjectID:
        st.session_state.subjectID = "entryID"

    # Show sample subject IDs (always resolved)
    sid = st.session_state.subjectID
    if sid:
        try:
            sample_ids = df[sid].head(10).tolist()
            if sample_ids:
                formatted = ", ".join([f'"{str(x)}"' for x in sample_ids])
                st.caption(f"**Sample subject IDs: {{{formatted}}}**")
        except (KeyError, AttributeError):
            st.caption("Selected ID column not accessible")

    # entry column selection section
    st.markdown("Step 2: Choose a column for text entry clustering")
    prompt_option_text = "-- Select an entry column --"
    
    # Get entry column suggestions first to filter options
    text_columns = []
    if backend_available:
        try:
            text_columns = st.session_state.backend.get_text_column_suggestions(df, st.session_state.session_id)
            # minimal sanity filter: must exist in df and not be entryID
            text_columns = [c for c in text_columns if c in df.columns and c != 'entryID']
        except AttributeError:
            # Simple fallback - check for text-like columns, excluding entryID
            text_columns = [col for col in df.columns
                            if col != 'entryID' and (is_object_dtype(df[col]) or is_string_dtype(df[col]))]
    
    # Create filtered options - only show text columns plus prompt
    text_options = [prompt_option_text] + text_columns
    
    # Preserve existing entry column selection
    current_text_selection = st.session_state.get('entry_column', prompt_option_text)
    
    # Only reset if no previous selection or selection is invalid
    if current_text_selection is None:
        current_text_selection = prompt_option_text
    
    # Ensure the current selection is valid for current data
    if current_text_selection not in text_options:
        current_text_selection = prompt_option_text
    
    text_column_index = get_safe_index(text_options, current_text_selection, 0)
    
    selected_text_column = st.selectbox(
        label="Entry Column:",
        label_visibility="collapsed",
        options=text_options,
        index=text_column_index,
        help="Select the column with text entries you want to cluster",
        key="text_selector"
    )
    
    # Enhanced change detection for entry column using unified reset system
    if selected_text_column != st.session_state.get('entry_column'):
        handle_column_selection_change(selected_text_column, st.session_state.get('entry_column'), "entry")
        st.session_state.entry_column = selected_text_column
    
    # Store user selections for output structure
    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {}

    # Only update user selections if valid columns are selected
    if selected_text_column != prompt_option_text and st.session_state.subjectID:
        st.session_state.user_selections.update({
            'id_column_choice': st.session_state.subjectID,
            'entry_column_choice': selected_text_column,
            'original_columns': [st.session_state.subjectID, selected_text_column]
                if st.session_state.subjectID != 'entryID' else [selected_text_column]
        })

    # Show feedback about entry column detection
    if selected_text_column and selected_text_column != prompt_option_text:
        if text_columns:
            # telling how many entry columns were detected
            pass
        else:
            st.warning("No obvious entry columns detected. Please verify your selection.")
    
    # Show 10 sample text entries with improved formatting and safety
    if selected_text_column and selected_text_column != prompt_option_text:
        try:
            sample_texts = df[selected_text_column].dropna().head(10).tolist()
            if sample_texts:
                formatted_samples = ", ".join([f'"{str(text)[:100] + "..." if len(str(text)) > 100 else str(text)}"' for text in sample_texts])
                st.caption(f"**Sample text entries: {{{formatted_samples}}}**")
        except (KeyError, AttributeError):
            st.caption("Selected entry column not accessible")

    # Validation and Quality Analysis
    if (selected_text_column and selected_text_column != prompt_option_text and 
        selected_text_column in df.columns):
        
        with st.spinner("Analyzing data quality..."):
            validation_result = st.session_state.backend.validate_columns(
                df, 
                selected_text_column, 
                st.session_state.subjectID,
                st.session_state.session_id
            )
        
        # Use original backend validation keys
        if validation_result["text_column_valid"]:
            st.success(f"{validation_result['text_column_message']}")
            stats = validation_result["text_quality"]

            # force numeric types (backend may return strings)
            _total = int(stats.get('total_texts', 0))
            _empty = int(stats.get('empty_texts', 0))
            _avg_len = float(stats.get('avg_length', 0))
            _avg_words = float(stats.get('avg_words', 0))
            _unique = int(stats.get('unique_texts', 0))
            
            st.markdown("**Text Entry Quality Metrics**")
            quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
            
            with quality_col1:
                text_entries = len(df[selected_text_column].dropna())
                st.metric(
                    "Total Text Entries", 
                    text_entries
                )
            with quality_col2:
                st.metric(
                    "Unique Text Entries",
                    _unique
                )
            with quality_col3:
                st.metric(
                    "Avg Text Length", 
                    f"{_avg_len:.0f} chars"
                )
            with quality_col4:
                st.metric(
                    "Avg Text Words", 
                    f"{_avg_words:.1f}"
                )
            
            # Sample text entries in a nice format
            with st.expander("Sample Text Entries Analysis", expanded=False):
                st.markdown("**Representative samples from your text entries:**")
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
            st.subheader("Ready to Proceed with:")
            
            # Summary display (showing only the subject id column and entry column basic info)
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("SubjectID Column:")
                sid = st.session_state.subjectID
                if sid:
                    if sid == 'entryID':
                        st.write("• column name: *entryID*")
                        st.write("• column type: (auto-generated) row numbers")
                    else:
                        st.write(f"• column name: *{sid}*")
                        col_type = 'Numeric' if pd.api.types.is_numeric_dtype(df[sid]) else 'Non-Numeric'
                        st.write(f"• column type: {col_type}")
            
            with summary_col2:
                st.markdown("Entry Column:")
                st.write(f"• column name: *{selected_text_column}*")
                col_type = 'Text' if pd.api.types.is_object_dtype(df[selected_text_column]) or pd.api.types.is_string_dtype(df[selected_text_column]) else 'Non-Text'
                st.write(f"• column type: {col_type}")
            
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
                        "entry_column": selected_text_column,
                        "id_column": st.session_state.subjectID,
                        "text_quality": stats
                    })
                
                # Show completion message
                st.success("Data Loading Complete!")
                st.info("Proceed to the **Preprocessing** tab to clean and prepare your text entries.")
                
                # Refresh sidebar immediately to show green button
                st.rerun()
                
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
                
                - Text entry length: Text entries with 20+ words work best  
                - Data quality: Remove or fix obviously corrupted text entries  
                - Language: Ensure all text entries are in the same language  
                - Relevance: All text entries should be about similar topics  
                - Volume: 50+ text entries recommended for meaningful clusters  
                
                **What happens next:**
                1. **Preprocessing:** Clean and prepare your text entries  
                2. **Clustering:** Run advanced algorithms to find patterns
                3. **Results:** Explore and export your findings
                """)
        
        else:
            # Use original backend error key
            st.error(f"{validation_result['text_column_message']}")
            # Provide helpful suggestions
            st.markdown("**Suggestions to fix this issue:**")
            st.markdown("""
            - Choose a different column that contains longer text entries  
            - Ensure the column has meaningful sentences, not just single words
            - Check that the column isn't mostly empty or contains mostly numbers/codes
            - Look for columns with survey responses, comments, or descriptions
            """)
            
            # If it was previously complete, mark incomplete and stop here
            if st.session_state.get('tab_data_loading_complete', False):
                st.session_state.tab_data_loading_complete = False