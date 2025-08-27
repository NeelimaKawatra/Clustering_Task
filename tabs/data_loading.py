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

def tab_a_data_loading(backend_available):
    """Tab A: Data Loading with automatic completion and cascading"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "data_loading"})
    
    # Introduction section
    st.markdown("""
    Welcome to Clustery! Start by uploading your data file containing text you want to cluster.
    
    **Supported formats:** CSV, Excel (.xlsx, .xls)  
    **Requirements:** At least 10 rows of text data
    """)
    
    # Check if data already exists in session state
    data_already_loaded = 'df' in st.session_state and st.session_state.df is not None
    
    # File upload section - always show, but with different messaging
    st.subheader("Upload Your File")
    upload_title = "Choose your data file"
    upload_help = "Upload your survey data or text file for clustering"
    upload_key = "data_file_uploader"
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            upload_title,
            type=["csv", "xlsx", "xls"],
            help=upload_help,
            key=upload_key,
            label_visibility="collapsed"
        )

    # Check if a new file was uploaded (different from previous)
    current_file_key = None
    if uploaded_file is not None:
        # Create a unique key for the file (name + size + type)
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
    
    # Detect file change and reset analysis if needed
    file_changed = False
    if 'previous_file_key' in st.session_state:
        if current_file_key != st.session_state.previous_file_key and current_file_key is not None:
            # New file detected - reset analysis
            reset_analysis()
            file_changed = True
    else:
        # First time loading - just store the key
        st.session_state.previous_file_key = current_file_key
        file_changed = True

    with col2:
        if uploaded_file is not None:
            # Show file info
            st.info(f"**File:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.info(f"**Type:** {uploaded_file.type}")
    
    # Process file upload if provided
    if uploaded_file is not None and file_changed == True:
        if not backend_available:
            st.error("Backend services not available. Please check backend installation.")
            return
        
        try:
            # Convert uploaded file to temporary path
            temp_file_path = get_file_from_upload(uploaded_file)
            
            # Use backend to load and validate file
            with st.spinner("Loading and validating file..."):
                success, df, message = st.session_state.backend.load_data(temp_file_path, st.session_state.session_id)
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            if not success:
                st.error(f"{message}")
                return
            
            # Store dataframe
            st.session_state.df = df
            
            # Update the file key after successful processing
            if file_changed:
                st.session_state.previous_file_key = current_file_key
            
            # Reset tab completion status since new data is loaded
            st.session_state.tab_a_complete = False
            
            # Show success message and let Streamlit re-render naturally
            st.success(f"{message}")
            st.balloons()
            
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
    
    # If no file uploaded and no data exists, return since nothing to do
    if not data_already_loaded and uploaded_file is None:
        return
    
    # If we reach here, we have data loaded (either from session state or new upload)
    # Display all the data overview, configuration, and validation sections
    df = st.session_state.df
    
    # File Overview Section
    st.subheader("File Overview")
    
    # Show file metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Rows", len(df))
    with metric_col2:
        st.metric("Columns", len(df.columns))
    with metric_col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024
        st.metric("Memory", f"{memory_usage:.1f} KB")
    with metric_col4:
        text_cols = len([col for col in df.columns if df[col].dtype == 'object'])
        st.metric("Text Columns", text_cols)
    
    # Data Overview Section
    with st.expander("Data Preview", expanded=True):
        st.markdown("**Your Loaded Data (first 300 rows):**")
        st.dataframe(
            df, 
            use_container_width=True,
            hide_index=True
        )

        # Column Statistics
        st.markdown("**Column Statistics:**")
        # Create a dictionary to store column statistics
        col_stats = {}
        for col in df.columns:
            total_rows = len(df)
            empty_rows = df[col].isna().sum() + (df[col] == '').sum()
            non_empty_rows = total_rows - empty_rows
            col_type = 'Text' if df[col].dtype == 'object' else 'Non-Text'
            col_stats[col] = [total_rows, empty_rows, non_empty_rows, col_type]
        
        # Create DataFrame with columns as columns and statistics as rows
        stats_df = pd.DataFrame(col_stats, index=['Total Rows', 'Empty Rows', 'Non-Empty Rows', 'Column Type'])
        # show the dataframe
        st.dataframe(stats_df, use_container_width=True)
    
    # Column Selection Section
    st.subheader("Column Selection")

    # Respondent ID Column Selection
    st.markdown("**Step 1: Choose an ID column (Optional)**")
    auto_option = "Auto-generate IDs"
    id_options = [auto_option] + list(df.columns)
    
    # Setting up the default values for the id column
    if "respondent_id_column" not in st.session_state:
        st.session_state.respondent_id_column = auto_option 
    
    # Get safe index for ID column selection
    id_column_index = get_safe_index(id_options, st.session_state.get('respondent_id_column'))
    
    selected_id = st.selectbox(
        label="Choose an ID column:",
        label_visibility="collapsed",
        options=id_options,
        index=id_column_index,
        help="Select a column to track individual responses",
        key="id_selector"  # Changed from "respondent_id_column"
)
    st.session_state.respondent_id_column = selected_id

# Removed the manual assignment line

    # Show sample IDs with improved formatting and safety checks (ACTIVE VERSION)
    if selected_id and selected_id != auto_option:
        try:
            sample_ids = df[selected_id].head(10).tolist()
            if sample_ids:
                formatted_ids = ", ".join([f'"{str(id)}"' for id in sample_ids])
                st.caption(f"**Sample IDs: {{{formatted_ids}}}**")
        except (KeyError, AttributeError):
            st.caption("Selected ID column not accessible")

    # COMMENTED VERSION: Original remote approach (unsafe - can crash on "Auto-generate IDs")
    """
    # Original version from remote (COMMENTED OUT - would crash on auto-generate)
    # show 10 sample ids with st.caption
    if st.session_state.respondent_id_column is not None:
        sample_ids = df[st.session_state.respondent_id_column].head(10).tolist()
        if sample_ids:
            formatted_ids = ", ".join([f'"{str(id)}"' for id in sample_ids])
            st.caption(f"**Sample IDs: {{{formatted_ids}}}**")
    """
    
    # Show ID column preview and validation
    
    if selected_id and selected_id != auto_option:
        try:
            id_column_data = df[selected_id]
            
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(id_column_data):
                st.success(f"✓ Numeric ID column selected: {selected_id}")
            else:
                st.warning(f"⚠ Non-numeric column selected. ID columns work best with numbers.")
                
        except (KeyError, TypeError):
            st.error(f"Column '{selected_id}' not found in data")
            return
        else:
            st.info("Will create sequential IDs: ID_001, ID_002, etc.")
            
    
    # Always generate clean IDs regardless of user choice
    if 'clean_ids' not in st.session_state or st.session_state.clean_ids is None:
        # Generate clean IDs for all rows
        if selected_id != auto_option and selected_id in df.columns:
            # Use existing column but clean it up
            raw_ids = df[selected_id].fillna("").astype(str)
            # Clean up the IDs - remove spaces, special chars, ensure uniqueness
            clean_ids = []
            seen_ids = set()
            for i, raw_id in enumerate(raw_ids):
                if raw_id.strip() and raw_id not in seen_ids:
                    clean_id = re.sub(r'[^\w\-_]', '', raw_id.strip())[:20]  # Clean and limit length
                    if clean_id:
                        clean_ids.append(clean_id)
                        seen_ids.add(raw_id)
                    else:
                        clean_ids.append(f"ID_{i+1:03d}")
                else:
                    clean_ids.append(f"ID_{i+1:03d}")
        else:
            # Auto-generate sequential IDs
            clean_ids = [f"ID_{i+1:03d}" for i in range(len(df))]
        
        st.session_state.clean_ids = clean_ids

    # COMMENTED VERSION: Legacy code from remote (old approach)
    """ 
    # Legacy code from remote (COMMENTED OUT - old approach)
    # if selected_id is not None:
    #     if selected_id == auto_option:
    #         st.session_state.respondent_id_column = None
    #         st.info("💡 Will create sequential IDs: ID_001, ID_002, etc.")
    #     else:
    #         st.session_state.respondent_id_column = selected_id
    #         # Show preview of selected ID column
    #         sample_ids = df[selected_id].head(5).tolist()
    #         st.code(f"Sample IDs: {sample_ids}")
    #     #id_index should be updated to reflect the index of the selected column
    #     st.session_state.id_index = id_options.index(selected_id)
    """    

    # Text Column Section
    st.markdown("**Step 2: Choose a Text Column for Clustering**")
    text_options = list(df.columns)
    
    # Setting up the default values for the text column (ACTIVE VERSION)
    if "text_column" not in st.session_state:
        st.session_state.text_column = None
    
    # Get safe index for text column selection
    text_column_index = get_safe_index(text_options, st.session_state.get('text_column'))
    
    # Get text column suggestions with error handling (ACTIVE VERSION)
    try:
        text_columns = st.session_state.backend.get_text_column_suggestions(df, st.session_state.session_id)
        if text_columns:
            st.success(f"✓ Detected {len(text_columns)} text columns suitable for clustering")
        else:
            st.warning("⚠ No obvious text columns detected. Please verify your selection.")
    except AttributeError:
        # Simple fallback - check if column is text type
        if selected_text_column and selected_text_column in df.columns:
            if pd.api.types.is_object_dtype(df[selected_text_column]):
                st.success("✓ Text column type detected")
            else:
                st.warning("⚠ Selected column may not contain text data")

    # COMMENTED VERSION: Original remote approach for text column setup
    """
    # Original remote version approach (COMMENTED OUT - less robust)
    else:
        # Key exists, but value might be None or a column name
        if st.session_state.text_column is None:
            st.session_state.text_index = None
        elif st.session_state.text_column in df.columns:
            st.session_state.text_index = text_options.index(st.session_state.text_column)
        else:
            st.session_state.text_index = None  # Fallback if column doesn't exist in dataframe
    
    # Get text column suggestions from backend
    text_columns = st.session_state.backend.get_text_column_suggestions(df, st.session_state.session_id)

    # st.success message to show the number of potential text columns
    if text_columns:
        st.success(f"💡 Detected {len(text_columns)} text columns usable for clustering")
        #for col in text_columns[:3]:  # Show top 3
        #    if not df[col].dropna().empty:
        #        sample_text = str(df[col].dropna().iloc[0])[:100] + "..." if len(str(df[col].dropna().iloc[0])) > 100 else str(df[col].dropna().iloc[0])
        #        st.caption(f"**{col}:** {sample_text}")
    """
    
    selected_text_column = st.selectbox(
        label="Choose a text column:",
        label_visibility="collapsed",
        options=text_options,
        index=text_column_index,
        help="Choose the column with text you want to cluster",
        key="text_selector"  # Changed from "text_column"
)
    st.session_state.text_column = selected_text_column
    
    # Show 10 sample texts with improved formatting and safety (ACTIVE VERSION)
    if selected_text_column is not None:
        try:
            sample_texts = df[selected_text_column].dropna().head(10).tolist()
            if sample_texts:
                formatted_samples = ", ".join([f'"{str(text)[:100] + "..." if len(str(text)) > 100 else str(text)}"' for text in sample_texts])
                st.caption(f"**Sample texts: {{{formatted_samples}}}**")
        except (KeyError, AttributeError):
            st.caption("Selected text column not accessible")

    # COMMENTED VERSION: Original remote approach (unsafe - no error handling)
    """
    # Original version from remote (COMMENTED OUT - no error handling)
    # show 10 sample texts with st.caption  
    if st.session_state.text_column is not None:
        sample_texts = df[st.session_state.text_column].dropna().head(10).tolist()
        if sample_texts:
            formatted_samples = ", ".join([f'"{str(text)}"' for text in sample_texts])
            st.caption(f"**Sample texts : {{{formatted_samples}}}**")
    """

    # Validation and Quality Analysis
    if selected_text_column and selected_text_column in df.columns:
        st.subheader("Data Quality Analysis")
        
        with st.spinner("Analyzing data quality..."):
            validation_result = st.session_state.backend.validate_columns(
                df, 
                selected_text_column, 
        selected_id, 
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
            
            # ID column analysis
            id_analysis = validation_result["id_column_analysis"]
            
            st.markdown("**ID Column Analysis**")
            if id_analysis["status"] == "perfect":
                st.success(f"{id_analysis['message']}")
            elif id_analysis["status"] in ["duplicates", "missing"]:
                st.warning(f"{id_analysis['message']}")
                
                # Show detailed ID analysis
                id_detail_col1, id_detail_col2, id_detail_col3 = st.columns(3)
                with id_detail_col1:
                    st.metric("Total IDs", id_analysis['total'])
                with id_detail_col2:
                    st.metric("Unique IDs", id_analysis['unique'])
                with id_detail_col3:
                    if 'duplicates' in id_analysis:
                        st.metric("Duplicates", id_analysis['duplicates'])
                    elif 'missing' in id_analysis:
                        st.metric("Missing", id_analysis['missing'])
            else:
                st.info(f"{id_analysis['message']}")
            
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
            
            with summary_col2:
                st.markdown("**Text Configuration:**")
                st.write(f"• **Column:** {st.session_state.text_column}")
                st.write(f"• **Avg length:** {stats['avg_length']:.0f} characters")
                st.write(f"• **Avg words:** {stats['avg_words']:.1f} words")
            
            with summary_col3:
                st.markdown("**ID Configuration:**")
                if st.session_state.respondent_id_column and st.session_state.respondent_id_column != "Auto-generate IDs":
                    st.write(f"• **Column:** {st.session_state.respondent_id_column}")
                    st.write(f"• **Status:** {id_analysis['status'].title()}")
                    st.write(f"• **Unique:** {id_analysis['unique']:,} IDs")
                else:
                    st.write("• **Type:** Auto-generated")
                    st.write("• **Format:** ID_001, ID_002...")
                    st.write("• **Count:** Sequential numbering")
            
            # Auto-completion with celebration message
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Check if step should auto-complete
            if not st.session_state.get('tab_a_complete', False):
                # Auto-complete when validation passes
                st.session_state.tab_a_complete = True
                
                # Track completion
                if backend_available:
                    st.session_state.backend.track_activity(st.session_state.session_id, "data_upload", {
                        "filename": "loaded_data",
                        "rows": len(df),
                        "columns": len(df.columns),
                        "text_column":st.write(f"• **Column:** {selected_text_column}"),
                        "id_column": st.write(f"• **Column:** {selected_id}"),
                        "text_quality": stats
                    })
                
                st.balloons()
                # Show completion message
                st.success("Data Loading Complete!")
                st.info("Your data is ready! Head over to the **Preprocessing** tab to clean and prepare your text data for clustering.")
                
            else:
                # Already completed - just show status
                st.success("Data Loading Complete")
                st.info("Your data configuration is saved. You can proceed to **Preprocessing** or modify settings above to trigger automatic reset.")
            
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
            if st.session_state.get('tab_a_complete', False):
                st.session_state.tab_a_complete = False
                st.rerun()