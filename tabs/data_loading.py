import streamlit as st
import os
from utils.helpers import get_file_from_upload

def tab_a_data_loading(backend_available):
    """Tab A: Data Loading using backend services with sidebar navigation"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "data_loading"})
    
    # Introduction section
    st.markdown("""
    Welcome to Clustery! Start by uploading your data file containing text you want to cluster.
    
    **Supported formats:** CSV, Excel (.xlsx, .xls)  
    **Requirements:** At least 10 rows of text data
    """)
    
    # File upload section
    st.subheader("📤 Upload Your Data")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=["csv", "xlsx", "xls"],
            help="Upload your survey data or text file for clustering",
            key="data_file_uploader"
        )
    
    with col2:
        if uploaded_file is not None:
            # Show file info
            st.info(f"**File:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.info(f"**Type:** {uploaded_file.type}")
    
    if uploaded_file is not None:
        if not backend_available:
            st.error("❌ Backend services not available. Please check backend installation.")
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
                st.error(f"❌ {message}")
                return
            
            st.success(f"✅ {message}")
            
            # Store dataframe
            st.session_state.df = df
            
            # File Overview Section
            st.subheader("📊 File Overview")
            
            # Show file metrics in columns
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("📄 Total Rows", len(df))
            with metric_col2:
                st.metric("📋 Columns", len(df.columns))
            with metric_col3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024
                st.metric("💾 Memory", f"{memory_usage:.1f} KB")
            with metric_col4:
                text_cols = len([col for col in df.columns if df[col].dtype == 'object'])
                st.metric("📝 Text Columns", text_cols)
            
            # Data preview with better styling
            with st.expander("👀 Data Preview", expanded=True):
                st.markdown("**First 5 rows of your data:**")
                st.dataframe(
                    df.head(), 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show column info
                st.markdown("**Column Information:**")
                col_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null = df[col].count()
                    null_count = len(df) - non_null
                    col_info.append({
                        'Column': col,
                        'Type': dtype,
                        'Non-Null': non_null,
                        'Null Values': null_count
                    })
                
                import pandas as pd
                st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)
            
            # Column Configuration Section
            st.subheader("⚙️ Column Configuration")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.markdown("**🆔 Respondent ID Column (Optional)**")
                id_options = ["Auto-generate IDs"] + list(df.columns)
                selected_id = st.selectbox(
                    "Choose ID column:",
                    id_options,
                    help="Select a column to track individual responses",
                    key="id_column_selector"
                )
                
                if selected_id == "Auto-generate IDs":
                    st.session_state.respondent_id_column = None
                    st.info("💡 Will create sequential IDs: ID_001, ID_002, etc.")
                else:
                    st.session_state.respondent_id_column = selected_id
                    # Show preview of selected ID column
                    sample_ids = df[selected_id].head(5).tolist()
                    st.code(f"Sample IDs: {sample_ids}")
            
            with config_col2:
                st.markdown("**📝 Text Column for Clustering**")
                
                # Get text column suggestions from backend
                text_columns = st.session_state.backend.get_text_column_suggestions(df, st.session_state.session_id)
                
                if text_columns:
                    st.success(f"💡 Detected {len(text_columns)} potential text columns")
                    for col in text_columns[:3]:  # Show top 3
                        if not df[col].dropna().empty:
                            sample_text = str(df[col].dropna().iloc[0])[:100] + "..." if len(str(df[col].dropna().iloc[0])) > 100 else str(df[col].dropna().iloc[0])
                            st.caption(f"**{col}:** {sample_text}")
                
                text_column = st.selectbox(
                    "Select text column:",
                    df.columns,
                    help="Choose the column with text you want to cluster",
                    key="text_column_selector"
                )
                
                if text_column:
                    st.session_state.text_column = text_column
                    # Show preview of selected text column
                    sample_texts = df[text_column].dropna().head(3).tolist()
                    if sample_texts:
                        with st.container():
                            st.markdown("**Sample texts:**")
                            for i, text in enumerate(sample_texts, 1):
                                st.text_area(
                                    f"Sample {i}:",
                                    str(text)[:200] + "..." if len(str(text)) > 200 else str(text),
                                    height=60,
                                    disabled=True,
                                    key=f"sample_text_{i}"
                                )
            
            # Validation and Quality Analysis
            if st.session_state.get('text_column'):
                st.subheader("🔍 Data Quality Analysis")
                
                with st.spinner("Analyzing data quality..."):
                    validation_result = st.session_state.backend.validate_columns(
                        df, 
                        st.session_state.text_column, 
                        st.session_state.respondent_id_column, 
                        st.session_state.session_id
                    )
                
                if validation_result["text_column_valid"]:
                    st.success(f"✅ {validation_result['text_column_message']}")
                    
                    # Show text quality metrics in an attractive layout
                    stats = validation_result["text_quality"]
                    
                    st.markdown("**📈 Text Quality Metrics**")
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
                            delta=f"Range: {stats['min_length']}-{stats['max_length']}"
                        )
                    with quality_col3:
                        st.metric(
                            "Avg Words", 
                            f"{stats['avg_words']:.1f}",
                            delta=f"Range: {stats['min_words']}-{stats['max_words']}"
                        )
                    with quality_col4:
                        uniqueness = (stats['unique_texts'] / stats['total_texts']) * 100
                        st.metric(
                            "Unique Texts", 
                            stats['unique_texts'],
                            delta=f"{uniqueness:.1f}% unique"
                        )
                    
                    # Sample texts in a nice format
                    with st.expander("📖 Sample Texts Analysis", expanded=False):
                        st.markdown("**Representative samples from your text data:**")
                        sample_texts = df[st.session_state.text_column].dropna().head(5)
                        
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
                                    st.caption(f"📝 {word_count} words")
                                    st.caption(f"🔤 {char_count} chars")
                                
                                st.markdown("---")
                    
                    # ID column analysis
                    id_analysis = validation_result["id_column_analysis"]
                    
                    st.markdown("**🆔 ID Column Analysis**")
                    if id_analysis["status"] == "perfect":
                        st.success(f"✅ {id_analysis['message']}")
                    elif id_analysis["status"] in ["duplicates", "missing"]:
                        st.warning(f"⚠️ {id_analysis['message']}")
                        
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
                        st.info(f"ℹ️ {id_analysis['message']}")
                    
                    # Ready to proceed section
                    st.markdown("---")
                    st.subheader("🚀 Ready to Proceed")
                    
                    # Summary display with better styling
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.markdown("**📊 Data Summary:**")
                        st.write(f"• **Total rows:** {len(df):,}")
                        st.write(f"• **Valid texts:** {stats['total_texts'] - stats['empty_texts']:,}")
                        st.write(f"• **Data quality:** {'Excellent' if stats['avg_length'] > 50 else 'Good' if stats['avg_length'] > 20 else 'Fair'}")
                    
                    with summary_col2:
                        st.markdown("**📝 Text Configuration:**")
                        st.write(f"• **Column:** {st.session_state.text_column}")
                        st.write(f"• **Avg length:** {stats['avg_length']:.0f} characters")
                        st.write(f"• **Avg words:** {stats['avg_words']:.1f} words")
                    
                    with summary_col3:
                        st.markdown("**🆔 ID Configuration:**")
                        if st.session_state.respondent_id_column:
                            st.write(f"• **Column:** {st.session_state.respondent_id_column}")
                            st.write(f"• **Status:** {id_analysis['status'].title()}")
                            st.write(f"• **Unique:** {id_analysis['unique']:,} IDs")
                        else:
                            st.write("• **Type:** Auto-generated")
                            st.write("• **Format:** ID_001, ID_002...")
                            st.write("• **Count:** Sequential numbering")
                    
                    # Large, prominent proceed button
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    proceed_col1, proceed_col2, proceed_col3 = st.columns([1, 2, 1])
                    with proceed_col2:
                        if st.button(
                            "🚀 Proceed to Preprocessing", 
                            type="primary", 
                            use_container_width=True,
                            help="Move to the next step to clean and prepare your text data"
                        ):
                            # Mark this tab as complete
                            st.session_state.tab_a_complete = True
                            
                            # Track completion
                            st.session_state.backend.track_activity(st.session_state.session_id, "data_upload", {
                                "filename": uploaded_file.name,
                                "rows": len(df),
                                "columns": len(df.columns),
                                "text_column": st.session_state.text_column,
                                "id_column": st.session_state.respondent_id_column,
                                "text_quality": stats
                            })
                            
                            # Navigate to preprocessing
                            st.session_state.current_page = "preprocessing"
                            
                            # Show success message
                            st.success("✅ Data loading complete! Moving to preprocessing...")
                            st.balloons()
                            
                            # Rerun to update sidebar and navigate
                            st.rerun()
                    
                    # Additional tips
                    st.markdown("---")
                    with st.expander("💡 Tips for Better Results", expanded=False):
                        st.markdown("""
                        **For optimal clustering results:**
                        
                        - ✅ **Text length:** Texts with 20+ words work best
                        - ✅ **Data quality:** Remove or fix obviously corrupted entries
                        - ✅ **Language:** Ensure all texts are in the same language
                        - ✅ **Relevance:** All texts should be about similar topics
                        - ✅ **Volume:** 50+ texts recommended for meaningful clusters
                        
                        **What happens next:**
                        1. **Preprocessing:** Clean and prepare your texts
                        2. **Clustering:** Run advanced algorithms to find patterns
                        3. **Results:** Explore and export your findings
                        """)
                
                else:
                    st.error(f"❌ {validation_result['text_column_message']}")
                    
                    # Provide helpful suggestions
                    st.markdown("**💡 Suggestions to fix this issue:**")
                    st.markdown("""
                    - Choose a different column that contains longer text
                    - Ensure the column has meaningful sentences, not just single words
                    - Check that the column isn't mostly empty or contains mostly numbers/codes
                    - Look for columns with survey responses, comments, or descriptions
                    """)
                    
                    st.session_state.text_column = None
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            
            # Provide helpful error guidance
            with st.expander("🔧 Troubleshooting Help"):
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
    
    else:
        # Show helpful getting started information
        st.markdown("---")
        st.subheader("🎯 Getting Started")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("""
            **What type of data works best?**
            
            📋 **Survey responses** - Open-ended questions
            💬 **Customer feedback** - Reviews, comments
            📝 **Social media posts** - Tweets, posts
            📰 **Articles** - News, blog posts
            📞 **Support tickets** - Customer inquiries
            """)
        
        with info_col2:
            st.markdown("""
            **File requirements:**
            
            📁 **Format:** CSV or Excel (.xlsx, .xls)
            📊 **Size:** At least 10 rows of data
            📝 **Text:** One column with meaningful text
            🆔 **IDs:** Optional column for tracking
            💾 **Limit:** Up to 300 rows (for performance)
            """)
        
        # Sample data suggestion
        st.markdown("---")
        st.info("""
        **💡 Don't have data ready?** You can test Clustery with sample data:
        - Create a CSV with customer feedback, survey responses, or any text collection
        - Include at least 20-30 text entries for meaningful results
        - Make sure each text entry has multiple words (not just single words or codes)
        """)
    
    # Show current progress in the main area too
    if st.session_state.get('tab_a_complete'):
        st.success("✅ **Data Loading Complete!** Your data is ready for preprocessing.")
        
        if st.button("➡️ Continue to Preprocessing", key="continue_to_preprocessing"):
            st.session_state.current_page = "preprocessing"
            st.rerun()