import streamlit as st
import os
from utils.helpers import get_file_from_upload

def tab_a_data_loading(backend_available):
    """Tab A: Data Loading using backend services"""
    
    # Track tab visit
    if backend_available:
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
        if not backend_available:
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