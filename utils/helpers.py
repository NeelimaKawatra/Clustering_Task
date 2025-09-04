# utils/helpers.py - Complete helper functions
import os
import tempfile
import streamlit as st

def get_file_from_upload(uploaded_file):
    """Convert uploaded file to temporary file path for backend processing"""
    if uploaded_file is not None:
        # Create temporary file with proper extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)
        
        try:
            # Write uploaded file content to temporary file
            with os.fdopen(temp_fd, 'wb') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
            
            return temp_path
        except Exception as e:
            # Clean up on error
            try:
                os.close(temp_fd)
            except:
                pass
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e
    
    return None

def validate_file_type(filename):
    """Validate if file type is supported"""
    if not filename:
        return False, "No filename provided"
    
    supported_extensions = ['.csv', '.xlsx', '.xls']
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension not in supported_extensions:
        return False, f"Unsupported file type: {file_extension}. Supported types: {', '.join(supported_extensions)}"
    
    return True, "File type supported"

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def safe_column_name(column_name):
    """Create a safe column name for exports"""
    if not column_name:
        return "unnamed_column"
    
    # Replace spaces and special characters
    import re
    safe_name = re.sub(r'[^\w\s-]', '', str(column_name))
    safe_name = re.sub(r'[\s]+', '_', safe_name)
    safe_name = safe_name.strip('_').lower()
    
    return safe_name if safe_name else "unnamed_column"

def get_column_data_type(series):
    """Determine the data type of a pandas series"""
    import pandas as pd
    
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif pd.api.types.is_bool_dtype(series):
        return "boolean"
    else:
        return "text"

def preview_text(text, max_length=100):
    """Create a preview of text with ellipsis if too long"""
    if not text:
        return ""
    
    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str
    else:
        return text_str[:max_length] + "..."

def count_words(text):
    """Count words in text"""
    if not text:
        return 0
    
    return len(str(text).split())

def clean_text_for_display(text):
    """Clean text for display purposes"""
    if not text:
        return ""
    
    import re
    text_str = str(text)
    
    # Remove extra whitespace
    text_str = re.sub(r'\s+', ' ', text_str)
    
    # Remove control characters
    text_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text_str)
    
    return text_str.strip()

def create_download_filename(base_name, file_type="csv", timestamp=True):
    """Create a standardized download filename"""
    import datetime
    
    # Clean base name
    safe_base = safe_column_name(base_name)
    
    # Add timestamp if requested
    if timestamp:
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_base}_{timestamp_str}.{file_type}"
    else:
        filename = f"{safe_base}.{file_type}"
    
    return filename

def estimate_processing_time(text_count, complexity="medium"):
    """Estimate processing time based on text count and complexity"""
    
    # Base times in seconds per text
    base_times = {
        "low": 0.001,      # Simple operations
        "medium": 0.01,    # Standard preprocessing + clustering
        "high": 0.1        # Complex clustering with multiple models
    }
    
    base_time = base_times.get(complexity, base_times["medium"])
    estimated_seconds = text_count * base_time
    
    # Add overhead
    overhead = 2.0  # 2 seconds base overhead
    total_time = estimated_seconds + overhead
    
    # Format time
    if total_time < 60:
        return f"~{int(total_time)} seconds"
    elif total_time < 3600:
        minutes = int(total_time / 60)
        return f"~{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        hours = int(total_time / 3600)
        return f"~{hours} hour{'s' if hours != 1 else ''}"

def check_session_state_health():
    """Check for potential session state issues"""
    issues = []
    
    # Check for required keys
    required_keys = ['df', 'text_column', 'respondent_id_column']
    for key in required_keys:
        if key not in st.session_state:
            issues.append(f"Missing session state key: {key}")
    
    # Check data consistency
    if st.session_state.get('df') is not None:
        df = st.session_state.df
        text_col = st.session_state.get('text_column')
        
        if text_col and text_col not in df.columns:
            issues.append(f"Selected text column '{text_col}' not found in data")
    
    # Check completion status consistency
    if st.session_state.get('tab_preprocessing_complete') and not st.session_state.get('tab_data_loading_complete'):
        issues.append("Tab preprocessing marked complete but Tab data_loading is not")
    
    if st.session_state.get('tab_clustering_complete') and not st.session_state.get('tab_preprocessing_complete'):
        issues.append("Tab clustering marked complete but Tab preprocessing is not")
    
    return issues

def log_session_state_debug():
    """Log current session state for debugging"""
    debug_info = {
        "df_exists": st.session_state.get('df') is not None,
        "df_shape": st.session_state.df.shape if st.session_state.get('df') is not None else None,
        "text_column": st.session_state.get('text_column'),
        "id_column": st.session_state.get('respondent_id_column'),
        "tab_completions": {
            "a": st.session_state.get('tab_data_loading_complete', False),
            "b": st.session_state.get('tab_preprocessing_complete', False),
            "c": st.session_state.get('tab_clustering_complete', False),
        },
        "processed_texts_count": len(st.session_state.get('processed_texts', [])),
        "clustering_success": bool(st.session_state.get('clustering_results', {}).get('success', False)),
        "current_page": st.session_state.get('current_page', 'unknown')
    }
    
    return debug_info