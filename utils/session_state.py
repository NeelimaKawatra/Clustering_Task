import streamlit as st
import time

def initialize_session_state(backend_available=True):
    """Initialize session state variables and backend"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'respondent_id_column' not in st.session_state:
        st.session_state.respondent_id_column = None
    if 'text_column' not in st.session_state:
        st.session_state.text_column = None
    if 'tab_a_complete' not in st.session_state:
        st.session_state.tab_a_complete = False
    if 'processed_texts' not in st.session_state:
        st.session_state.processed_texts = None
    if 'preprocessing_settings' not in st.session_state:
        st.session_state.preprocessing_settings = {}
    if 'tab_b_complete' not in st.session_state:
        st.session_state.tab_b_complete = False
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None
    
    # Initialize state fingerprints for change detection
    if 'state_fingerprints' not in st.session_state:
        st.session_state.state_fingerprints = {}
    
    # Initialize backend and session
    if 'backend' not in st.session_state and backend_available:
        from backend import ClusteryBackend
        st.session_state.backend = ClusteryBackend()
        st.session_state.session_id = f"user_{int(time.time())}"
        st.session_state.backend.start_session(st.session_state.session_id)

def cascade_from_data_loading():
    """Automatically reset downstream steps when data loading changes"""
    
    # Clear preprocessing and everything after it
    downstream_keys = [
        'processed_texts', 'original_texts', 'preprocessing_settings', 'tab_b_complete',
        'clustering_results', 'tab_c_complete'
    ]
    
    reset_items = []
    for key in downstream_keys:
        if key in st.session_state and st.session_state[key] is not None:
            if key == 'tab_b_complete' and st.session_state[key]:
                reset_items.append("preprocessing")
            elif key == 'clustering_results' and st.session_state[key]:
                reset_items.append("clustering")
            del st.session_state[key]
    
    if reset_items:
        reset_list = ", ".join(set(reset_items))  # Remove duplicates
        st.warning(f"Data configuration changed. Automatically reset: {reset_list}")
        
        # Log the cascade reset
        if hasattr(st.session_state, 'backend'):
            st.session_state.backend.track_activity(st.session_state.session_id, "cascade_reset", {
                "trigger": "data_loading_change",
                "steps_reset": reset_items
            })

def cascade_from_preprocessing():
    """Automatically reset downstream steps when preprocessing changes"""
    
    # Clear clustering and everything after it
    downstream_keys = ['clustering_results', 'tab_c_complete']
    
    reset_occurred = False
    for key in downstream_keys:
        if key in st.session_state and st.session_state[key] is not None:
            del st.session_state[key]
            reset_occurred = True
    
    if reset_occurred:
        st.warning("Preprocessing changed. Automatically reset clustering results.")
        
        # Log the cascade reset
        if hasattr(st.session_state, 'backend'):
            st.session_state.backend.track_activity(st.session_state.session_id, "cascade_reset", {
                "trigger": "preprocessing_change", 
                "steps_reset": ["clustering"]
            })

def check_automatic_completion():
    """Check conditions and auto-complete steps when ready"""
    
    # Auto-complete Data Loading
    if (not st.session_state.get('tab_a_complete', False) and
        st.session_state.get('df') is not None and
        st.session_state.get('text_column') is not None):
        
        # Additional validation check
        if hasattr(st.session_state, 'backend') and st.session_state.backend:
            try:
                validation = st.session_state.backend.validate_columns(
                    st.session_state.df, 
                    st.session_state.text_column,
                    st.session_state.respondent_id_column,
                    st.session_state.session_id
                )
                
                if validation.get('text_column_valid', False):
                    st.session_state.tab_a_complete = True
            except:
                pass  # Skip auto-completion if validation fails
    
    # Auto-complete Preprocessing
    if (not st.session_state.get('tab_b_complete', False) and
        st.session_state.get('tab_a_complete', False) and
        st.session_state.get('processed_texts') is not None and
        len(st.session_state.get('processed_texts', [])) >= 10):
        
        st.session_state.tab_b_complete = True

def detect_changes_and_cascade():
    """Detect changes and automatically cascade reset dependent steps"""
    
    # Generate current state fingerprint
    current_fingerprint = {
        'file_key': st.session_state.get('previous_file_key'),
        'text_column': st.session_state.get('text_column'),
        'id_column': st.session_state.get('respondent_id_column'),
        'df_shape': tuple(st.session_state.df.shape) if st.session_state.get('df') is not None else None
    }
    
    previous_fingerprint = st.session_state.state_fingerprints.get('last_known', {})
    
    # Check for data loading changes that affect downstream work
    significant_changes = []
    
    if (current_fingerprint.get('file_key') != previous_fingerprint.get('file_key') and 
        previous_fingerprint.get('file_key') is not None):
        significant_changes.append("file")
    
    if (current_fingerprint.get('text_column') != previous_fingerprint.get('text_column') and 
        previous_fingerprint.get('text_column') is not None):
        significant_changes.append("text_column")
    
    if (current_fingerprint.get('df_shape') != previous_fingerprint.get('df_shape') and 
        previous_fingerprint.get('df_shape') is not None):
        significant_changes.append("data_structure")
    
    # Only cascade if there are significant changes AND we have downstream work to reset
    if (significant_changes and 
        (st.session_state.get('tab_b_complete') or st.session_state.get('clustering_results'))):
        
        change_description = ", ".join(significant_changes)
        st.warning(f"Detected changes in {change_description}. Resetting dependent steps...")
        cascade_from_data_loading()
    
    # Store current fingerprint
    st.session_state.state_fingerprints['last_known'] = current_fingerprint

def reset_analysis():
    """Reset the analysis by clearing all relevant session state data"""
    # Clear session state
    keys_to_clear = [key for key in st.session_state.keys()
                     if key.startswith(('df', 'processed_', 'clustering_', 'tab_', 'text_', 'respondent_', 'previous_file_key', 'state_fingerprints', 'original_texts'))]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.current_page = "data_loading"