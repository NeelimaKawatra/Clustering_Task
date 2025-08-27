# utils/session_state.py - Unified and Complete Version
import streamlit as st
import time

def initialize_session_state(backend_available=True):
    """Initialize all session state variables with proper defaults"""
    
    # Core data storage
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'clean_ids' not in st.session_state:
        st.session_state.clean_ids = []
    
    # User selections
    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {
            'id_column_choice': None,
            'text_column_choice': None,
            'id_is_auto_generated': True,
            'original_columns': []
        }
    
    # Column selections (keep both naming conventions for compatibility)
    if 'respondent_id_column' not in st.session_state:
        st.session_state.respondent_id_column = "Auto-generate IDs"
    
    if 'text_column' not in st.session_state:
        st.session_state.text_column = None
    
    # Processing data
    if 'original_texts' not in st.session_state:
        st.session_state.original_texts = []
    
    if 'processed_texts' not in st.session_state:
        st.session_state.processed_texts = None
    
    if 'preprocessing_metadata' not in st.session_state:
        st.session_state.preprocessing_metadata = {}
    
    if 'preprocessing_settings' not in st.session_state:
        st.session_state.preprocessing_settings = {
            'method': 'none',
            'details': 'No preprocessing applied',
            'custom_settings': {}
        }
    
    # Row alignment tracking (simplified approach)
    if 'row_alignment' not in st.session_state:
        st.session_state.row_alignment = []  # Maps processed text index to original row index
    
    # Clustering results
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None
    
    # Tab completion status
    if 'tab_a_complete' not in st.session_state:
        st.session_state.tab_a_complete = False
    
    if 'tab_b_complete' not in st.session_state:
        st.session_state.tab_b_complete = False
    
    if 'tab_c_complete' not in st.session_state:
        st.session_state.tab_c_complete = False
    
    # Navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "data_loading"
    
    # Change detection
    if 'state_fingerprints' not in st.session_state:
        st.session_state.state_fingerprints = {}
    
    if 'previous_file_key' not in st.session_state:
        st.session_state.previous_file_key = None
    
    # Session management
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"user_{int(time.time())}"
    
    # Backend initialization
    if backend_available and 'backend' not in st.session_state:
        try:
            from backend import ClusteryBackend
            st.session_state.backend = ClusteryBackend()
            st.session_state.backend.start_session(
                st.session_state.session_id,
                {"user_agent": "streamlit", "timestamp": time.time()}
            )
        except Exception as e:
            st.error(f"Backend initialization failed: {e}")
            st.session_state.backend = None

def reset_analysis():
    """Reset all analysis data while preserving navigation state"""
    keys_to_reset = [
        'df', 'clean_ids', 'original_texts', 'processed_texts',
        'preprocessing_metadata', 'preprocessing_settings', 'row_alignment',
        'clustering_results', 'tab_a_complete', 'tab_b_complete',
        'tab_c_complete', 'user_selections', 'text_column',
        'respondent_id_column', 'previous_file_key', 'state_fingerprints'
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            if key == 'user_selections':
                st.session_state[key] = {
                    'id_column_choice': None,
                    'text_column_choice': None,
                    'id_is_auto_generated': True,
                    'original_columns': []
                }
            elif key == 'preprocessing_settings':
                st.session_state[key] = {
                    'method': 'none',
                    'details': 'No preprocessing applied',
                    'custom_settings': {}
                }
            elif key in ['clean_ids', 'original_texts', 'row_alignment']:
                st.session_state[key] = []
            else:
                st.session_state[key] = None
    
    # Reset to first page
    st.session_state.current_page = "data_loading"

def cascade_from_data_loading():
    """Reset downstream steps when data changes"""
    downstream_keys = [
        'processed_texts', 'original_texts', 'preprocessing_settings', 'preprocessing_metadata',
        'row_alignment', 'tab_b_complete', 'clustering_results', 'tab_c_complete'
    ]
    
    reset_items = []
    for key in downstream_keys:
        if key in st.session_state and st.session_state[key] is not None:
            if key == 'tab_b_complete' and st.session_state[key]:
                reset_items.append("preprocessing")
            elif key == 'clustering_results' and st.session_state[key]:
                reset_items.append("clustering")
            
            # Reset the key
            if key in ['original_texts', 'row_alignment']:
                st.session_state[key] = []
            elif key == 'preprocessing_settings':
                st.session_state[key] = {
                    'method': 'none',
                    'details': 'No preprocessing applied',
                    'custom_settings': {}
                }
            else:
                st.session_state[key] = None
    
    if reset_items:
        reset_list = ", ".join(set(reset_items))
        st.warning(f"Data configuration changed. Reset: {reset_list}")
        
        if hasattr(st.session_state, 'backend') and st.session_state.backend:
            st.session_state.backend.track_activity(st.session_state.session_id, "cascade_reset", {
                "trigger": "data_loading_change",
                "steps_reset": reset_items
            })

def cascade_from_preprocessing():
    """Reset downstream steps when preprocessing changes"""
    downstream_keys = ['clustering_results', 'tab_c_complete']
    
    reset_occurred = False
    for key in downstream_keys:
        if key in st.session_state and st.session_state[key] is not None:
            st.session_state[key] = None
            reset_occurred = True
    
    if reset_occurred:
        st.warning("Preprocessing changed. Reset clustering results.")
        
        if hasattr(st.session_state, 'backend') and st.session_state.backend:
            st.session_state.backend.track_activity(st.session_state.session_id, "cascade_reset", {
                "trigger": "preprocessing_change", 
                "steps_reset": ["clustering"]
            })

# utils/session_state.py - Quick fix for the AttributeError
# Replace the detect_changes_and_cascade function with this version:

def detect_changes_and_cascade():
    """Detect significant changes and cascade resets"""
    
    # Initialize state_fingerprints if it doesn't exist or is None
    if 'state_fingerprints' not in st.session_state or st.session_state.state_fingerprints is None:
        st.session_state.state_fingerprints = {}
    
    current_fingerprint = {
        'file_key': st.session_state.get('previous_file_key'),
        'text_column': st.session_state.get('text_column'),
        'id_column': st.session_state.get('respondent_id_column'),
        'df_shape': tuple(st.session_state.df.shape) if st.session_state.get('df') is not None else None
    }
    
    previous_fingerprint = st.session_state.state_fingerprints.get('last_known', {})
    
    # Check for significant changes
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
    
    # Cascade if needed
    if (significant_changes and 
        (st.session_state.get('tab_b_complete') or st.session_state.get('clustering_results'))):
        cascade_from_data_loading()
    
    # Store current fingerprint
    st.session_state.state_fingerprints['last_known'] = current_fingerprint
def check_automatic_completion():
    """Check and auto-complete steps when conditions are met"""
    
    # Auto-complete Data Loading
    if (not st.session_state.get('tab_a_complete', False) and
        st.session_state.get('df') is not None and
        st.session_state.get('text_column') is not None):
        
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
            except Exception:
                pass  # Skip auto-completion if validation fails
    
    # Auto-complete Preprocessing
    if (not st.session_state.get('tab_b_complete', False) and
        st.session_state.get('tab_a_complete', False) and
        st.session_state.get('processed_texts') is not None and
        len(st.session_state.get('processed_texts', [])) >= 10):
        
        st.session_state.tab_b_complete = True
    
    # Auto-complete Clustering
    if (not st.session_state.get('tab_c_complete', False) and
        st.session_state.get('clustering_results') is not None and
        st.session_state.clustering_results.get('success', False)):
        
        st.session_state.tab_c_complete = True