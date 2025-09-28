# utils/session_state.py - Updated with unified reset system
import streamlit as st
import time

def initialize_session_state(backend_available=True):
    """Initialize all session state variables with proper defaults"""
    
    # Core data storage
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # User selections - Don't reset if already exist
    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {
            'id_column_choice': None,
            'entry_column_choice': None,
            'original_columns': []
        }
    
    # Column selections - Preserve existing selections
    if 'subjectID' not in st.session_state:
        st.session_state.subjectID = "-- Select a subject ID column--"
    
    if 'entry_column' not in st.session_state:
        st.session_state.entry_column = "-- Select an entry column --"
    
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
    
    # Row alignment tracking
    if 'row_alignment' not in st.session_state:
        st.session_state.row_alignment = []
    
    # Clustering results
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None
    
    # Finetuning results
    if 'finetuning_results' not in st.session_state:
        st.session_state.finetuning_results = None
    
    if 'finetuning_initialized' not in st.session_state:
        st.session_state.finetuning_initialized = False
    
    if 'finetuning_ever_visited' not in st.session_state:
        st.session_state.finetuning_ever_visited = False
    
    # Tab completion status
    if 'tab_data_loading_complete' not in st.session_state:
        st.session_state.tab_data_loading_complete = False
    
    if 'tab_preprocessing_complete' not in st.session_state:
        st.session_state.tab_preprocessing_complete = False
    
    if 'tab_clustering_complete' not in st.session_state:
        st.session_state.tab_clustering_complete = False
    
    # Processing tracking
    if 'preprocessing_tracked' not in st.session_state:
        st.session_state.preprocessing_tracked = False
    
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
    
    # File uploader management
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = f"uploader_{int(time.time())}"
    
    if 'file_uploader_reset' not in st.session_state:
        st.session_state.file_uploader_reset = False

    if 'file_reset_reason' not in st.session_state:
        st.session_state.file_reset_reason = None
    
    if 'data_loading_alerts' not in st.session_state:
        st.session_state.data_loading_alerts = []

    # Permanent progress tracking
    if 'permanent_progress' not in st.session_state:
        st.session_state.permanent_progress = {
            'data_loading': False,
            'preprocessing': False,
            'clustering': False
        }


def reset_analysis():
    """Reset all analysis data using unified reset system"""
    from .reset_manager import reset_full_analysis
    return reset_full_analysis(preserve_columns=False, show_message=True)


def cascade_from_data_loading():
    """Reset downstream steps when data changes using unified reset system"""
    from .reset_manager import reset_from_column_change
    return reset_from_column_change("data_loading", show_message=True)


def cascade_from_preprocessing():
    """Reset downstream steps when preprocessing changes using unified reset system"""
    from .reset_manager import reset_from_preprocessing_change
    return reset_from_preprocessing_change(show_message=True)


def detect_changes_and_cascade():
    """Detect significant changes and cascade resets using unified reset system"""
    
    # Initialize state_fingerprints if it doesn't exist
    if 'state_fingerprints' not in st.session_state or st.session_state.state_fingerprints is None:
        st.session_state.state_fingerprints = {}
    
    current_fingerprint = {
        'file_key': st.session_state.get('previous_file_key'),
        'df_shape': tuple(st.session_state.df.shape) if st.session_state.get('df') is not None else None,
        'entry_column': st.session_state.get('entry_column'),
        'id_column': st.session_state.get('subjectID')
    }
    
    previous_fingerprint = st.session_state.state_fingerprints.get('last_known', {})
    
    # Check for significant changes
    significant_changes = []
    
    # File changes - most significant
    if (current_fingerprint.get('file_key') != previous_fingerprint.get('file_key') and 
        previous_fingerprint.get('file_key') is not None):
        significant_changes.append("file")
    
    # Data structure changes
    if (current_fingerprint.get('df_shape') != previous_fingerprint.get('df_shape') and 
        previous_fingerprint.get('df_shape') is not None):
        significant_changes.append("data_structure")
    
    # Column selection changes - should reset downstream processing
    if (current_fingerprint.get('entry_column') != previous_fingerprint.get('entry_column') and
        previous_fingerprint.get('entry_column') is not None and
        previous_fingerprint.get('entry_column') not in ["-- Select an entry column --", None] and
        current_fingerprint.get('entry_column') not in ["-- Select an entry column --", None]):
        significant_changes.append("entry_column")
    
    if (current_fingerprint.get('id_column') != previous_fingerprint.get('id_column') and
        previous_fingerprint.get('id_column') not in [None, "-- Select a subject ID column--"] and
        current_fingerprint.get('id_column') not in [None, "-- Select a subject ID column--"]):
        significant_changes.append("id_column")
    
    # Use unified reset system for changes
    if (significant_changes and 
        (st.session_state.get('tab_preprocessing_complete') or 
         st.session_state.get('clustering_results') or 
         st.session_state.get('finetuning_results'))):
        
        from .reset_manager import reset_from_file_change, reset_from_column_change
        
        if "file" in significant_changes:
            reset_from_file_change(show_message=True)
        elif "entry_column" in significant_changes:
            reset_from_column_change("entry", show_message=True)
        elif "id_column" in significant_changes:
            reset_from_column_change("id", show_message=True)
        else:
            cascade_from_data_loading()
    
    # Store current fingerprint
    st.session_state.state_fingerprints['last_known'] = current_fingerprint


def check_automatic_completion():
    """Check and auto-complete steps when conditions are met"""
    
    # Auto-complete Data Loading
    if (not st.session_state.get('tab_data_loading_complete', False) and
        st.session_state.get('df') is not None and
        st.session_state.get('entry_column') is not None and
        st.session_state.get('entry_column') not in [None, "-- Select an entry column --"]):
        
        if hasattr(st.session_state, 'backend') and st.session_state.backend:
            try:
                validation = st.session_state.backend.validate_columns(
                    st.session_state.df, 
                    st.session_state.entry_column,
                    st.session_state.subjectID,
                    st.session_state.session_id
                )
                
                if validation.get('text_column_valid', False):
                    st.session_state.tab_data_loading_complete = True
                    st.session_state.permanent_progress['data_loading'] = True
            except Exception:
                pass  # Skip auto-completion if validation fails
    
    # Auto-complete Preprocessing
    if (not st.session_state.get('tab_preprocessing_complete', False) and
        st.session_state.get('tab_data_loading_complete', False) and
        st.session_state.get('processed_texts') is not None and
        len(st.session_state.get('processed_texts', [])) > 0):
        
        st.session_state.tab_preprocessing_complete = True
        st.session_state.permanent_progress['preprocessing'] = True
    
    # Auto-complete Clustering
    if (not st.session_state.get('tab_clustering_complete', False) and
        st.session_state.get('clustering_results') is not None and
        st.session_state.clustering_results.get('success', False)):
        
        st.session_state.tab_clustering_complete = True
        st.session_state.permanent_progress['clustering'] = True


# Legacy functions for backward compatibility
def clear_file_uploader():
    """Clear file uploader state by generating a new key"""
    st.session_state.file_uploader_key = f"uploader_{int(time.time())}"

def reset_file_state():
    """Reset only file-related state variables using unified reset system"""
    from .reset_manager import ResetManager
    manager = ResetManager()
    return manager.unified_reset(
        reset_type="file_change",
        preserve_columns=True,
        preserve_navigation=True,
        trigger_reason="file_reset",
        show_message=False
    )

def preserve_column_selections():
    """Helper to preserve column selections during navigation"""
    preserved_selections = {
        'subjectID': st.session_state.get('subjectID'),
        'entry_column': st.session_state.get('entry_column'),
        'user_selections': st.session_state.get('user_selections', {}).copy()
    }
    return preserved_selections

def restore_column_selections(selections):
    """Helper to restore column selections"""
    for key, value in selections.items():
        if value is not None:
            st.session_state[key] = value

def auto_navigate_to_next_available():
    """Automatically navigate to the next available step"""
    st.session_state.should_navigate_next = True
    
    import time
    time.sleep(0.1)
    
    # Determine next step for logging
    data_complete = bool(st.session_state.get('tab_data_loading_complete', False))
    preprocessing_complete = bool(st.session_state.get('tab_preprocessing_complete', False))
    clustering_complete = bool(st.session_state.get('clustering_results') and 
                             st.session_state.clustering_results.get("success", False))
    
    if not data_complete:
        next_step = "data_loading"
    elif not preprocessing_complete:
        next_step = "preprocessing"
    elif not clustering_complete:
        next_step = "clustering"
    else:
        next_step = "finetuning_or_results"
    
    st.session_state.auto_nav_target = next_step