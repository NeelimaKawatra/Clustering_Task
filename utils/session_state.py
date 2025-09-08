# utils/session_state.py - Fixed version with proper persistent selections
import streamlit as st
import time


def initialize_session_state(backend_available=True):
    """Initialize all session state variables with proper defaults"""
    
    # Core data storage
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # User selections - FIXED: Don't reset if already exist
    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {
            'id_column_choice': None,
            'text_column_choice': None,
            'original_columns': []
        }
    
    # Column selections - FIXED: Preserve existing selections
    if 'subjectID' not in st.session_state:
        st.session_state.subjectID = "-- Select a subject ID column--"
    
    if 'text_column' not in st.session_state:
        st.session_state.text_column = "-- Select a text column --"
    
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
    
    # Tab completion status
    if 'tab_data_loading_complete' not in st.session_state:
        st.session_state.tab_data_loading_complete = False
    
    if 'tab_preprocessing_complete' not in st.session_state:
        st.session_state.tab_preprocessing_complete = False
    
    if 'tab_clustering_complete' not in st.session_state:
        st.session_state.tab_clustering_complete = False
    
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

def reset_analysis():
    """Reset all analysis data while preserving navigation state - FIXED VERSION"""
    
    # Keys to completely reset (including column selections)
    keys_to_reset = [
        'df', 'original_texts', 'processed_texts',
        'preprocessing_metadata', 'preprocessing_settings', 'row_alignment',
        'clustering_results', 'tab_data_loading_complete', 'tab_preprocessing_complete',
        'tab_clustering_complete', 'previous_file_key', 'state_fingerprints'
    ]
    
    # FIXED: Also reset column selections when doing full analysis reset
    column_selection_keys = [
        'subjectID', 
        'text_column', 
        'user_selections'
    ]
    
    # Reset main keys
    for key in keys_to_reset:
        if key in st.session_state:
            if key == 'preprocessing_settings':
                st.session_state[key] = {
                    'method': 'none',
                    'details': 'No preprocessing applied',
                    'custom_settings': {}
                }
            elif key in ['clean_ids', 'original_texts', 'row_alignment']:
                st.session_state[key] = []
            else:
                st.session_state[key] = None
    
    # Reset column selections
    for key in column_selection_keys:
        if key == 'user_selections':
            st.session_state[key] = {
                'id_column_choice': None,
                'text_column_choice': None,
                'original_columns': []
            }
        elif key == 'text_column':
            st.session_state[key] = "-- Select a text column --"
        elif key == 'subjectID':
            st.session_state[key] = "-- Select a subject ID column--"
    
    # Reset to first page
    st.session_state.current_page = "data_loading"
    
    # Generate new file uploader key to force file uploader to clear
    st.session_state.file_uploader_key = f"uploader_{int(time.time())}"
    st.session_state.file_uploader_reset = True

def cascade_from_data_loading():
    """Reset downstream steps when data changes - FIXED to preserve column selections"""
    
    downstream_keys = [
        'processed_texts', 'original_texts', 'preprocessing_settings', 'preprocessing_metadata',
        'row_alignment', 'tab_preprocessing_complete', 'clustering_results', 'tab_clustering_complete'
    ]
    
    reset_items = []
    for key in downstream_keys:
        if key in st.session_state and st.session_state[key] is not None:
            if key == 'tab_preprocessing_complete' and st.session_state[key]:
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
    
    # IMPORTANT: Do NOT reset column selections during cascade
    # Column selections should only reset on full analysis reset or new file upload
    
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
    downstream_keys = ['clustering_results', 'tab_clustering_complete']
    
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

def detect_changes_and_cascade():
    """Detect significant changes and cascade resets - Enhanced with column change detection"""
    
    # Initialize state_fingerprints if it doesn't exist
    if 'state_fingerprints' not in st.session_state or st.session_state.state_fingerprints is None:
        st.session_state.state_fingerprints = {}
    
    current_fingerprint = {
        'file_key': st.session_state.get('previous_file_key'),
        'df_shape': tuple(st.session_state.df.shape) if st.session_state.get('df') is not None else None,
        'text_column': st.session_state.get('text_column'),
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
    if (current_fingerprint.get('text_column') != previous_fingerprint.get('text_column') and
        previous_fingerprint.get('text_column') is not None and
        previous_fingerprint.get('text_column') not in ["-- Select a text column --", None] and
        current_fingerprint.get('text_column') not in ["-- Select a text column --", None]):
        significant_changes.append("text_column")
    
    # treat changes between real selections as significant (ignore prompts/None)
    if (current_fingerprint.get('id_column') != previous_fingerprint.get('id_column') and
        previous_fingerprint.get('id_column') not in [None, "-- Select a subject ID column--"] and
        current_fingerprint.get('id_column') not in [None, "-- Select a subject ID column--"]):
        significant_changes.append("id_column")
    
    # Cascade if there are significant changes AND downstream processing exists
    if (significant_changes and 
        (st.session_state.get('tab_preprocessing_complete') or st.session_state.get('clustering_results'))):
        cascade_from_data_loading()
    
    # Store current fingerprint
    st.session_state.state_fingerprints['last_known'] = current_fingerprint

def check_automatic_completion():
    """Check and auto-complete steps when conditions are met"""
    
    # Auto-complete Data Loading
    if (not st.session_state.get('tab_data_loading_complete', False) and
        st.session_state.get('df') is not None and
        st.session_state.get('text_column') is not None and
        st.session_state.get('text_column') != "-- Select a text column --"):
        
        if hasattr(st.session_state, 'backend') and st.session_state.backend:
            try:
                validation = st.session_state.backend.validate_columns(
                    st.session_state.df, 
                    st.session_state.text_column,
                    st.session_state.subjectID,
                    st.session_state.session_id
                )
                
                if validation.get('text_column_valid', False):
                    st.session_state.tab_data_loading_complete = True
            except Exception:
                pass  # Skip auto-completion if validation fails
    
    # Auto-complete Preprocessing
    if (not st.session_state.get('tab_preprocessing_complete', False) and
        st.session_state.get('tab_data_loading_complete', False) and
        st.session_state.get('processed_texts') is not None and
        len(st.session_state.get('processed_texts', [])) >= 10):
        
        st.session_state.tab_preprocessing_complete = True
    
    # Auto-complete Clustering
    if (not st.session_state.get('tab_clustering_complete', False) and
        st.session_state.get('clustering_results') is not None and
        st.session_state.clustering_results.get('success', False)):
        
        st.session_state.tab_clustering_complete = True

def clear_file_uploader():
    """Clear file uploader state by generating a new key"""
    st.session_state.file_uploader_key = f"uploader_{int(time.time())}"
    st.session_state.file_uploader_reset = True

def reset_file_state():
    """Reset only file-related state variables"""
    file_keys = ['df', 'clean_ids', 'previous_file_key', 'file_uploader_key']
    
    for key in file_keys:
        if key in st.session_state:
            if key == 'clean_ids':
                st.session_state[key] = []
            elif key == 'file_uploader_key':
                st.session_state[key] = f"uploader_{int(time.time())}"
            else:
                st.session_state[key] = None
    
    # Reset tab completion since file data is cleared
    if 'tab_data_loading_complete' in st.session_state:
        st.session_state.tab_data_loading_complete = False
    
    # Set flag to show reset message
    st.session_state.file_uploader_reset = True

def preserve_column_selections():
    """Helper to preserve column selections during navigation - used internally"""
    # This is called during navigation to ensure selections are maintained
    preserved_selections = {
        'subjectID': st.session_state.get('subjectID'),
        'text_column': st.session_state.get('text_column'),
        'user_selections': st.session_state.get('user_selections', {}).copy()
    }
    return preserved_selections

def restore_column_selections(selections):
    """Helper to restore column selections - used internally"""
    for key, value in selections.items():
        if value is not None:
            st.session_state[key] = value

# Updated auto_navigate_to_next_available function for session_state.py

def auto_navigate_to_next_available():
    """Automatically navigate to the next available step"""
    # Set the flag that the sidebar navigation will check
    st.session_state.should_navigate_next = True
    
    # Optional: Add a small delay to ensure the flag is processed
    import time
    time.sleep(0.1)
    
    # Log what step we're trying to navigate to (for debugging)
    data_complete = bool(st.session_state.get('tab_data_loading_complete', False))
    preprocessing_complete = bool(st.session_state.get('tab_preprocessing_complete', False))
    clustering_complete = bool(st.session_state.get('clustering_results') and 
                             st.session_state.clustering_results.get("success", False))
    
    # Determine next step for logging
    if not data_complete:
        next_step = "data_loading"
    elif not preprocessing_complete:
        next_step = "preprocessing"
    elif not clustering_complete:
        next_step = "clustering"
    else:
        next_step = "finetuning_or_results"
    
    # Store for debugging
    st.session_state.auto_nav_target = next_step
    
    """
    # Optional: Show a brief success message
    if hasattr(st, 'success'):
        current_step = st.session_state.get('current_page', 'unknown')
        if current_step != next_step:
            st.success(f"✅ Step completed! Ready for {next_step.replace('_', ' ').title()}")
    """