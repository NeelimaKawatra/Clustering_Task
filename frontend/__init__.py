# frontend/__init__.py
"""
Tab modules for Clustery application with sidebar navigation support
"""

from .frontend_data_loading import tab_data_loading
from .frontend_preprocessing import tab_preprocessing
from .frontend_clustering import tab_clustering
from .frontend_results import tab_results
from .frontend_finetuning import tab_finetuning
from .frontend_llm_settings import tab_llm_settings  # NEW

# Navigation helper functions
def get_tab_status():
    """Get the completion status of all tabs"""
    import streamlit as st
    
    return {
        'data_loading': bool(st.session_state.get('tab_data_loading_complete', False)),
        'preprocessing': bool(st.session_state.get('tab_preprocessing_complete', False)),
        'clustering': bool(st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success", False)),
        'results': bool(st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success", False)),
        'finetuning': bool(st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success", False)),
        'llm_settings': True  # Always accessible
    }

def navigate_to_next_tab():
    """Navigate to the next available tab"""
    import streamlit as st
    
    status = get_tab_status()
    
    if not status['data_loading']:
        st.session_state.current_page = 'data_loading'
    elif not status['preprocessing']:
        st.session_state.current_page = 'preprocessing'
    elif not status['clustering']:
        st.session_state.current_page = 'clustering'
    elif not status['finetuning']:
        st.session_state.current_page = 'finetuning'
    else:
        st.session_state.current_page = 'results'

def set_current_page(page_name):
    """Set the current page in navigation"""
    import streamlit as st
    st.session_state.current_page = page_name

__all__ = [
    'tab_data_loading',
    'tab_preprocessing', 
    'tab_clustering',
    'tab_results',
    'tab_finetuning',
    'tab_llm_settings',  # NEW
    'get_tab_status',
    'navigate_to_next_tab',
    'set_current_page'
]