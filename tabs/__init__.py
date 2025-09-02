# tabs/__init__.py
"""
Tab modules for Clustery application with sidebar navigation support
"""

from .data_loading import tab_a_data_loading
from .preprocessing import tab_b_preprocessing
from .clustering import tab_c_clustering
from .results import tab_d_results
from .finetuning import tab_e_finetuning

# Navigation helper functions
def get_tab_status():
    """Get the completion status of all tabs"""
    import streamlit as st
    
    return {
        'data_loading': bool(st.session_state.get('tab_a_complete', False)),
        'preprocessing': bool(st.session_state.get('tab_b_complete', False)),
        'clustering': bool(st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success", False)),
        'results': bool(st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success", False)),
        'finetuning': bool(st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success", False))
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
    'tab_a_data_loading',
    'tab_b_preprocessing', 
    'tab_c_clustering',
    'tab_d_results',
    'tab_e_finetuning',
    'get_tab_status',
    'navigate_to_next_tab',
    'set_current_page'
]