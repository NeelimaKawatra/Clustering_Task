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
    
    # Initialize backend and session
    if 'backend' not in st.session_state and backend_available:
        from backend import ClusteryBackend
        st.session_state.backend = ClusteryBackend()
        st.session_state.session_id = f"user_{int(time.time())}"
        st.session_state.backend.start_session(st.session_state.session_id)