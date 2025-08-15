import streamlit as st
import pandas as pd
import os
import time
from collections import Counter

# Import the backend
try:
    from backend import ClusteryBackend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    st.error("❌ Backend not found! Please ensure backend.py is in your project directory.")

# Import tab modules
from tabs.data_loading import tab_a_data_loading
from tabs.preprocessing import tab_b_preprocessing  
from tabs.clustering import tab_c_clustering
from tabs.results import tab_d_results
from utils.session_state import initialize_session_state
from utils.styles import apply_custom_styles

def main():
    """Main app with clean, app-like header"""
    st.set_page_config(
        page_title="Clustery - Text Clustering Tool",
        page_icon="🔍",
        layout="wide"
    )
    
    # Apply custom styles
    apply_custom_styles()
    
    # Simple app header
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <h1 class="logo">🔍 Clustery</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show backend status
    if not BACKEND_AVAILABLE:
        st.error("❌ **Backend not available!** Please ensure backend.py is in your project directory.")
        st.stop()
    
    # Initialize session state
    initialize_session_state(BACKEND_AVAILABLE)
    
    # Create tabs - these are the functional navigation
    tabs = st.tabs(["📁 Data Loading", "🔧 Preprocessing", "⚙️ Clustering", "📊 Results"])
    
    # TAB A: DATA LOADING
    with tabs[0]:
        tab_a_data_loading(BACKEND_AVAILABLE)
    
    # TAB B: PREPROCESSING
    with tabs[1]:
        if st.session_state.get('tab_a_complete', False):
            tab_b_preprocessing(BACKEND_AVAILABLE)
        else:
            st.info("💡 Complete Data Loading first to unlock this tab.")
    
    # TAB C: CLUSTERING
    with tabs[2]:
        if st.session_state.get('tab_b_complete', False):
            tab_c_clustering(BACKEND_AVAILABLE)
        else:
            st.info("💡 Complete Data Loading and Preprocessing first.")
    
    # TAB D: RESULTS
    with tabs[3]:
        if st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success"):
            tab_d_results(BACKEND_AVAILABLE)
        else:
            st.info("💡 Complete clustering to see results.")

def show_analytics_sidebar():
    """Optional analytics sidebar for development/admin"""
    if BACKEND_AVAILABLE and st.session_state.get('backend'):
        with st.sidebar:
            st.subheader("📊 Session Analytics")
            
            if st.button("Show Analytics"):
                session_summary = st.session_state.backend.get_session_analytics(st.session_state.session_id)
                
                st.write(f"**Session ID:** {st.session_state.session_id}")
                st.write(f"**Duration:** {session_summary.get('duration_seconds', 0):.0f}s")
                st.write(f"**Completion:** {session_summary.get('completion_percentage', 0):.0f}%")
                
                activity_counts = session_summary.get('activity_counts', {})
                if activity_counts:
                    st.write("**Activities:**")
                    for activity, count in activity_counts.items():
                        st.write(f"• {activity}: {count}")

if __name__ == "__main__":
    main()
    
    # Optional: Show analytics in sidebar during development
    # Uncomment the line below to enable analytics sidebar
    # show_analytics_sidebar()