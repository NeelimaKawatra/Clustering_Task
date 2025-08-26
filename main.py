import streamlit as st
import pandas as pd
import os
import time
from collections import Counter
from utils.session_state import reset_analysis

# ============================================================================
# FAST APP STARTUP WITH PROGRESSIVE LOADING
# ============================================================================

def initialize_app_with_progress():
    """Initialize app with smart progress feedback"""
    
    if 'app_initialized' not in st.session_state:
        # Show initialization screen
        st.markdown("# 🔍 Starting Clustery...")
        
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0.0)  # Fixed: Use 0.0
        
        with status_container:
            status_text = st.empty()
        
        # Step 1: Basic imports (fast)
        status_text.info("🔧 Loading core components...")
        progress_bar.progress(0.25)  # Fixed: Use 0.25
        time.sleep(0.5)
        
        # Step 2: Import backend (now much faster!)
        status_text.info("🚀 Setting up backend services...")
        progress_bar.progress(0.5)  # Fixed: Use 0.5
        
        try:
            from backend import ClusteryBackend
            st.session_state.BACKEND_AVAILABLE = True
        except ImportError:
            st.session_state.BACKEND_AVAILABLE = False
            st.error("❌ Backend not found! Please ensure backend.py is in your project directory.")
            return False
        
        # Step 3: Import tab modules (fast)
        status_text.info("📋 Loading application modules...")
        progress_bar.progress(0.75)  # Fixed: Use 0.75
        
        try:
            from tabs.data_loading import tab_a_data_loading
            from tabs.preprocessing import tab_b_preprocessing  
            from tabs.clustering import tab_c_clustering
            from tabs.results import tab_d_results
            from utils.session_state import initialize_session_state
            from utils.styles import apply_custom_styles
            
            # Store in session state for reuse
            st.session_state.tab_functions = {
                'data_loading': tab_a_data_loading,
                'preprocessing': tab_b_preprocessing,
                'clustering': tab_c_clustering,
                'results': tab_d_results
            }
            st.session_state.initialize_session_state = initialize_session_state
            st.session_state.apply_custom_styles = apply_custom_styles
            
        except ImportError as e:
            st.error(f"❌ Failed to import modules: {e}")
            return False
        
        # Step 4: Complete setup
        status_text.success("✅ Clustery ready!")
        progress_bar.progress(1.0)  # Fixed: Use 1.0 
        
        time.sleep(0.5)
        
        # Clean up initialization UI
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.app_initialized = True
        st.balloons()
        time.sleep(1)
        st.rerun()
    
    return True

# ============================================================================
# SIDEBAR NAVIGATION WITH BUTTONS
# ============================================================================

def create_sidebar_navigation():
    """Create clean sidebar with always-accessible navigation buttons"""
    
    with st.sidebar:
        # App branding
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #667eea; margin: 0; font-size: 1.8rem;">🔍 Clustery</h1>
            <p style="color: #666; margin: 5px 0 0 0; font-size: 0.9rem;">Text Clustering Tool</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Check completion status
        data_complete = bool(st.session_state.get('tab_a_complete', False))
        preprocessing_complete = bool(st.session_state.get('tab_b_complete', False))
        clustering_complete = bool(st.session_state.get('clustering_results') and st.session_state.clustering_results.get("success", False))
        
        # Initialize current page if not set
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "data_loading"
        
        # Navigation buttons - all always accessible
        
        # 1. Data Loading - always available
        completion_indicator = "✅" if data_complete else "⭕"
        if st.button(f"{completion_indicator} Data Loading", 
                    type="primary" if st.session_state.current_page == "data_loading" else "secondary",
                    use_container_width=True,
                    key="nav_data_loading"):
            st.session_state.current_page = "data_loading"
            st.rerun()
        
        # 2. Preprocessing - always accessible
        completion_indicator = "✅" if preprocessing_complete else "⭕"
        if st.button(f"{completion_indicator} Preprocessing", 
                    type="primary" if st.session_state.current_page == "preprocessing" else "secondary",
                    use_container_width=True,
                    key="nav_preprocessing"):
            st.session_state.current_page = "preprocessing"
            st.rerun()
        
        # 3. Clustering - always accessible
        completion_indicator = "✅" if clustering_complete else "⭕"
        if st.button(f"{completion_indicator} Clustering", 
                    type="primary" if st.session_state.current_page == "clustering" else "secondary",
                    use_container_width=True,
                    key="nav_clustering"):
            st.session_state.current_page = "clustering"
            st.rerun()
        
        # 4. Results - always accessible
        completion_indicator = "✅" if clustering_complete else "⭕"
        if st.button(f"{completion_indicator} Results", 
                    type="primary" if st.session_state.current_page == "results" else "secondary",
                    use_container_width=True,
                    key="nav_results"):
            st.session_state.current_page = "results"
            st.rerun()
        
        st.markdown("---")
        
        # Progress indicator
        progress_steps = [data_complete, preprocessing_complete, clustering_complete]
        completed_steps = sum(progress_steps)
        progress_percentage = completed_steps / len(progress_steps)
        
        st.markdown("**Progress:**")
        st.progress(progress_percentage)
        st.caption(f"{completed_steps}/{len(progress_steps)} steps completed")
        
        st.markdown("---")
        
        # Reset button at the bottom
        if st.button("🔄 Start New Analysis", 
                    help="Clear all data and start over",
                    use_container_width=True,
                    key="reset_analysis"):
            reset_analysis()
            st.rerun()


def show_session_analytics():
    """Show session analytics in sidebar"""
    if st.session_state.get('backend'):
        try:
            session_summary = st.session_state.backend.get_session_analytics(st.session_state.session_id)
            
            st.markdown("**Session Analytics:**")
            st.caption(f"Duration: {session_summary.get('duration_seconds', 0):.0f}s")
            st.caption(f"Activities: {session_summary.get('total_activities', 0)}")
            
            activity_counts = session_summary.get('activity_counts', {})
            if activity_counts:
                st.caption("**Activity Breakdown:**")
                for activity, count in activity_counts.items():
                    activity_name = activity.replace('_', ' ').title()
                    st.caption(f"• {activity_name}: {count}")
        except Exception as e:
            st.caption(f"Analytics error: {str(e)}")


# ============================================================================
# MAIN CONTENT RENDERING
# ============================================================================

def render_main_content():
    """Render the main content area based on selected page"""
    
    current_page = st.session_state.get('current_page', 'data_loading')
    
    # Show backend status warning if needed
    if not st.session_state.get('BACKEND_AVAILABLE', False):
        st.error("Backend not available! Please ensure backend.py is in your project directory.")
        st.stop()
    
    # Auto-detect changes and handle cascading
    from utils.session_state import detect_changes_and_cascade, check_automatic_completion
    detect_changes_and_cascade()
    check_automatic_completion()
    
    # Get tab functions from session state or import them
    if 'tab_functions' in st.session_state:
        tab_functions = st.session_state.tab_functions
    else:
        # Fallback import if not in session state
        from tabs.data_loading import tab_a_data_loading
        from tabs.preprocessing import tab_b_preprocessing
        from tabs.clustering import tab_c_clustering
        from tabs.results import tab_d_results
        
        tab_functions = {
            'data_loading': tab_a_data_loading,
            'preprocessing': tab_b_preprocessing,
            'clustering': tab_c_clustering,
            'results': tab_d_results
        }
    
    # Render appropriate page content
    if current_page == "data_loading":
        st.markdown("# Data Loading")
        st.markdown("Upload and configure your data for clustering analysis.")
        st.markdown("---")
        tab_functions['data_loading'](st.session_state.get('BACKEND_AVAILABLE', False))
        
    elif current_page == "preprocessing":
        st.markdown("# Text Preprocessing")
        st.markdown("Clean and prepare your text data for optimal clustering results.")
        st.markdown("---")
        tab_functions['preprocessing'](st.session_state.get('BACKEND_AVAILABLE', False))
        
    elif current_page == "clustering":
        st.markdown("# Clustering Configuration")
        st.markdown("Configure parameters and run the clustering algorithm.")
        st.markdown("---")
        tab_functions['clustering'](st.session_state.get('BACKEND_AVAILABLE', False))
        
    elif current_page == "results":
        st.markdown("# Clustering Results")
        st.markdown("Explore your clustering results and export findings.")
        st.markdown("---")
        tab_functions['results'](st.session_state.get('BACKEND_AVAILABLE', False))

        
# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main app with fast startup and sidebar navigation"""
    st.set_page_config(
        page_title="Clustery - Text Clustering Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check if app needs initialization
    if not st.session_state.get('app_initialized', False):
        if not initialize_app_with_progress():
            st.stop()
        return
    
    # Apply custom styles
    if 'apply_custom_styles' in st.session_state:
        apply_styles = st.session_state.apply_custom_styles
    else:
        from utils.styles import apply_custom_styles
        apply_styles = apply_custom_styles
    
    apply_styles()
    
    # Initialize session state
    if 'initialize_session_state' in st.session_state:
        init_session = st.session_state.initialize_session_state
    else:
        from utils.session_state import initialize_session_state
        init_session = initialize_session_state
    
    init_session(st.session_state.get('BACKEND_AVAILABLE', False))
    
    # Create sidebar navigation
    create_sidebar_navigation()
    
    # Render main content
    render_main_content()

if __name__ == "__main__":
    main()