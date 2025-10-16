# main.py - Complete updated Clustery Application with unified reset system
import streamlit as st
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Clustery - Text Clustering Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            progress_bar = st.progress(0.0)
        
        with status_container:
            status_text = st.empty()
        
        # Step 1: Basic imports (fast)
        status_text.info("🔧 Loading core components...")
        progress_bar.progress(0.25)
        time.sleep(0.3)
        
        # Step 2: Import backend
        status_text.info("🚀 Setting up backend services...")
        progress_bar.progress(0.5)
        
        try:
            from backend import ClusteryBackend
            st.session_state.BACKEND_AVAILABLE = True
            status_text.success("✅ Backend connected")
        except ImportError as e:
            st.session_state.BACKEND_AVAILABLE = False
            st.error(f"❌ Backend not found! {e}")
            st.error("Please ensure backend.py is in your project directory.")
            return False
        except Exception as e:
            st.session_state.BACKEND_AVAILABLE = False
            st.error(f"❌ Backend initialization failed: {e}")
            return False
        
        # Step 3: Import tab modules
        status_text.info("📋 Loading application modules...")
        progress_bar.progress(0.75)
        
        try:
            from frontend.frontend_data_loading import tab_data_loading
            from frontend.frontend_preprocessing import tab_preprocessing  
            from frontend.frontend_clustering import tab_clustering
            from frontend.frontend_results import tab_results
            from frontend.frontend_finetuning import tab_finetuning
            from utils.session_state import initialize_session_state, reset_analysis
            from utils.styles import apply_custom_styles
            
            # Store in session state for reuse
            st.session_state.tab_functions = {
                'data_loading': tab_data_loading,
                'preprocessing': tab_preprocessing,
                'clustering': tab_clustering,
                'results': tab_results,
                'finetuning': tab_finetuning
            }
            st.session_state.initialize_session_state = initialize_session_state
            st.session_state.apply_custom_styles = apply_custom_styles
            st.session_state.reset_analysis = reset_analysis
            
        except ImportError as e:
            st.error(f"❌ Failed to import modules: {e}")
            st.error("Please check that all files are in the correct directories.")
            return False
        
        # Step 4: Initialize session state
        status_text.info("🔧 Initializing session...")
        progress_bar.progress(0.9)
        
        # Initialize session state and apply styles
        initialize_session_state(st.session_state.BACKEND_AVAILABLE)
        apply_custom_styles()
        
        # Complete setup
        status_text.success("✅ Clustery ready!")
        progress_bar.progress(1.0)
        
        time.sleep(0.5)
        
        # Clean up initialization UI
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.app_initialized = True
        st.balloons()
        time.sleep(1)
        st.rerun()
    
    return True

def create_sidebar_navigation():
    """Create clean sidebar with unified reset system"""
    
    # Disable sidebar scrolling
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        overflow: hidden !important;
        max-height: 100vh !important;
    }
    section[data-testid="stSidebar"] > div {
        overflow: hidden !important;
        height: 100vh !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # App branding
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #667eea; margin: 0; font-size: 1.8rem;">🔍 Clustery</h1>
            <p style="color: #666; margin: 5px 0 0 0; font-size: 0.9rem;">Text Clustering Tool</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Initialize permanent progress tracking - these NEVER get reset except by explicit user reset
        if 'permanent_progress' not in st.session_state:
            st.session_state.permanent_progress = {
                'data_loading': False,
                'preprocessing': False, 
                'clustering': False
            }
        
        # Update permanent progress based on current completion (only goes UP, never DOWN)
        current_data_complete = bool(st.session_state.get('tab_data_loading_complete', False))
        current_preprocessing_complete = bool(st.session_state.get('tab_preprocessing_complete', False))
        current_clustering_complete = bool(st.session_state.get('clustering_results') and 
                                         st.session_state.clustering_results.get("success", False))
        
        # STICKY updates: only set to True, never False (unless explicit reset)
        st.session_state.permanent_progress['data_loading'] = (
            st.session_state.permanent_progress.get('data_loading', False) or current_data_complete
        )
        if current_preprocessing_complete:
            st.session_state.permanent_progress['preprocessing'] = True
        if current_clustering_complete:
            st.session_state.permanent_progress['clustering'] = True
        
        # Get permanent status for display
        data_ever_completed = st.session_state.permanent_progress['data_loading']
        preprocessing_ever_completed = st.session_state.permanent_progress['preprocessing'] 
        clustering_ever_completed = st.session_state.permanent_progress['clustering']
        
        # Initialize current page if not set
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "data_loading"

        st.markdown("### Navigation")
        
        # 1. Data Loading - Always accessible
        completion_indicator = "✅" if current_data_complete else "⭕"
        button_style = "primary" if st.session_state.current_page == "data_loading" else "secondary"
        
        if st.button(f"{completion_indicator} Data Loading", 
                    type=button_style,
                    use_container_width=True,
                    key="nav_data_loading"):
            st.session_state.current_page = "data_loading"
            st.rerun()
        
        # 2. Preprocessing - Accessible if ever completed data loading
        completion_indicator = "✅" if current_preprocessing_complete else "⭕"
        is_accessible = data_ever_completed
        
        if is_accessible:
            button_style = "primary" if st.session_state.current_page == "preprocessing" else "secondary"
            if st.button(f"{completion_indicator} Preprocessing", 
                        type=button_style,
                        use_container_width=True,
                        key="nav_preprocessing"):
                st.session_state.current_page = "preprocessing"
                st.rerun()
        else:
            if st.button(f"{completion_indicator} Preprocessing", 
                        type="secondary",
                        use_container_width=True,
                        key="nav_preprocessing",
                        disabled=True):
                pass
        
        # 3. Clustering - Accessible if ever completed preprocessing
        completion_indicator = "✅" if current_clustering_complete else "⭕"
        is_accessible = preprocessing_ever_completed
        
        if is_accessible:
            button_style = "primary" if st.session_state.current_page == "clustering" else "secondary"
            if st.button(f"{completion_indicator} Clustering", 
                        type=button_style,
                        use_container_width=True,
                        key="nav_clustering"):
                st.session_state.current_page = "clustering"
                st.rerun()
        else:
            if st.button(f"{completion_indicator} Clustering", 
                        type="secondary",
                        use_container_width=True,
                        key="nav_clustering",
                        disabled=True):
                pass
        
        # 4. Fine-tuning - Accessible if clustering was ever completed
        finetuning_ever_visited = st.session_state.get("finetuning_ever_visited", False)
        completion_indicator = "✅" if finetuning_ever_visited else "⭕"
        is_accessible = clustering_ever_completed
        
        if is_accessible:
            button_style = "primary" if st.session_state.current_page == "finetuning" else "secondary"
            if st.button(f"{completion_indicator} Fine-tuning",
                        type=button_style,
                        use_container_width=True,
                        key="nav_finetuning"):
                st.session_state["finetuning_ever_visited"] = True
                st.session_state.current_page = "finetuning"
                st.rerun()
        else:
            if st.button(f"{completion_indicator} Fine-tuning",
                        type="secondary",
                        use_container_width=True,
                        key="nav_finetuning",
                        disabled=True):
                pass
        
        # 5. Results - Accessible if finetuning is visited
        finetuning_ever_visited = st.session_state.get("finetuning_ever_visited", False)
        results_ever_visited = st.session_state.get("results_ever_visited", False)
        completion_indicator = "✅" if results_ever_visited else "⭕"
        is_accessible = finetuning_ever_visited
        
        if is_accessible:
            button_style = "primary" if st.session_state.current_page == "results" else "secondary"
            if st.button(f"{completion_indicator} Results", 
                        type=button_style,
                        use_container_width=True,
                        key="nav_results"):
                st.session_state["results_ever_visited"] = True
                st.session_state.current_page = "results"
                st.rerun()
        else:
            if st.button(f"{completion_indicator} Results", 
                        type="secondary",
                        use_container_width=True,
                        key="nav_results",
                        disabled=True):
                pass
        
        st.markdown("---")
        
        # Debug info
        with st.expander("Debug", expanded=False):
            st.write("Current Status:")
            st.write(f"Data Complete: {current_data_complete}")
            st.write(f"Preprocessing Complete: {current_preprocessing_complete}") 
            st.write(f"Clustering Complete: {current_clustering_complete}")
            st.write("Permanent Progress:")
            st.write(f"Data Ever: {data_ever_completed}")
            st.write(f"Preprocessing Ever: {preprocessing_ever_completed}")
            st.write(f"Clustering Ever: {clustering_ever_completed}")
        
        # Reset button using unified reset system
        if st.button("🔄 Start New Analysis", 
                    help="Clear all data and start over",
                    use_container_width=True,
                    key="reset_analysis_btn"):
            
            # Use unified reset system for complete reset
            from utils.reset_manager import reset_full_analysis
            reset_summary = reset_full_analysis(
                preserve_columns=False, 
                show_message=True
            )
            
            # Additional UI state for file uploader
            st.session_state.file_uploader_reset = True
            st.session_state.file_reset_reason = "start_new_analysis"
            st.session_state["data_loading_alerts"] = []
            
            st.rerun()


def reset_downstream_from_data_loading():
    """Use unified reset system for data loading changes"""
    from utils.reset_manager import reset_from_column_change
    return reset_from_column_change("data_loading", show_message=False)


# ============================================================================
# MAIN CONTENT RENDERING
# ============================================================================

def render_main_content():
    """Render the main content area based on selected page"""
    
    current_page = st.session_state.get('current_page', 'data_loading')
    
    # Show backend status warning if needed
    if not st.session_state.get('BACKEND_AVAILABLE', False):
        st.error("⚠️ Backend not available! Some features may be limited.")
        st.info("Please ensure backend.py is in your project directory and all dependencies are installed.")
    
    # Get tab functions
    if 'tab_functions' in st.session_state:
        tab_functions = st.session_state.tab_functions
    else:
        # Fallback import if not in session state
        try:
            from frontend.frontend_data_loading import tab_data_loading
            from frontend.frontend_preprocessing import tab_preprocessing
            from frontend.frontend_clustering import tab_clustering
            from frontend.frontend_results import tab_results
            from frontend.frontend_finetuning import tab_finetuning

            tab_functions = {
                'data_loading': tab_data_loading,
                'preprocessing': tab_preprocessing,
                'clustering': tab_clustering,
                'results': tab_results,
                'finetuning': tab_finetuning,
            }
        except ImportError as e:
            st.error(f"Failed to import tab functions: {e}")
            st.stop()

    
    # Render appropriate page content
    try:
        if current_page == "data_loading":
            st.markdown("# 📁 Data Loading")
            st.markdown("Upload and configure your data for clustering analysis.")
            st.markdown("---")
            tab_functions['data_loading'](st.session_state.get('BACKEND_AVAILABLE', False))
            
        elif current_page == "preprocessing":
            st.markdown("# 🧹 Preprocessing")
            st.markdown("Clean and prepare your text data for optimal clustering results.")
            st.markdown("---")
            tab_functions['preprocessing'](st.session_state.get('BACKEND_AVAILABLE', False))
            
        elif current_page == "clustering":
            st.markdown("# 🔍 Clustering")
            st.markdown("Configure parameters and run the clustering algorithm.")
            st.markdown("---")
            tab_functions['clustering'](st.session_state.get('BACKEND_AVAILABLE', False))
            
        elif current_page == "finetuning":
            st.markdown("# 🧩 Fine-tuning (Optional)")
            st.markdown("Manually adjust your clustering results using drag-and-drop and AI assistance.")
            st.markdown("---")
            tab_functions['finetuning'](st.session_state.get('BACKEND_AVAILABLE', False))
            
        elif current_page == "results":
            st.markdown("# 📊 Results")
            st.markdown("Explore your clustering results and export findings.")
            st.markdown("---")
            tab_functions['results'](st.session_state.get('BACKEND_AVAILABLE', False))
            
    except Exception as e:
        st.error(f"Error rendering {current_page}: {e}")
        st.exception(e)
        st.info("Try refreshing the page or restarting the analysis.")

def main():
    """Main app entry point"""
    
    # Check if app needs initialization
    if not st.session_state.get('app_initialized', False):
        if not initialize_app_with_progress():
            st.stop()
        return
    
    # Apply custom styles (already initialized)
    if 'apply_custom_styles' in st.session_state:
        st.session_state.apply_custom_styles()

    # >>> Ensure flags are up-to-date BEFORE building the sidebar <<<
    try:
        from utils.session_state import detect_changes_and_cascade, check_automatic_completion
        detect_changes_and_cascade()
        check_automatic_completion()
    except Exception as e:
        st.warning(f"Session state management warning: {e}")
    
    # Create sidebar navigation (reads fresh flags now)
    create_sidebar_navigation()
    
    # Render main content
    render_main_content()

if __name__ == "__main__":
    main()

