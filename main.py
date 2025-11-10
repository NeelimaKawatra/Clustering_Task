# main.py - Complete Clustery Application with ultra-compact sidebar
import streamlit as st
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

# Load the .env that setup_api_keys.py created
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


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
            from frontend.frontend_llm_settings import tab_llm_settings
            from utils.session_state import initialize_session_state, reset_analysis
            from utils.styles import apply_custom_styles
            
            # Store in session state for reuse
            st.session_state.tab_functions = {
                'data_loading': tab_data_loading,
                'preprocessing': tab_preprocessing,
                'clustering': tab_clustering,
                'results': tab_results,
                'finetuning': tab_finetuning,
                'llm_settings': tab_llm_settings
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
    """Create ultra-compact sidebar navigation (fits in one screen)"""
    
    # Ultra-compact CSS styling
    st.markdown("""
    <style>
    /* Minimal padding everywhere */
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Ultra-compact buttons */
    [data-testid="stSidebar"] button {
        padding: 0.25rem 0.5rem !important;
        font-size: 0.8rem !important;
        margin: 0.15rem 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Remove button height constraints */
    [data-testid="stSidebar"] button {
        min-height: unset !important;
        height: auto !important;
    }
    
    /* Ultra-compact headers */
    [data-testid="stSidebar"] h3 {
        font-size: 0.85rem !important;
        margin: 0.4rem 0 0.2rem 0 !important;
        font-weight: 600 !important;
    }
    
    /* Minimal dividers */
    [data-testid="stSidebar"] hr {
        margin: 0.4rem 0 !important;
    }
    
    /* Ultra-compact branding */
    .sidebar-branding h1 {
        font-size: 1.2rem !important;
        margin: 0 !important;
        line-height: 1.3 !important;
    }
    
    .sidebar-branding p {
        font-size: 0.7rem !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Remove extra spacing from Streamlit */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0 !important;
    }
    
    /* Compact markdown */
    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Ultra-compact branding
        st.markdown("""
        <div class="sidebar-branding" style="text-align: center; padding: 0.3rem 0;">
            <h1 style="color: #667eea;">🔍 Clustery</h1>
            <p style="color: #666;">Text Clustering Tool</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Initialize permanent progress tracking
        if 'permanent_progress' not in st.session_state:
            st.session_state.permanent_progress = {
                'data_loading': False,
                'preprocessing': False, 
                'clustering': False
            }
        
        # Update permanent progress based on current completion
        current_data_complete = bool(st.session_state.get('tab_data_loading_complete', False))
        current_preprocessing_complete = bool(st.session_state.get('tab_preprocessing_complete', False))
        current_clustering_complete = bool(st.session_state.get('clustering_results') and 
                                         st.session_state.clustering_results.get("success", False))
        
        # STICKY updates: only set to True, never False
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
        
        # 6. LLM Settings - Always accessible
        button_style = "primary" if st.session_state.current_page == "llm_settings" else "secondary"
        
        # Show LLM status in button text
        if 'llm_config' in st.session_state and st.session_state.llm_config.get('initialized'):
            llm_text = "✅ LLM Settings"
        else:
            llm_text = "🤖 LLM Settings"
        
        if st.button(llm_text,
                    type=button_style,
                    use_container_width=True,
                    key="nav_llm_settings"):
            st.session_state.current_page = "llm_settings"
            st.rerun()
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Start New Analysis", 
                    use_container_width=True,
                    key="reset_analysis_btn"):
            from utils.reset_manager import reset_full_analysis
            reset_full_analysis(preserve_columns=False, show_message=True)
            st.session_state.file_uploader_reset = True
            st.session_state.file_reset_reason = "start_new_analysis"
            st.session_state["data_loading_alerts"] = []
            st.rerun()


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
            from frontend.frontend_llm_settings import tab_llm_settings

            tab_functions = {
                'data_loading': tab_data_loading,
                'preprocessing': tab_preprocessing,
                'clustering': tab_clustering,
                'results': tab_results,
                'finetuning': tab_finetuning,
                'llm_settings': tab_llm_settings
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
        
        elif current_page == "llm_settings":
            st.markdown("# 🤖 LLM Configuration")
            st.markdown("Configure AI assistant settings for all operations.")
            st.markdown("---")
            tab_functions['llm_settings'](st.session_state.get('BACKEND_AVAILABLE', False))
            
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

    # Ensure flags are up-to-date BEFORE building the sidebar
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