# main.py - Fixed Clustery Application
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

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================


def create_sidebar_navigation():
    """Create clean sidebar with auto-navigation support"""
    
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
        
        # # Backend status
        # if st.session_state.get('BACKEND_AVAILABLE', False):
        #     st.success("✅ Backend Connected")
        # else:
        #     st.error("❌ Backend Unavailable")
        
        # st.markdown("---")
        
        # Check completion status
        data_complete = bool(st.session_state.get('tab_data_loading_complete', False))
        preprocessing_complete = bool(st.session_state.get('tab_preprocessing_complete', False))
        clustering_complete = bool(st.session_state.get('clustering_results') and 
                                 st.session_state.clustering_results.get("success", False))
        
        # Initialize current page if not set
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "data_loading"
        
        
        # AUTO-NAVIGATION LOGIC - This is what was missing!
        # Check if we should auto-navigate after completion
        if st.session_state.get('should_navigate_next', False):
            # Determine next step
            if data_complete and not preprocessing_complete:
                st.session_state.current_page = "preprocessing"
            elif preprocessing_complete and not clustering_complete:
                st.session_state.current_page = "clustering"  
            elif clustering_complete and st.session_state.current_page != "results":
                # Give user choice between results and finetuning
                if st.session_state.current_page == "clustering":
                    st.session_state.current_page = "finetuning"
        
            # Clear the navigation flag
            st.session_state.should_navigate_next = False

        st.markdown("### Navigation")
        
        # 1. Data Loading - always available
        completion_indicator = "✅" if data_complete else "⭕"
        if st.button(f"{completion_indicator} Data Loading", 
                    type="primary" if st.session_state.current_page == "data_loading" else "secondary",
                    use_container_width=True,
                    key="nav_data_loading"):
            st.session_state.current_page = "data_loading"
            st.rerun()
        
        # 2. Preprocessing - accessible but show warning if prerequisites not met
        completion_indicator = "✅" if preprocessing_complete else "⭕"
        button_type = "primary" if st.session_state.current_page == "preprocessing" else "secondary"
        
        if st.button(f"{completion_indicator} Preprocessing", 
                    type=button_type,
                    use_container_width=True,
                    key="nav_preprocessing"):
            if not data_complete:
                st.error("⚠️ Complete Data Loading first!")
                st.session_state.current_page = "data_loading"
            else:
                st.session_state.current_page = "preprocessing"
            st.rerun()
        
        # 3. Clustering - accessible but show warning if prerequisites not met
        completion_indicator = "✅" if clustering_complete else "⭕"
        button_type = "primary" if st.session_state.current_page == "clustering" else "secondary"
        
        if st.button(f"{completion_indicator} Clustering", 
                    type=button_type,
                    use_container_width=True,
                    key="nav_clustering"):
            if not preprocessing_complete:
                st.error("⚠️ Complete Preprocessing first!")
                if not data_complete:
                    st.session_state.current_page = "data_loading"
                else:
                    st.session_state.current_page = "preprocessing"
            else:
                st.session_state.current_page = "clustering"
            st.rerun()
        
        # 4. Fine-tuning - accessible but show warning if prerequisites not met
        completion_indicator = "✅" if clustering_complete else "⭕"
        button_type = "primary" if st.session_state.current_page == "finetuning" else "secondary"
        
        if st.button(f"{completion_indicator} Fine-tuning", 
                    type=button_type,
                    use_container_width=True,
                    key="nav_finetuning"):
            if not clustering_complete:
                st.error("⚠️ Complete Clustering first!")
                # Navigate to the incomplete step
                if not data_complete:
                    st.session_state.current_page = "data_loading"
                elif not preprocessing_complete:
                    st.session_state.current_page = "preprocessing"
                else:
                    st.session_state.current_page = "clustering"
            else:
                st.session_state.current_page = "finetuning"
            st.rerun()
        
        # 5. Results - accessible but show warning if prerequisites not met
        completion_indicator = "✅" if clustering_complete else "⭕"
        button_type = "primary" if st.session_state.current_page == "results" else "secondary"
        
        if st.button(f"{completion_indicator} Results", 
                    type=button_type,
                    use_container_width=True,
                    key="nav_results"):
            if not clustering_complete:
                st.error("⚠️ Complete Clustering first!")
                # Navigate to the incomplete step
                if not data_complete:
                    st.session_state.current_page = "data_loading"
                elif not preprocessing_complete:
                    st.session_state.current_page = "preprocessing"
                else:
                    st.session_state.current_page = "clustering"
            else:
                st.session_state.current_page = "results"
            st.rerun()
        
        st.markdown("---")
        
            # # Progress indicator
            # progress_steps = [data_complete, preprocessing_complete, clustering_complete]
            # completed_steps = sum(progress_steps)
            # progress_percentage = completed_steps / len(progress_steps)
            
            # st.markdown("**Progress:**")
            # st.progress(progress_percentage)
            # st.caption(f"{completed_steps}/{len(progress_steps)} core steps completed")
            
            
            # st.markdown("---")
            
            # from utils.session_state import reset_analysis

            # # Reset button at the bottom  
            # if st.button("🔄 Start New Analysis", 
            #             help="Clear all data and start over",
            #             use_container_width=True,
            #             key="reset_analysis_btn"):   # ✅ renamed key
            #     reset_analysis()   # ✅ call the function directly
            #     st.rerun()
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
    
    # Run change detection and auto-completion
    try:
        from utils.session_state import detect_changes_and_cascade, check_automatic_completion
        detect_changes_and_cascade()
        check_automatic_completion()
    except Exception as e:
        st.warning(f"Session state management warning: {e}")
    
    # Get tab functions
    if 'tab_functions' in st.session_state:
        tab_functions = st.session_state.tab_functions
    else:
        # Fallback import if not in session state
        # Fallback import if not in session state
        try:
            from frontend.frontend_data_loading import tab_data_loading
            from frontend.frontend_preprocessing import tab_preprocessing
            from frontend.frontend_clustering import tab_clustering
            from frontend.frontend_results import tab_results
            from frontend.frontend_finetuning import tab_finetuning  # ← add this

            tab_functions = {
                'data_loading': tab_data_loading,
                'preprocessing': tab_preprocessing,
                'clustering': tab_clustering,
                'results': tab_results,
                'finetuning': tab_finetuning,  # ← and add this
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
            st.markdown("# 🧹 Text Preprocessing")
            st.markdown("Clean and prepare your text data for optimal clustering results.")
            st.markdown("---")
            tab_functions['preprocessing'](st.session_state.get('BACKEND_AVAILABLE', False))
            
        elif current_page == "clustering":
            st.markdown("# 🔍 Clustering Configuration")
            st.markdown("Configure parameters and run the clustering algorithm.")
            st.markdown("---")
            tab_functions['clustering'](st.session_state.get('BACKEND_AVAILABLE', False))
            
        elif current_page == "finetuning":
            st.markdown("# 🧩 Fine-tuning")
            st.markdown("Manually adjust your clustering results using drag-and-drop and AI assistance.")
            st.markdown("---")
            tab_functions['finetuning'](st.session_state.get('BACKEND_AVAILABLE', False))
            
        elif current_page == "results":
            st.markdown("# 📊 Clustering Results")
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
    
    # Create sidebar navigation
    create_sidebar_navigation()
    
    
    # Render main content
    render_main_content()

if __name__ == "__main__":
    main()
