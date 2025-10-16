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
        
        # Only update to True, never to False (except on explicit reset)
        st.session_state.permanent_progress['data_loading'] = current_data_complete
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
    
    # Create sidebar navigation
    create_sidebar_navigation()
    
    # Render main content
    render_main_content()

if __name__ == "__main__":
    main()




################################################


# frontend/frontend_data_loading.py — simplified, stable draft→apply selection + "Ready to proceed" summary
import os
import streamlit as st
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from utils.helpers import get_file_from_upload
from utils.reset_manager import reset_from_file_change, reset_from_column_change


# -----------------------------
# tiny helpers (keep it boring)
# -----------------------------
def _ensure(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _safe_index(value, options):
    try:
        return options.index(value) if value in options else 0
    except ValueError:
        return 0


# -----------------------------
# main tab
# -----------------------------
def tab_data_loading(backend_available: bool):
    """Data Loading tab with explicit Apply for column choices. No mid-edit reruns."""

    # Activity log (best effort)
    if backend_available:
        try:
            st.session_state.backend.track_activity(
                st.session_state.session_id, "tab_visit", {"tab_name": "data_loading"}
            )
        except Exception:
            pass

    st.markdown("""
        Welcome to Clustery! Start by uploading your data file with text entries.

        **Supported:** CSV, Excel (.xlsx, .xls)  
        **Note:** An `entryID` column (1..N) is auto-added for stable row tracking.
        """)

    # Handle explicit "Start New Analysis"
    if (st.session_state.get("file_uploader_reset")
        and st.session_state.get("file_reset_reason") == "start_new_analysis"):
        st.info("📁 File cleared. Please upload a new file to restart the analysis.")
        st.session_state["file_uploader_reset"] = False
        st.session_state["file_reset_reason"] = None
        st.session_state["data_loading_alerts"] = []

    # Persistent alerts (carry across reruns)
    for kind, text in st.session_state.get("data_loading_alerts", []):
        if kind == "warning":
            st.warning(text)
        elif kind == "success":
            st.success(text)
        else:
            st.info(text)

    # =========
    # Upload
    # =========
    st.subheader("Upload Your File")
    upload_key = st.session_state.get("file_uploader_key", "data_file_uploader")

    if st.session_state.get("file_uploader_reset") and st.session_state.get("file_reset_reason") == "start_new_analysis":
        st.info("📁 File cleared. Please upload a new file to restart the analysis.")
        st.session_state.file_uploader_reset = False
        st.session_state.file_reset_reason = None

    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls"],
        help="Upload your survey/data file with text entries to cluster",
        key=upload_key,
        label_visibility="collapsed",
    )

    current_file_key = None
    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"

    # Detect change vs previous
    file_changed = False
    if current_file_key and current_file_key != st.session_state.get("previous_file_key"):
        has_work = any([
            st.session_state.get("tab_data_loading_complete"),
            st.session_state.get("tab_preprocessing_complete"),
            st.session_state.get("clustering_results"),
            st.session_state.get("processed_texts"),
            st.session_state.get("finetuning_initialized"),
        ])
        if has_work:
            st.warning("🔄 New file uploaded! This will reset your previous work.")
        # Unified reset for file change
        reset_from_file_change(show_message=has_work)
        file_changed = True
        st.session_state["data_loading_alerts"] = []
    else:
        st.session_state.previous_file_key = current_file_key
        file_changed = bool(current_file_key)

    if uploaded_file is not None and file_changed:
        if not backend_available:
            st.error("Backend services not available. Please check backend installation.")
            return
        try:
            temp_file_path = get_file_from_upload(uploaded_file)
            with st.spinner("Loading and validating file."):
                success, df, message = st.session_state.backend.load_data(
                    temp_file_path, st.session_state.session_id
                )
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if not success:
                st.error(message)
                return

            # Ensure entryID exists and is 1..N contiguous
            df = df.copy()
            if "entryID" not in df.columns:
                df.insert(0, "entryID", range(1, len(df) + 1))
            else:
                df["entryID"] = range(1, len(df) + 1)

            # Persist df + flags
            st.session_state.df = df
            st.session_state.previous_file_key = current_file_key
            st.session_state.tab_data_loading_complete = False
            st.session_state.uploaded_filename = uploaded_file.name

            # Optional global progress flags
            if "permanent_progress" in st.session_state:
                st.session_state.permanent_progress["data_loading"] = False
                st.session_state.permanent_progress["preprocessing"] = False
                st.session_state.permanent_progress["clustering"] = False

            # Build success/warning alerts (truncate note etc.)
            alerts = []
            if "truncated to 300" in (message or "").lower():
                try:
                    trunc = message.split("File loaded successfully", 1)[1].strip().strip("()")
                except Exception:
                    trunc = "File truncated to 300 rows"
                alerts.append(("warning", trunc if trunc.endswith(".") else f"{trunc}."))
                alerts.append(("success", "File uploaded successfully."))
            else:
                alerts.append(("success", message or "File uploaded successfully."))
            st.session_state["data_loading_alerts"] = alerts

            st.rerun()
        except Exception as e:
            st.error(f"Error processing file: {e}")
            with st.expander("Troubleshooting Help"):
                st.markdown("""
- Ensure the file is valid CSV/XLSX.
- Try UTF-8 encoding for CSV.
- Very large files can be slow; start with a subset.
- Avoid exotic characters in headers.
""")
            return

    # No file and no prior df -> stop
    if "df" not in st.session_state or st.session_state.df is None:
        return

    # =========
    # Data present
    # =========
    df = st.session_state.df
    if "entryID" not in df.columns:
        df = df.copy()
        df.insert(0, "entryID", range(1, len(df) + 1))
        st.session_state.df = df

    if df.empty:
        st.error("No data loaded. Please upload a non-empty file.")
        return

    st.markdown("---")
    st.subheader("File Overview")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("File Name", st.session_state.get("uploaded_filename", "Loaded Data"))
    with m2:
        st.metric("Total Rows", len(df))
    with m3:
        st.metric("Total Columns", len(df.columns))
    with m4:
        text_cols = sum(1 for c in df.columns if c != "entryID" and (is_object_dtype(df[c]) or is_string_dtype(df[c])))
        st.metric("Text-like Columns", text_cols)

    with st.expander("Data Preview", expanded=True):
        cols = ["entryID"] + [c for c in df.columns if c != "entryID"]
        st.dataframe(df[cols], width="stretch", hide_index=True)

        # quick per-column stats
        stats_overview = {}
        for col in df.columns:
            total = len(df)
            empty = int(df[col].isna().sum())
            if is_object_dtype(df[col]) or is_string_dtype(df[col]):
                empty += int((df[col] == "").sum())
            non_empty = total - empty
            col_type = "(Auto-generated)" if col == "entryID" else ("Text" if (is_object_dtype(df[col]) or is_string_dtype(df[col])) else "Non-Text")
            stats_overview[col] = {"Total Rows": total, "Empty Rows": empty, "Non-Empty Rows": non_empty, "Column Type": col_type}
        ordered = ["entryID"] + [c for c in stats_overview.keys() if c != "entryID"]
        st.dataframe(pd.DataFrame(stats_overview)[ordered], width="stretch")

    # =========
    # Column Selection (draft → apply)
    # =========
    st.markdown("---")
    st.subheader("Column Selection")

    # Buckets
    config = _ensure("config", {"id_col": None, "entry_col": None})
    temp = _ensure("temp", {"id_col": None, "entry_col": None})

    # Option list: keep it stable and clean (no prompt items)
    columns = [c for c in df.columns]  # include entryID; users may intentionally use it as ID

    # First-run temp defaults (do NOT write to config here)
    if temp["id_col"] is None and columns:
        temp["id_col"] = "entryID" if "entryID" in columns else columns[0]
    if temp["entry_col"] is None and columns:
        candidates = [c for c in columns if c != "entryID" and (is_object_dtype(df[c]) or is_string_dtype(df[c]))]
        temp["entry_col"] = candidates[0] if candidates else (columns[0] if columns else None)

    st.caption("Make your draft choices below, then click **Apply** to commit and reset downstream if needed.")

    # Draft widgets (bound to temp only)
    temp["id_col"] = st.selectbox(
        "SubjectID column",
        options=columns,
        index=_safe_index(temp["id_col"], columns),
        key="temp_id_col_select",
        help="Identifier used to group responses (use `entryID` for per-row IDs).",
    )
    temp["entry_col"] = st.selectbox(
        "Text entry column",
        options=columns,
        index=_safe_index(temp["entry_col"], columns),
        key="temp_entry_col_select",
        help="Column containing the text to cluster.",
    )

    # Actions
    a, b, _ = st.columns([1, 1, 6])
    apply_clicked = a.button("✅ Apply")
    revert_clicked = b.button("↩️ Revert draft")

    if apply_clicked:
        if temp["id_col"] == temp["entry_col"]:
            st.error("SubjectID and Text entry must be different columns.")
        else:
            # Commit draft → config
            config["id_col"] = temp["id_col"]
            config["entry_col"] = temp["entry_col"]

            # Mirror to legacy fields expected elsewhere
            st.session_state.subjectID = config["id_col"]
            st.session_state.entry_column = config["entry_col"]

            # One unified downstream reset
            try:
                reset_from_column_change(changed_column="both", show_message=True)
            except Exception:
                pass

            st.success(f"Applied: ID = `{config['id_col']}`, Entry = `{config['entry_col']}`")
            st.rerun()

    if revert_clicked:
        temp["id_col"] = config["id_col"] if config["id_col"] in columns else ("entryID" if "entryID" in columns else (columns[0] if columns else None))
        temp["entry_col"] = config["entry_col"] if config["entry_col"] in columns else (
            [c for c in columns if c != "entryID" and (is_object_dtype(df[c]) or is_string_dtype(df[c]))][0]
            if any(c != "entryID" and (is_object_dtype(df[c]) or is_string_dtype(df[c])) for c in columns)
            else (columns[0] if columns else None)
        )
        st.info("Draft reverted to last applied configuration.")
        st.rerun()

    applied = (
        f"ID = `{config['id_col']}` | Entry = `{config['entry_col']}`"
        if config["id_col"] and config["entry_col"]
        else "not set"
    )
    st.caption(f"**Currently applied config:** {applied}")

    # =========
    # Validation (read committed only)
    # =========
    if config["id_col"] and config["entry_col"]:
        with st.spinner("Analyzing data quality."):
            try:
                validation = st.session_state.backend.validate_columns(
                    df, config["entry_col"], config["id_col"], st.session_state.session_id
                )
            except Exception as e:
                st.error(f"Validation error: {e}")
                return

        if validation.get("text_column_valid", False):
            st.success(validation.get("text_column_message", "Text column looks good."))
            stats = validation.get("text_quality", {}) or {}

            # normalize numbers
            total = int(stats.get("total_texts", 0))
            empty = int(stats.get("empty_texts", 0))
            avg_len = float(stats.get("avg_length", 0))
            avg_words = float(stats.get("avg_words", 0))
            unique = int(stats.get("unique_texts", 0))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Texts", total)
            c2.metric("Empty", empty)
            c3.metric("Avg. Length", f"{avg_len:.1f}")
            c4.metric("Avg. Words", f"{avg_words:.1f}")
            c5.metric("Unique", unique)

            # mark step complete
            st.session_state.tab_data_loading_complete = True
            if "permanent_progress" in st.session_state:
                st.session_state.permanent_progress["data_loading"] = True

            # =========
            # Ready to proceed section 
            # =========
            st.markdown("---")
            st.subheader("Ready to Proceed with:")

            summary_col1, summary_col2 = st.columns(2)

            # --- Ready to proceed section (defensive guards) ---
            st.markdown("---")
            st.subheader("Ready to Proceed with:")

            summary_col1, summary_col2 = st.columns(2)

            def _valid_col(name, df):
                return isinstance(name, str) and (name == "entryID" or name in df.columns)

            with summary_col1:
                st.markdown("SubjectID Column:")
                sid = st.session_state.get("subjectID")
                if _valid_col(sid, df):
                    if sid == "entryID":
                        st.write("• column name: *entryID*")
                        st.write("• column type: (auto-generated) row numbers")
                    else:
                        st.write(f"• column name: *{sid}*")
                        col_type = 'Numeric' if pd.api.types.is_numeric_dtype(df[sid]) else 'Non-Numeric'
                        st.write(f"• column type: {col_type}")
                else:
                    st.write("• column name: _not set_")

            with summary_col2:
                st.markdown("Entry Column:")
                entry_col = st.session_state.get("entry_column")
                if _valid_col(entry_col, df):
                    st.write(f"• column name: *{entry_col}*")
                    col_type = 'Text' if (pd.api.types.is_object_dtype(df[entry_col]) or pd.api.types.is_string_dtype(df[entry_col])) else 'Non-Text'
                    st.write(f"• column type: {col_type}")
                else:
                    st.write("• column name: _not set_")

            st.markdown("<br>", unsafe_allow_html=True)


            # Track completion and guide next step (once)
            if backend_available:
                try:
                    st.session_state.backend.track_activity(
                        st.session_state.session_id,
                        "data_upload",
                        {
                            "filename": st.session_state.get("uploaded_filename", "loaded_data"),
                            "rows": len(df),
                            "columns": len(df.columns),
                            "entry_column": entry_col,
                            "id_column": sid,
                            "text_quality": stats,
                        },
                    )
                except Exception:
                    pass

            # Friendly nudge (only once)
            if st.session_state.pop("show_data_loading_success", True):
                st.success("Data Loading Complete!")
                st.info("Proceed to the **Preprocessing** tab to clean and prepare your text entries.")
            else:
                st.success("Data Loading Complete!")
                st.info("Your data configuration is saved. You can proceed to **Preprocessing** or modify settings above to trigger automatic reset.")

            # Helpful tip if there were changes
            if st.session_state.get("data_loading_changes_made"):
                st.info("💡 **Tip:** Your changes have been saved and downstream processing has been reset. Navigate to the next tab to continue.")
                del st.session_state["data_loading_changes_made"]

            # Tips section
            st.markdown("---")
            with st.expander("Tips for Better Results", expanded=False):
                st.markdown("""
**For optimal clustering results:**

- Text entry length: Text entries with 20+ words work best  
- Data quality: Remove or fix obviously corrupted text entries  
- Language: Ensure all text entries are in the same language  
- Relevance: All text entries should be about similar topics  
- Volume: 50+ text entries recommended for meaningful clusters  

**What happens next:**
1. **Preprocessing:** Clean and prepare your text entries  
2. **Clustering:** Run advanced algorithms to find patterns  
3. **Results:** Explore and export your findings
""")

        else:
            st.error(validation.get("text_column_message", "Selected text column is not valid."))
            st.session_state.tab_data_loading_complete = False




'''
def get_safe_index(options_list, value, default=0):
    """Safely get index of value in list, return default if not found"""
    try:
        if value is not None and value in options_list:
            return options_list.index(value)
        else:
            return default
    except (ValueError, TypeError):
        return default


def handle_column_selection_change(new_selection, current_selection, selection_type):
    """Handle column selection changes using unified reset system"""
    
    # Define what constitutes a meaningful change
    prompt_values = {
        "ID": ["-- Select a subject ID column--", None, "use entryID (row numbers) as subject IDs"],
        "entry": ["-- Select an entry column --", None]
    }
    
    # Check if this is a meaningful change (simplified logic)
    is_meaningful_change = (
        new_selection != current_selection and
        current_selection not in prompt_values.get(selection_type, []) and
        new_selection not in prompt_values.get(selection_type, [])
    )
    
    # Also trigger if there's any downstream work that would be affected
    has_downstream_work = (
        st.session_state.get('tab_preprocessing_complete') or 
        st.session_state.get('clustering_results') or
        st.session_state.get('processed_texts')
    )
    
    if is_meaningful_change and has_downstream_work:
        # Use unified reset system
        reset_summary = reset_from_column_change(
            changed_column=selection_type,
            show_message=True
        )
        
        # Reset the OTHER column selection
        #if selection_type == "ID":
        #    st.session_state.entry_column = "-- Select an entry column --"
        #elif selection_type == "entry":
        #    st.session_state.subjectID = "-- Select a subject ID column--"
        
        st.success("✅ Reset complete. Please reselect your columns.")
        
        # Force immediate UI refresh
        st.rerun()

def tab_data_loading(backend_available):
    """Tab: Data Loading with unified reset system"""
    
    # Track tab visit
    if backend_available:
        st.session_state.backend.track_activity(st.session_state.session_id, "tab_visit", {"tab_name": "data_loading"})
    
    # Introduction section
    st.markdown("""
    Welcome to Clustery! Start by uploading your data file containing text entries you want to cluster.
    
    **Supported formats:** CSV, Excel (.xlsx, .xls)  
    **Requirements:** Any number of rows with text entries
    
    **Note:** An `entryID` column (row numbers) will be automatically added to your data for tracking purposes.
    """)
    
    # --- Handle explicit 'Start New Analysis' first ---
    if (st.session_state.get("file_uploader_reset")
        and st.session_state.get("file_reset_reason") == "start_new_analysis"):
        st.info("📁 File cleared. Please upload a new file to restart the analysis.")
        st.session_state["file_uploader_reset"] = False
        st.session_state["file_reset_reason"] = None
        st.session_state["data_loading_alerts"] = []  

    # --- Persistent alerts (survive reruns) ---
    alerts = st.session_state.get("data_loading_alerts", [])
    for kind, text in alerts:
        if kind == "warning":
            st.warning(text)
        elif kind == "success":
            st.success(text)
        else:
            st.info(text)

    # Check if data already exists in session state
    data_already_loaded = 'df' in st.session_state and st.session_state.df is not None
    
    # File upload section
    st.subheader("Upload Your File")
    upload_key = st.session_state.get('file_uploader_key', 'data_file_uploader')
    
    # Show message if file uploader was recently reset
    if st.session_state.get('file_uploader_reset') and st.session_state.get('file_reset_reason') == "start_new_analysis":
        st.info("📁 File cleared. Please upload a new file to restart the analysis.")
        st.session_state.file_uploader_reset = False
        st.session_state.file_reset_reason = None
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls"],
        help="Upload your survey data or file containing text entries for clustering",
        key=upload_key,
        label_visibility="collapsed"
    )

    # Check if a new file was uploaded (different from previous)
    current_file_key = None
    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
    
    # Detect file change and reset analysis if needed using unified reset system
    file_changed = False
    if current_file_key and current_file_key != st.session_state.get('previous_file_key'):
        has_work = any([
            st.session_state.get('tab_data_loading_complete'),
            st.session_state.get('tab_preprocessing_complete'),
            st.session_state.get('clustering_results'),
            st.session_state.get('finetuning_initialized'),
        ])
        if has_work:
            st.warning("🔄 New file uploaded! This will reset all your previous work.")
        # Only show the green banner when there was actually something to reset
        reset_from_file_change(show_message=has_work)
        file_changed = True
        st.session_state["data_loading_alerts"] = []
    else:
        st.session_state.previous_file_key = current_file_key
        file_changed = bool(current_file_key)


    # Process file upload if provided and changed
    if uploaded_file is not None and file_changed:
        if not backend_available:
            st.error("Backend services not available. Please check backend installation.")
            return

        try:
            temp_file_path = get_file_from_upload(uploaded_file)

            with st.spinner("Loading and validating file..."):
                success, df, message = st.session_state.backend.load_data(
                    temp_file_path, st.session_state.session_id
                )

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            if not success:
                st.error(f"{message}")
                return

            # Ensure entryID is present and clean (1..N, int)
            df = df.copy()
            if "entryID" not in df.columns:
                df.insert(0, "entryID", range(1, len(df) + 1))
            else:
                # Recreate to guarantee consistent integer IDs starting at 1
                df["entryID"] = range(1, len(df) + 1)

            # Store dataframe and reset progress flags
            st.session_state.df = df
            st.session_state.previous_file_key = current_file_key
            st.session_state.tab_data_loading_complete = False
            st.session_state.uploaded_filename = uploaded_file.name

            # If you track overall progress, also reset here
            if "permanent_progress" in st.session_state:
                st.session_state.permanent_progress["data_loading"] = False
                st.session_state.permanent_progress["preprocessing"] = False
                st.session_state.permanent_progress["clustering"] = False

            # Build persistent alerts
            alerts = []
            if "truncated to 300" in (message or "").lower():
                # Make a yellow warning + green success
                # Extract the truncation phrase if present
                try:
                    # message looks like: "File loaded successfully (truncated to 300 rows from X)"
                    trunc = message.split("File loaded successfully", 1)[1].strip()
                    trunc = trunc.strip("()")
                except Exception:
                    trunc = "File truncated to 300 rows"
                alerts.append(("warning", f"{trunc.capitalize()}." if not trunc.endswith(".") else trunc))
                alerts.append(("success", "File uploaded successfully."))
            else:
                alerts.append(("success", message or "File uploaded successfully."))

            st.session_state["data_loading_alerts"] = alerts
            # Rerun so sidebar status updates; alerts will re-render on next run
            st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            with st.expander("Troubleshooting Help"):
                st.markdown("""
                **Common issues and solutions:**
                - **File format:** Ensure your file is a valid CSV or Excel format
                - **Encoding:** Try saving your CSV with UTF-8 encoding
                - **File size:** Large files (>10MB) may take longer to process
                - **Column names:** Avoid special characters in column headers
                - **Data quality:** Ensure your file isn't corrupted

                **Need help?** Check that your file:
                1. Opens correctly in Excel or a text editor
                2. Has clear column headers
                3. Contains the text entries you want to analyze
                """)
            return
    
    # If no file uploaded and no data exists, return
    if not data_already_loaded and uploaded_file is None:
        return
    
    # From here, we have data loaded - display all configuration sections
    df = st.session_state.df
    
    # Ensure entryID exists even if df came from an older session
    if 'entryID' not in df.columns:
        df = df.copy()
        df.insert(0, 'entryID', range(1, len(df) + 1))
        st.session_state.df = df

    # Ensure dataframe is valid before proceeding
    if df is None or df.empty:
        st.error("No data loaded. Please upload a file first.")
        return
    
    st.markdown("---")

    # File Overview Section
    st.subheader("File Overview")
    
    # Show file metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        file_name = st.session_state.get("uploaded_filename", "Loaded Data")
        st.metric("File Name", file_name)
    with metric_col2:
        st.metric("Total Rows", len(df))
    with metric_col3:
        st.metric("Total Columns", len(df.columns))
    with metric_col4:
        # Count text-like columns excluding entryID
        text_cols = sum(1 for col in df.columns
                        if col != 'entryID' and (is_object_dtype(df[col]) or is_string_dtype(df[col]))
                    )
        st.metric("Total Text Columns", text_cols)
    
    # Data Overview Section
    with st.expander("Data Preview", expanded=True):
        st.markdown("**Your Loaded Data (first 300 rows):**")
        cols = ['entryID'] + [c for c in df.columns if c != 'entryID']
        st.dataframe(df[cols], width="stretch", hide_index=True)

        # Column Statistics
        st.markdown("**Column Statistics:**")
        
        # Create statistics with columns as columns and metrics as rows
        stats_data = {}
        
        # Initialize the stats dictionary with each column
        for col in df.columns:
            total_rows = len(df)
            empty_rows = int(df[col].isna().sum())
            if is_object_dtype(df[col]) or is_string_dtype(df[col]):
                empty_rows += int((df[col] == '').sum())
            non_empty_rows = int(len(df) - empty_rows)
            
            # calculate column type (special handling for entryID column)
            if col == 'entryID':
                col_type = '(Auto-generated)'
            else:
                is_text_like = is_object_dtype(df[col]) or is_string_dtype(df[col])
                col_type = 'Text' if is_text_like else 'Non-Text'

            stats_data[col] = {
                'Total Rows': total_rows,
                'Empty Rows': empty_rows,
                'Non-Empty Rows': non_empty_rows,
                'Column Type': col_type
            }
        
        # display the column stats (entryID first, then the rest)
        ordered_cols = ['entryID'] + [c for c in stats_data.keys() if c != 'entryID']
        stats_df = pd.DataFrame(stats_data)[ordered_cols]

        st.dataframe(stats_df, width="stretch")

    # Column Selection Section
    st.markdown("---")
    st.subheader("Column Selection:")


    

    # ---- 0) tiny helpers ----
    def _ensure(key, default):
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]

    def _safe_index(value, options):
        try:
            return options.index(value) if value in options else 0
        except ValueError:
            return 0

    # ---- 1) initialize state buckets ----
    config = _ensure("config", {"id_col": None, "entry_col": None})
    temp   = _ensure("temp",   {"id_col": None, "entry_col": None})

    # Your real column list; keep order stable across reruns.
    # Avoid including a "prompt option" in options; show prompts via label/help text instead.
    columns = st.session_state.get("available_columns", [])

    # First-run defaults for temp (do NOT auto-write to config here)
    if temp["id_col"] is None and columns:
        temp["id_col"] = columns[0]
    if temp["entry_col"] is None and columns:
        temp["entry_col"] = columns[0]

    # ---- 2) render *draft* selectors bound to temp only ----
    st.write("### Select columns (draft)")
    temp["id_col"] = st.selectbox(
        "SubjectID column",
        options=columns,
        index=_safe_index(temp["id_col"], columns),
        key="temp_id_col_select",
        help="This is the identifier used to group responses."
    )

    temp["entry_col"] = st.selectbox(
        "Text entry column",
        options=columns,
        index=_safe_index(temp["entry_col"], columns),
        key="temp_entry_col_select",
        help="This is the column containing the text content."
    )

    # ---- 3) actions ----
    colA, colB, colC = st.columns([1,1,6])
    apply_clicked = colA.button("✅ Apply")
    reset_clicked = colB.button("↩️ Revert draft")

    if apply_clicked:
        # Validate minimally; no if/else explosion
        if temp["id_col"] == temp["entry_col"]:
            st.error("SubjectID and Text entry must be different columns.")
        else:
            # Single commit point: draft -> committed config
            config["id_col"] = temp["id_col"]
            config["entry_col"] = temp["entry_col"]

            # Perform ONE downstream reset here (whatever your project already does)
            # reset_from_column_change(changed_column="both", show_message=True)
            st.success(f"Applied: ID = `{config['id_col']}`, Entry = `{config['entry_col']}`")
            st.rerun()

    if reset_clicked:
        # Restore draft to the last committed config (or first options if not set)
        temp["id_col"]   = config["id_col"] if config["id_col"] in columns else (columns[0] if columns else None)
        temp["entry_col"] = config["entry_col"] if config["entry_col"] in columns else (columns[0] if columns else None)
        st.info("Draft reverted to last applied configuration.")
        st.rerun()

    # ---- 4) show currently applied config (read-only) ----
    applied = f"ID = `{config['id_col']}` | Entry = `{config['entry_col']}`" if config["id_col"] and config["entry_col"] else "not set"
    st.caption(f"**Currently applied config:** {applied}")
    st.markdown("---")
    st.markdown("---")
    st.markdown("---")
    


    # subject id column selection section
    st.markdown("Step 1: Choose a column for subject identification (Subject IDs)")
    auto_option = "use entryID (row numbers) as subject IDs"
    prompt_option = "-- Select a subject ID column--"

    # Include entryID as a selectable option
    available_columns = [col for col in df.columns if col != 'entryID']
    id_options = [prompt_option, auto_option] + available_columns
    
    # Ensure we have valid options
    if not id_options or len(id_options) == 0:
        st.error("No valid options available for ID selection.")
        return
    
    # get the current subject id selection
    current_id_selection = st.session_state.get('subjectID', prompt_option) or prompt_option

    # Display "entryID" as the auto_option label in the UI
    if current_id_selection == "entryID":
        current_id_selection = auto_option
    if current_id_selection not in id_options:
        current_id_selection = prompt_option

    # get the index of the current subject id selection
    id_column_index = get_safe_index(id_options, current_id_selection, 0)

    # selectbox for the subject id column
    selected_id = st.selectbox(
        label="SubjectID Column:",
        label_visibility="collapsed",
        options=id_options,
        index=id_column_index,
        help="Select a column to track individual responses",
        key="id_selector"
    )

    # Map any auto label to the real column
    if selected_id == auto_option:
        selected_id = "entryID"

     # get the previous subject id selection
    prev = st.session_state.get('subjectID')
    if selected_id != prev:
        # Persist the new selection before any potential rerun so the UI reflects it
        st.session_state.subjectID = selected_id
        handle_column_selection_change(selected_id, prev, "ID")
    else:
        st.session_state.subjectID = selected_id

    # fallback: save entryID (row numbers) as subjectID, if the selected subject id is the prompt option or not set
    #if st.session_state.subjectID == prompt_option or not st.session_state.subjectID:
    #    st.session_state.subjectID = "entryID"

    # Show sample subject IDs (always resolved)
    sid = st.session_state.subjectID
    if sid:
        try:
            sample_ids = df[sid].head(10).tolist()
            if sample_ids:
                formatted = ", ".join([f'"{str(x)}"' for x in sample_ids])
                st.caption(f"**Sample subject IDs: {{{formatted}}}**")
        except (KeyError, AttributeError):
            st.caption("Selected ID column not accessible")

    # entry column selection section
    st.markdown("Step 2: Choose a column for text entry clustering")
    prompt_option_text = "-- Select an entry column --"
    
    # Get entry column suggestions first to filter options
    text_columns = []
    if backend_available:
        try:
            text_columns = st.session_state.backend.get_text_column_suggestions(df, st.session_state.session_id)
            # minimal sanity filter: must exist in df and not be entryID
            text_columns = [c for c in text_columns if c in df.columns and c != 'entryID']
        except AttributeError:
            # Simple fallback - check for text-like columns, excluding entryID
            text_columns = [col for col in df.columns
                            if col != 'entryID' and (is_object_dtype(df[col]) or is_string_dtype(df[col]))]
    
    # Create filtered options - only show text columns plus prompt
    text_options = [prompt_option_text] + text_columns
    
    # Preserve existing entry column selection
    current_text_selection = st.session_state.get('entry_column', prompt_option_text)
    
    # Only reset if no previous selection or selection is invalid
    if current_text_selection is None:
        current_text_selection = prompt_option_text
    
    # Ensure the current selection is valid for current data
    if current_text_selection not in text_options:
        current_text_selection = prompt_option_text
    
    text_column_index = get_safe_index(text_options, current_text_selection, 0)
    
    selected_text_column = st.selectbox(
        label="Entry Column:",
        label_visibility="collapsed",
        options=text_options,
        index=text_column_index,
        help="Select the column with text entries you want to cluster",
        key="text_selector"
    )
    
    # Enhanced change detection for entry column using unified reset system
    previous_entry_column = st.session_state.get('entry_column')
    if selected_text_column != previous_entry_column:
        # Persist the new selection before any potential rerun so the UI reflects it
        st.session_state.entry_column = selected_text_column
        handle_column_selection_change(selected_text_column, previous_entry_column, "entry")
    else:
        st.session_state.entry_column = selected_text_column
    
    # Store user selections for output structure
    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {}

    # Only update user selections if valid columns are selected
    if selected_text_column != prompt_option_text and st.session_state.subjectID:
        st.session_state.user_selections.update({
            'id_column_choice': st.session_state.subjectID,
            'entry_column_choice': selected_text_column,
            'original_columns': [st.session_state.subjectID, selected_text_column]
                if st.session_state.subjectID != 'entryID' else [selected_text_column]
        })

    # Show feedback about entry column detection
    if selected_text_column and selected_text_column != prompt_option_text:
        if text_columns:
            # telling how many entry columns were detected
            pass
        else:
            st.warning("No obvious entry columns detected. Please verify your selection.")
    
    # Show 10 sample text entries with improved formatting and safety
    if selected_text_column and selected_text_column != prompt_option_text:
        try:
            sample_texts = df[selected_text_column].dropna().head(10).tolist()
            if sample_texts:
                formatted_samples = ", ".join([f'"{str(text)[:100] + "..." if len(str(text)) > 100 else str(text)}"' for text in sample_texts])
                st.caption(f"**Sample text entries: {{{formatted_samples}}}**")
        except (KeyError, AttributeError):
            st.caption("Selected entry column not accessible")

    # Validation and Quality Analysis
    if (selected_text_column and selected_text_column != prompt_option_text and 
        selected_text_column in df.columns):
        
        with st.spinner("Analyzing data quality..."):
            validation_result = st.session_state.backend.validate_columns(
                df, 
                selected_text_column, 
                st.session_state.subjectID,
                st.session_state.session_id
            )
        
        # Use original backend validation keys
        if validation_result["text_column_valid"]:
            st.success(f"{validation_result['text_column_message']}")
            stats = validation_result["text_quality"]

            # force numeric types (backend may return strings)
            _total = int(stats.get('total_texts', 0))
            _empty = int(stats.get('empty_texts', 0))
            _avg_len = float(stats.get('avg_length', 0))
            _avg_words = float(stats.get('avg_words', 0))
            _unique = int(stats.get('unique_texts', 0))
            
            st.markdown("**Text Entry Quality Metrics**")
            quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
            
            with quality_col1:
                text_entries = len(df[selected_text_column].dropna())
                st.metric(
                    "Total Text Entries", 
                    text_entries
                )
            with quality_col2:
                st.metric(
                    "Unique Text Entries",
                    _unique
                )
            with quality_col3:
                st.metric(
                    "Avg Text Length", 
                    f"{_avg_len:.0f} chars"
                )
            with quality_col4:
                st.metric(
                    "Avg Text Words", 
                    f"{_avg_words:.1f}"
                )
            
            # Sample text entries in a nice format
            with st.expander("Sample Text Entries Analysis", expanded=False):
                st.markdown("**Representative samples from your text entries:**")
                sample_texts = df[selected_text_column].dropna().head(5)
                
                for i, text in enumerate(sample_texts, 1):
                    text_str = str(text)
                    word_count = len(text_str.split())
                    char_count = len(text_str)
                    
                    with st.container():
                        col_text, col_stats = st.columns([4, 1])
                        
                        with col_text:
                            st.markdown(f"**Sample {i}:**")
                            display_text = text_str[:300] + "..." if len(text_str) > 300 else text_str
                            st.markdown(f"*{display_text}*")
                        
                        with col_stats:
                            st.caption(f"Words: {word_count}")
                            st.caption(f"Chars: {char_count}")
                        
                        st.markdown("---")
            
            # Ready to proceed section
            st.markdown("---")
            st.subheader("Ready to Proceed with:")
            
            # Summary display (showing only the subject id column and entry column basic info)
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("SubjectID Column:")
                sid = st.session_state.subjectID
                if sid:
                    if sid == 'entryID':
                        st.write("• column name: *entryID*")
                        st.write("• column type: (auto-generated) row numbers")
                    else:
                        st.write(f"• column name: *{sid}*")
                        col_type = 'Numeric' if pd.api.types.is_numeric_dtype(df[sid]) else 'Non-Numeric'
                        st.write(f"• column type: {col_type}")
            
            with summary_col2:
                st.markdown("Entry Column:")
                st.write(f"• column name: *{selected_text_column}*")
                col_type = 'Text' if pd.api.types.is_object_dtype(df[selected_text_column]) or pd.api.types.is_string_dtype(df[selected_text_column]) else 'Non-Text'
                st.write(f"• column type: {col_type}")
            
            # Auto-completion with celebration message
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Check if step should auto-complete
            if not st.session_state.get('tab_data_loading_complete', False):
                # Auto-complete when validation passes
                st.session_state.tab_data_loading_complete = True

                # Track completion
                if backend_available:
                    st.session_state.backend.track_activity(st.session_state.session_id, "data_upload", {
                        "filename": "loaded_data",
                        "rows": len(df),
                        "columns": len(df.columns),
                        "entry_column": selected_text_column,
                        "id_column": st.session_state.subjectID,
                        "text_quality": stats
                    })
                
                # Show completion message
                st.success("Data Loading Complete!")
                st.info("Proceed to the **Preprocessing** tab to clean and prepare your text entries.")
                
                # Flag to show celebration message once more after rerun
                st.session_state["show_data_loading_success"] = True

                # Refresh sidebar immediately to show green button
                st.rerun()
                
            else:
                # Already completed - just show status
                st.success("Data Loading Complete!")
                if st.session_state.pop("show_data_loading_success", False):
                    st.info("Proceed to the **Preprocessing** tab to clean and prepare your text entries.")
                else:
                    st.info("Your data configuration is saved. You can proceed to **Preprocessing** or modify settings above to trigger automatic reset.")
            
            # Show feedback if changes were made during this session
            if st.session_state.get('data_loading_changes_made'):
                st.info("💡 **Tip:** Your changes have been saved and downstream processing has been reset. Navigate to the next tab to continue.")
                # Clear the flag
                del st.session_state['data_loading_changes_made']
            
            # Additional tips
            st.markdown("---")
            with st.expander("Tips for Better Results", expanded=False):
                st.markdown("""
                **For optimal clustering results:**
                
                - Text entry length: Text entries with 20+ words work best  
                - Data quality: Remove or fix obviously corrupted text entries  
                - Language: Ensure all text entries are in the same language  
                - Relevance: All text entries should be about similar topics  
                - Volume: 50+ text entries recommended for meaningful clusters  
                
                **What happens next:**
                1. **Preprocessing:** Clean and prepare your text entries  
                2. **Clustering:** Run advanced algorithms to find patterns
                3. **Results:** Explore and export your findings
                """)
        
        else:
            # Use original backend error key
            st.error(f"{validation_result['text_column_message']}")
            # Provide helpful suggestions
            st.markdown("**Suggestions to fix this issue:**")
            st.markdown("""
            - Choose a different column that contains longer text entries  
            - Ensure the column has meaningful sentences, not just single words
            - Check that the column isn't mostly empty or contains mostly numbers/codes
            - Look for columns with survey responses, comments, or descriptions
            """)
            
            # If it was previously complete, mark incomplete and stop here
            if st.session_state.get('tab_data_loading_complete', False):
                st.session_state.tab_data_loading_complete = False
'''