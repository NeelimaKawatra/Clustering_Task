# frontend/frontend_data_loading.py — stable draft→apply + guarded "Ready to Proceed"
import os
import streamlit as st
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from utils.helpers import get_file_from_upload
from utils.reset_manager import reset_from_file_change, reset_from_column_change


# -----------------------------
# tiny helpers
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


def _valid_col(name, df):
    return isinstance(name, str) and (name == "entryID" or name in df.columns)


# -----------------------------
# main tab
# -----------------------------
def tab_data_loading(backend_available: bool):
    """Data Loading tab with explicit Apply for column choices. No mid-edit reruns."""

    # activity (best effort)
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

    # explicit "Start New Analysis"
    if (st.session_state.get("file_uploader_reset")
        and st.session_state.get("file_reset_reason") == "start_new_analysis"):
        st.info("📁 File cleared. Please upload a new file to restart the analysis.")
        st.session_state["file_uploader_reset"] = False
        st.session_state["file_reset_reason"] = None
        st.session_state["data_loading_alerts"] = []

    # persistent alerts
    for kind, text in st.session_state.get("data_loading_alerts", []):
        if kind == "warning":
            st.warning(text)
        elif kind == "success":
            st.success(text)
        else:
            st.info(text)

    # ========= Upload =========
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

    # detect change
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

            # ensure entryID 1..N
            df = df.copy()
            if "entryID" not in df.columns:
                df.insert(0, "entryID", range(1, len(df) + 1))
            else:
                df["entryID"] = range(1, len(df) + 1)

            st.session_state.df = df
            st.session_state.previous_file_key = current_file_key
            st.session_state.tab_data_loading_complete = False
            st.session_state.uploaded_filename = uploaded_file.name

            if "permanent_progress" in st.session_state:
                st.session_state.permanent_progress["data_loading"] = False
                st.session_state.permanent_progress["preprocessing"] = False
                st.session_state.permanent_progress["clustering"] = False

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

    # need data to continue
    if "df" not in st.session_state or st.session_state.df is None:
        return

    # ========= Data present =========
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

    # ========= Column Selection (draft → apply) =========
    st.markdown("---")
    st.subheader("Column Selection")

    # buckets
    config = _ensure("config", {"id_col": None, "entry_col": None})
    temp   = _ensure("temp",   {"id_col": None, "entry_col": None})

    # options from df (not session.available_columns!)
    columns = [c for c in df.columns]  # include entryID; user may pick it as ID  :contentReference[oaicite:3]{index=3}

    # temp defaults (do NOT write to config)
    if temp["id_col"] is None and columns:
        temp["id_col"] = "entryID" if "entryID" in columns else columns[0]
    if temp["entry_col"] is None and columns:
        candidates = [c for c in columns if c != "entryID" and (is_object_dtype(df[c]) or is_string_dtype(df[c]))]
        temp["entry_col"] = candidates[0] if candidates else (columns[0] if columns else None)

    st.caption("Choose your columns, then click **Apply** to commit.")

    # draft widgets
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

    # actions
    a, _ = st.columns([1, 6])
    apply_clicked = a.button("✅ Apply Changes")

    if apply_clicked:
        if temp["id_col"] == temp["entry_col"]:
            st.error("SubjectID and Text entry must be different columns.")
        else:
            # commit draft → config
            config["id_col"] = temp["id_col"]
            config["entry_col"] = temp["entry_col"]

            # mirror to legacy fields expected elsewhere
            st.session_state.subjectID = config["id_col"]
            st.session_state.entry_column = config["entry_col"]

            # one unified downstream reset
            try:
                reset_from_column_change(changed_column="both", show_message=True)
            except Exception:
                pass

            st.success(f"Applied: ID = `{config['id_col']}`, Entry = `{config['entry_col']}`")
            st.rerun()


    applied = (
        f"SubjectID = `{config['id_col']}` | TextEntry = `{config['entry_col']}`"
        if config["id_col"] and config["entry_col"]
        else "not set"
    )
    st.caption(f"**Currently Selected Columns:** {applied}")

    # ========= Validation (read committed only) =========
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

            # metrics
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

            # mark complete
            st.session_state.tab_data_loading_complete = True
            if "permanent_progress" in st.session_state:
                st.session_state.permanent_progress["data_loading"] = True

            # ========= Ready to proceed (guarded) =========
            st.markdown("---")
            st.subheader("Ready to Proceed with:")

            summary_col1, summary_col2 = st.columns(2)
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

            # friendly nudge
            if st.session_state.pop("show_data_loading_success", True):
                st.success("Data Loading Complete!")
                st.info("Proceed to the **Preprocessing** tab to clean and prepare your text entries.")
            else:
                st.success("Data Loading Complete!")
                st.info("Your data configuration is saved. You can proceed to **Preprocessing** or modify settings above to trigger automatic reset.")

        else:
            st.error(validation.get("text_column_message", "Selected text column is not valid."))
            st.session_state.tab_data_loading_complete = False
