# frontend/frontend_finetuning.py
# Finetuning tab with Drag & Drop + Improved LLM helper with better prompting and quality awareness.

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Backend entrypoint
from backend.finetuning_backend import get_finetuning_backend

# --- Optional drag & drop addon (graceful fallback if missing) ---
try:
    # pip install streamlit-sortables
    from streamlit_sortables import sort_items
except Exception:
    sort_items = None  # type: ignore


# helper function
def expander_open_once(key: str, default: bool = False) -> bool:
    """Return True once if session flag is set; then clear it."""
    return st.session_state.pop(key, default)


# --- zero-width ID codec (hide IDs in labels but keep them recoverable) ---
_ZW0 = "\u200b"  # ZWSP
_ZW1 = "\u200c"  # ZWNJ
_ZWS = "\u2060"  # WORD JOINER (delimiter)
_ZWB = "\u2062"  # INVISIBLE TIMES (start)
_ZWE = "\u2063"  # INVISIBLE SEPARATOR (end)


def _encode_hidden_id(eid: str) -> str:
    bits = "".join(f"{ord(c):08b}" for c in eid)  # 8-bit per char
    payload = "".join(_ZW1 if b == "1" else _ZW0 for b in bits)
    return f"{_ZWB}{payload}{_ZWE}"  # invisible token


def _decode_hidden_id(label: str) -> Optional[str]:
    # Extract the invisible token and turn it back into the original string
    if _ZWB in label and _ZWE in label:
        enc = label.split(_ZWB, 1)[1].split(_ZWE, 1)[0]
        bits = "".join("1" if ch == _ZW1 else ("0" if ch == _ZW0 else "") for ch in enc)
        if bits and len(bits) % 8 == 0:
            chars = [chr(int(bits[i : i + 8], 2)) for i in range(0, len(bits), 8)]
            return "".join(chars)
    return None


def _force_dnd_refresh() -> None:
    """Force the drag-and-drop board to refresh by updating its key (SMART VERSION)."""
    current_token = st.session_state.get("finetuning_refresh_token", 0)
    st.session_state["finetuning_refresh_token"] = current_token + 1

    # Only clear drag-and-drop specific state (not all sortable keys)
    widget_prefix = "dnd_board_"
    keys_to_remove = [k for k in list(st.session_state.keys()) if k.startswith(widget_prefix)]
    for key in keys_to_remove:
        try:
            del st.session_state[key]
        except Exception:
            pass

    # Clear Streamlit's caches
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass


# =============================================================================
# MAIN TAB
# =============================================================================


def tab_finetuning(backend_available: bool):
    """Human-in-the-Loop Fine-tuning tied to backend API"""

    # Track tab visit
    if backend_available and hasattr(st.session_state, "backend") and st.session_state.backend:
        st.session_state.backend.track_activity(
            st.session_state.session_id, "tab_visit", {"tab_name": "finetuning"}
        )

    # Mark Fine-tuning as visited
    st.session_state["finetuning_ever_visited"] = True

    # Prerequisite check
    if not st.session_state.get("clustering_results") or not st.session_state.clustering_results.get(
        "success", False
    ):
        st.error("Please complete Clustering first.")
        st.info("Go to the Clustering tab and run the clustering analysis to see results here.")
        return

    # Initialize fine-tuning backend state from clustering outcome
    if not _initialize_backend():
        st.error("Failed to initialize fine-tuning backend.")
        return

    backend = get_finetuning_backend()

    # view, rename, delete + create, merge
    show_cluster_management_interface(backend)

    # search, inspect, move a single entry
    show_entry_management_interface(backend)

    # drag & drop with filtering
    show_drag_drop_board(backend)

    # AI assist with structured operations and Cursor-like review
    show_ai_assist_interface(backend)


# =============================================================================
# HELPERS
# =============================================================================


def _initialize_backend() -> bool:
    """Initialize fine-tuning backend from clustering results in session."""
    if st.session_state.get("finetuning_initialized"):
        return True

    backend = get_finetuning_backend()

    clustering_results = st.session_state.clustering_results
    df = st.session_state.df

    subject_id_column = st.session_state.get("subjectID")
    # Fallback: check user_selections
    if not subject_id_column:
        user_selections = st.session_state.get("user_selections", {})
        subject_id_column = user_selections.get("id_column_choice")
    # Final fallback: entryID if present in df
    if not subject_id_column and "entryID" in df.columns:
        subject_id_column = "entryID"

    text_column = st.session_state.get("entry_column")

    success = backend.initialize_from_clustering_results(
        clustering_results, df, text_column, subject_id_column
    )

    if success:
        st.session_state.finetuning_initialized = True
        save_finetuning_results_to_session(backend)
        return True

    return False


def show_cluster_management_interface(backend):
    # Flash messages (shown at top)
    if "finetuning_success_message" in st.session_state:
        st.success(st.session_state.finetuning_success_message)
        del st.session_state.finetuning_success_message
    if "finetuning_error_message" in st.session_state:
        st.error(st.session_state.finetuning_error_message)
        del st.session_state.finetuning_error_message

    # ---- Summary
    with st.expander("Cluster Summary", expanded=True):
        all_clusters = backend.getAllClusters()
        all_entries = backend.getAllEntries()
        modification_summary = backend.getModificationSummary()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Clusters", len(all_clusters))
        with col2:
            st.metric("Total Text Entries", len(all_entries))
        with col3:
            st.metric("Manual Clusters", modification_summary.get("manual_clusters_created", 0))
        with col4:
            st.metric(
                "Modified Entries",
                f"{modification_summary.get('modification_percentage', 0):.1f}%",
            )

    # ---- View, Rename, Delete clusters
    if "ft_manage_expanded" not in st.session_state:
        st.session_state.ft_manage_expanded = False

    with st.expander(
        "💡 You can view, rename, check confidence, and delete clusters:",
        expanded=st.session_state.ft_manage_expanded,
    ):
        all_clusters = backend.getAllClusters()

        for i, (cid, cdata) in enumerate(all_clusters.items()):
            key_prefix = f"{cid}_{i}"
            cluster_name = cdata["cluster_name"]

            # Track each child expander state
            child_exp_key = f"ft_child_{key_prefix}"
            if child_exp_key not in st.session_state:
                st.session_state[child_exp_key] = False

            with st.expander(
                f"🗂️ {cluster_name} ({len(cdata['entry_ids'])} entries)",
                expanded=st.session_state[child_exp_key],
            ):
                col1, col2, col3 = st.columns([2, 1, 1])

                # Rename - Enter to submit
                with col1:
                    prev_name_key = f"prev_name_{key_prefix}"
                    if prev_name_key not in st.session_state:
                        st.session_state[prev_name_key] = cluster_name

                    def handle_rename():
                        new_name = st.session_state[f"name_{key_prefix}"]
                        prev_name = st.session_state[prev_name_key]

                        if new_name.strip() != prev_name and new_name.strip() != "":
                            ok, msg = backend.changeClusterName(cid, new_name.strip())
                            if ok:
                                st.session_state.finetuning_success_message = f"✏️ {msg}"
                                st.session_state[prev_name_key] = new_name.strip()
                                st.session_state.ft_manage_expanded = True
                                st.session_state[child_exp_key] = True
                                save_finetuning_results_to_session(backend)
                                _force_dnd_refresh()
                            else:
                                st.session_state.finetuning_error_message = msg
                                st.session_state.ft_manage_expanded = True
                                st.session_state[child_exp_key] = True

                    st.text_input(
                        "Cluster name (press Enter to rename)",
                        value=cluster_name,
                        key=f"name_{key_prefix}",
                        on_change=handle_rename,
                    )

                # Stats
                with col2:
                    stats = backend.getClusterStatistics(cid)
                    if stats:
                        st.metric("Avg Confidence", f"{stats['avg_probability']:.2f}")

                # Delete - button only
                with col3:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Delete", key=f"delete_{key_prefix}"):
                        ok, msg = backend.deleteCluster(cid)
                        if ok:
                            st.session_state.finetuning_success_message = f"🗑️ {msg}"
                            if prev_name_key in st.session_state:
                                del st.session_state[prev_name_key]
                            st.session_state.ft_manage_expanded = True
                            save_finetuning_results_to_session(backend)
                            _force_dnd_refresh()
                            st.rerun()
                        else:
                            st.session_state.finetuning_error_message = msg
                            st.session_state.ft_manage_expanded = True
                            st.session_state[child_exp_key] = True
                            st.rerun()

    # ---- Create + Merge
    if "ft_create_merge_expanded" not in st.session_state:
        st.session_state.ft_create_merge_expanded = False

    with st.expander(
        "💡 You can create new clusters or merge existing clusters:",
        expanded=st.session_state.ft_create_merge_expanded,
    ):
        current_clusters = backend.getAllClusters()
        col1, col2 = st.columns(2)

        # Create - Enter to submit
        with col1:
            st.markdown("**Create New Cluster**")

            def handle_create():
                new_cluster_name = st.session_state.get("new_cluster_name_input", "").strip()
                if new_cluster_name:
                    ok, result = backend.createNewCluster(new_cluster_name)
                    if ok:
                        st.session_state.finetuning_success_message = (
                            f"✅ Created cluster: '{result}'"
                        )
                        st.session_state.new_cluster_name_input = ""
                        st.session_state.ft_create_merge_expanded = True
                        save_finetuning_results_to_session(backend)
                        _force_dnd_refresh()
                    else:
                        st.session_state.finetuning_error_message = result
                        st.session_state.ft_create_merge_expanded = True

            st.text_input(
                "New cluster name (press Enter to create)",
                value="",
                placeholder="Type here...",
                key="new_cluster_name_input",
                on_change=handle_create,
            )

        # Merge - button
        with col2:
            st.markdown("**Merge Clusters**")
            cluster_ids = list(current_clusters.keys())
            if len(cluster_ids) >= 2:
                options = [
                    f"{current_clusters[c]['cluster_name']} ({len(current_clusters[c]['entry_ids'])} entries)"
                    for c in cluster_ids
                ]

                idx1 = st.selectbox(
                    "First cluster",
                    range(len(options)),
                    key="merge_cluster1",
                    format_func=lambda x: options[x],
                )
                idx2 = st.selectbox(
                    "Second cluster",
                    range(len(options)),
                    key="merge_cluster2",
                    format_func=lambda x: options[x],
                )

                merge_name = st.text_input(
                    "Merged name (optional)",
                    value="",
                    key="merge_name_input",
                    placeholder="Leave blank for auto",
                )

                if st.button("🔗 Merge Clusters", key="merge_clusters_btn"):
                    if idx1 != idx2:
                        c1, c2 = cluster_ids[idx1], cluster_ids[idx2]
                        ok, result = backend.mergeClusters(c1, c2, merge_name.strip() or None)
                        if ok:
                            st.session_state.finetuning_success_message = f"🔄 Merged into: '{result}'"
                            st.session_state.merge_name_input = ""
                            st.session_state.ft_create_merge_expanded = True
                            save_finetuning_results_to_session(backend)
                            _force_dnd_refresh()
                            st.rerun()
                        else:
                            st.session_state.finetuning_error_message = result
                            st.session_state.ft_create_merge_expanded = True
                            st.rerun()
                    else:
                        st.warning("Please select two different clusters")
            else:
                st.info("Need at least 2 clusters to merge")


def show_drag_drop_board(backend):
    """
    Drag & Drop board for cluster reassignment.
    Adds keyword + confidence filters so only matching entries are shown.
    Sticky while interacting; collapses only after a successful Apply.

    First-open behavior: reset filters to show ALL entries.
    """
    clusters: Dict[str, dict] = backend.getAllClusters()
    if not clusters:
        st.info("No clusters to display.")
        return

    # Initialize expander state - open by default on first visit
    if not st.session_state.get("dnd_filters_initialized", False):
        st.session_state["exp_drag_open"] = True
    else:
        st.session_state.setdefault("exp_drag_open", False)

    def _keep_drag_open():
        st.session_state["exp_drag_open"] = True

    with st.expander(
        "💡 You can drag entries across clusters. Type to filter. Click **Apply changes** to commit.",
        expanded=st.session_state["exp_drag_open"],
    ):
        # ---------- FIRST-OPEN DEFAULTS (show everything) ----------
        if not st.session_state.get("dnd_filters_initialized", False):
            st.session_state["dnd_filter_text"] = ""
            st.session_state["dnd_conf_level"] = "All"
            st.session_state["dnd_filters_initialized"] = True

        # --- Filters
        colf1, colf2 = st.columns([2, 1])

        with colf1:
            filter_text = st.text_input(
                "Search by keyword (case-insensitive)",
                placeholder='e.g., "ai"',
                key="dnd_filter_text",
                on_change=_keep_drag_open,
            )

        with colf2:
            conf_choice = st.selectbox(
                "Confidence Level",
                ["All", "High (≥ 0.7)", "Medium (0.3–0.7)", "Low (< 0.3)"],
                key="dnd_conf_level",
                on_change=_keep_drag_open,
            )
            conf_map = {
                "All": None,
                "High (≥ 0.7)": "high",
                "Medium (0.3–0.7)": "medium",
                "Low (< 0.3)": "low",
            }
            conf_level = conf_map.get(conf_choice)

        # --- Build containers from filtered/unfiltered view
        clusters = backend.getAllClusters()

        try:
            if filter_text or conf_level:
                filtered_map = backend.getEntriesByClusterFiltered(filter_text, conf_level)
            else:
                filtered_map = backend.getEntriesByCluster()
        except AttributeError:
            if filter_text:
                filtered_map = backend.getEntriesByCluster(filter_text)
            else:
                filtered_map = {cid: cdata.get("entry_ids", []) for cid, cdata in clusters.items()}

        old_cluster_of: Dict[str, str] = {}
        containers: List[dict] = []
        orig_container_ids: List[str] = []

        total_shown = 0
        for cluster_id, cdata in clusters.items():
            total_eids = cdata.get("entry_ids", [])
            shown_eids = filtered_map.get(cluster_id, [])

            items: List[str] = []
            for eid in shown_eids:
                entry = backend.getEntry(eid)
                if not entry:
                    continue
                text = (entry.get("entry_text") or "").replace("\n", " ").strip()
                disp = text[:120] + ("…" if len(text) > 120 else "")
                items.append(f"{disp}{_encode_hidden_id(str(eid))}")

                old_cluster_of[str(eid)] = cluster_id

            header = f"{cdata.get('cluster_name', cluster_id)} ({len(shown_eids)} / {len(total_eids)} shown)"
            containers.append({"header": header, "items": items})
            orig_container_ids.append(cluster_id)
            total_shown += len(items)

        if filter_text or conf_level:
            if conf_level:
                st.caption(
                    f'Filter "{filter_text or "*"}" + confidence={conf_level} → showing {total_shown} item(s).'
                )
            else:
                st.caption(f'Filter "{filter_text}" → showing {total_shown} item(s).')

        # --- Render DnD (guarded)
        result = None
        filter_hash = hash((filter_text or "", conf_level or "all"))
        _sortable_key = f"dnd_board_{st.session_state.get('finetuning_refresh_token', 0)}_{filter_hash}"

        if sort_items is not None:
            try:
                css = ".sortable-component{min-height:240px}.sortable-container{min-width:260px}"
                result = sort_items(
                    containers,
                    multi_containers=True,
                    direction="horizontal",
                    custom_style=css,
                    key=_sortable_key,
                )
            except Exception:
                result = None

        if result is None:
            st.warning(
                "Drag-and-drop widget unavailable (install/enable `streamlit-sortables`). "
                "Showing read-only lists."
            )
            for c in containers:
                st.markdown(f"**{c['header']}**")
                for it in c["items"]:
                    st.write(f"- {it}")
            st.caption(
                "No pending changes." if not (filter_text or conf_level) else "No pending changes among the filtered items."
            )
            return

        # --- Compute pending moves (visible items only)
        def _eid_from_item(item: Any) -> Optional[str]:
            if not isinstance(item, str):
                return None
            hid = _decode_hidden_id(item)
            if hid:
                return hid
            if ":" in item:
                return item.split(":", 1)[0].strip()
            return None

        pending_moves: List[Tuple[str, str]] = []
        for i, container in enumerate(result or containers):
            new_cid = orig_container_ids[i] if i < len(orig_container_ids) else None
            if not new_cid:
                continue
            for item in container.get("items", []):
                eid = _eid_from_item(item)
                if eid and old_cluster_of.get(str(eid)) != new_cid:
                    pending_moves.append((str(eid), new_cid))

        if pending_moves and st.session_state.get("debug_dnd", False):
            st.write(f"Debug: {len(pending_moves)} pending moves detected")
            for eid, new_cid in pending_moves[:3]:
                st.write(f"  - {eid}: {old_cluster_of.get(eid, 'unknown')} -> {new_cid}")

        if pending_moves:
            st.session_state["exp_drag_open"] = True
            if st.button("Apply changes", use_container_width=True, key="apply_dnd_changes_btn"):
                ok = 0
                failed = []
                for eid, target_cid in pending_moves:
                    success, msg = backend.moveEntry(eid, target_cid)
                    if success:
                        ok += 1
                    else:
                        failed.append(f"{eid}: {msg}")

                if ok > 0:
                    st.session_state.finetuning_success_message = f"✅ Applied {ok} change(s)."
                if failed:
                    st.session_state.finetuning_error_message = f"Failed: {', '.join(failed[:3])}"

                st.session_state["exp_drag_open"] = True
                save_finetuning_results_to_session(backend)
                _force_dnd_refresh()
                st.rerun()
        else:
            st.caption(
                "No pending changes."
                if not (filter_text or conf_level)
                else "No pending changes among the filtered items."
            )


def show_entry_management_interface(backend):
    """Search, inspect, and move a single entry with form controls."""
    st.session_state.setdefault("exp_entry_open", False)

    with st.expander(
        "💡 You can search, inspect, and move a single entry:",
        expanded=st.session_state["exp_entry_open"],
    ):
        all_entries = backend.getAllEntries()
        all_clusters = backend.getAllClusters()

        change_count = backend.change_counter if hasattr(backend, "change_counter") else 0

        if not all_entries:
            st.info("No entries available.")
            return

        entry_ids = list(all_entries.keys())

        def _keep_entry_open():
            st.session_state["exp_entry_open"] = True

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Find Entry**")
            search_text = st.text_input(
                "Search in text entries",
                placeholder="Type to search...",
                key="entry_search_text",
                on_change=_keep_entry_open,
            )

            if search_text:
                matching_entries = [
                    eid
                    for eid, ed in all_entries.items()
                    if search_text.lower() in (ed.get("entry_text") or "").lower()
                ]
                if matching_entries:
                    selected_entry = st.selectbox(
                        "Matching entries",
                        matching_entries,
                        key=f"matching_entries_box_{change_count}",
                        format_func=lambda eid: f"{all_entries[eid]['subjectID']}: {all_entries[eid]['entry_text']}",
                        on_change=_keep_entry_open,
                    )
                else:
                    st.info("No entries match your search.")
                    selected_entry = None
            else:
                selected_entry = st.selectbox(
                    "Select entry",
                    entry_ids,
                    key=f"all_entries_box_{change_count}",
                    format_func=lambda eid: f"{all_entries[eid]['subjectID']}: {all_entries[eid]['entry_text']}",
                    on_change=_keep_entry_open,
                )

        with col2:
            if "selected_entry" in locals() and selected_entry:
                st.markdown("**Entry Details**")
                entry_data = backend.getEntry(selected_entry)
                if entry_data:
                    st.text(f"Subject ID: {entry_data['subjectID']}")
                    st.text_area(
                        "Complete Text Entry",
                        value=entry_data["entry_text"],
                        height=120,
                        disabled=True,
                        key=f"text_{selected_entry}",
                    )

                    cluster_id = entry_data.get("clusterID", "Unassigned")
                    cluster_info = all_clusters.get(cluster_id)
                    if cluster_info:
                        st.text(
                            f"Current Cluster: {cluster_info['cluster_name']} ({len(cluster_info['entry_ids'])} entries)"
                        )
                    else:
                        st.text(f"Current Cluster: {cluster_id}")

                    st.text(f"Confidence: {entry_data.get('probability', 0):.2f}")

                    cluster_options = list(all_clusters.keys())
                    current_cluster = entry_data.get("clusterID")
                    current_index = (
                        cluster_options.index(current_cluster)
                        if current_cluster in cluster_options
                        else 0
                    )

                    target_cluster = st.selectbox(
                        "Move Selected Entry to Cluster",
                        cluster_options,
                        index=current_index,
                        format_func=lambda x: f"{all_clusters[x]['cluster_name']} ({len(all_clusters[x]['entry_ids'])} entries)",
                        key=f"move_{selected_entry}_{change_count}",
                        on_change=_keep_entry_open,
                    )

                    if target_cluster != current_cluster:
                        if st.button(
                            f"Move to {all_clusters[target_cluster]['cluster_name']}",
                            key=f"move_btn_{selected_entry}_{change_count}",
                        ):
                            success, message = backend.moveEntry(selected_entry, target_cluster)
                            if success:
                                st.session_state.finetuning_success_message = f"📦 {message}"
                                st.session_state["exp_entry_open"] = False
                                save_finetuning_results_to_session(backend)
                                _force_dnd_refresh()
                                st.rerun()
                            else:
                                st.session_state.finetuning_error_message = message
                                st.session_state["exp_entry_open"] = True
                                st.rerun()


def show_ai_assist_interface(backend):
    """AI assist with improved prompting, quality awareness, and Cursor-like review UX"""

    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Fine-tuning Assistant")

    with st.expander("ℹ️ What can AI Assist do?", expanded=False):
        st.markdown(
            """
**AI can help you:**
- 🏷️ **Name clusters** based on their content
- 🔄 **Move entries** to better-fitting clusters
- 🔧 **Merge/split** clusters for better organization

**Important Notes:**
- AI only suggests changes that would **clearly improve** your clustering
- If your clustering is already good, AI may return **no suggestions** (this is good!)
- Each suggestion includes a **confidence score** - only high-confidence suggestions are shown
- You can review and accept/reject each suggestion individually

**Cost:** Mock (free) • GPT-4o-mini ($0.01-0.05) • GPT-4o (10x more)

💡 Configure LLM Settings first, then generate suggestions below.
"""
        )

    from frontend.frontend_llm_settings import check_llm_configuration

    llm_status = check_llm_configuration()

    if not llm_status["configured"]:
        st.warning(f"⚠️ **{llm_status['message']}**")
        st.markdown(
            """
**AI Assist requires LLM configuration.**

Go to **LLM Settings** in the sidebar to:
- Choose your LLM provider (Mock for testing, OpenAI for production)
- Select model and temperature
- Configure API credentials
"""
        )

        col1, _, _ = st.columns([1, 1, 2])
        with col1:
            if st.button("⚙️ Open LLM Settings", use_container_width=True):
                st.session_state.current_page = "llm_settings"
                st.rerun()
        return

    llm_config = st.session_state.llm_config
    provider = llm_config["provider"]
    model = llm_config["model"]
    temperature = llm_config["temperature"]

    st.success(f"✅ **AI Assistant Active**: {provider.upper()} ({model}) • Temperature: {temperature:.2f}")

    active_sessions = [
        key.replace("suggestion_session_", "")
        for key in st.session_state.keys()
        if key.startswith("suggestion_session_")
        and st.session_state[key].get("status") == "reviewing"
    ]

    if active_sessions:
        session_id = active_sessions[0]
        result = show_suggestion_review_panel(session_id, backend)

        if result["action"] == "accept_all" and result["applied_count"] > 0:
            st.success(f"✅ Applied {result['applied_count']} suggestion(s)!")
            st.balloons()
            time.sleep(1)
            st.rerun()
        elif result["action"] == "partial" and result["applied_count"] > 0:
            st.success("✅ Applied suggestion!")
            save_finetuning_results_to_session(backend)
        elif result["action"] == "reject":
            st.info("All suggestions rejected.")
            time.sleep(0.5)
            st.rerun()
        elif result["action"] == "regenerate":
            st.info("Ready to regenerate suggestions...")
            time.sleep(0.5)
            st.rerun()

        return

    with st.expander("🤖 Generate AI Suggestions", expanded=True):
        st.markdown("**Choose AI Operation**")
        operation = st.selectbox(
            "What would you like AI to help with?",
            ["Suggest Cluster Names", "Suggest Entry Moves", "Suggest Merges/Splits"],
            key="ft_ai_operation",
        )

        llm_wrapper = get_llm_wrapper()
        llm_wrapper.backend = backend

        if provider != "mock":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OpenAI API key not found in environment.")
                if st.button("⚙️ Go to LLM Settings"):
                    st.session_state.current_page = "llm_settings"
                    st.rerun()
                return

            success = llm_wrapper.initLLM(provider, {"model": model, "api_key": api_key})
            if not success:
                st.error("❌ Failed to initialize LLM. Check your configuration in LLM Settings.")
                if st.button("⚙️ Go to LLM Settings"):
                    st.session_state.current_page = "llm_settings"
                    st.rerun()
                return
        else:
            llm_wrapper.initLLM("mock", {"model": "mock"})

        params: Dict[str, Any] = {}
        if operation == "Suggest Entry Moves":
            st.markdown("**Configure Entry Analysis**")

            col1, col2 = st.columns(2)
            with col1:
                filter_cluster = st.selectbox(
                    "Analyze entries from",
                    ["All clusters"] + list(backend.getAllClusters().keys()),
                    key="ft_ai_move_cluster",
                    help="Which cluster's entries to analyze for potential moves",
                )
                params["filter_cluster"] = filter_cluster

            with col2:
                input_confidence_filter = st.slider(
                    "Analyze entries with confidence below",
                    0.0,
                    1.0,
                    0.5,
                    0.1,
                    help="Only analyze uncertain entries (lower confidence = more uncertain)",
                    key="ft_ai_input_filter",
                )
                params["input_confidence_filter"] = input_confidence_filter
                st.caption(f"✓ Will analyze entries with confidence < {input_confidence_filter:.1f}")

            st.markdown("**Suggestion Quality Control**")
            suggestion_quality_threshold = st.slider(
                "Minimum suggestion quality",
                0.0,
                1.0,
                0.6,
                0.1,
                help="LLM's confidence that a move would help (higher = stricter)",
                key="ft_ai_quality_threshold",
            )
            params["suggestion_quality_threshold"] = suggestion_quality_threshold
            st.caption(f"✓ Only show suggestions with quality ≥ {suggestion_quality_threshold:.1f}")

        if st.button("🔮 Generate Suggestions", type="primary", use_container_width=True):
            operation_map = {
                "Suggest Cluster Names": "cluster_names",
                "Suggest Entry Moves": "entry_moves",
                "Suggest Merges/Splits": "cluster_operations",
            }
            operation_type = operation_map[operation]
            session_id = llm_wrapper.create_suggestion_session(operation_type, params)

            with st.spinner("🤖 Analyzing your data and generating suggestions..."):
                context = llm_wrapper.build_context(include_texts=True, max_samples=5)
                success = llm_wrapper.generate_suggestions_async(session_id, context)

                if success:
                    session_data = st.session_state.get(f"suggestion_session_{session_id}", {})
                    suggestions = session_data.get("suggestions")
                    suggestion_count = _count_suggestions(suggestions, operation_type)

                    if suggestion_count == 0:
                        st.info("✅ Your clustering looks good! No improvements suggested.")
                        st.caption("This means your current cluster assignments are already high-quality.")
                        _cleanup_suggestion_session(session_id)
                    else:
                        st.success(f"✅ {suggestion_count} quality suggestions generated! Review them below.")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.error("❌ Failed to generate suggestions. Try again.")
                    _cleanup_suggestion_session(session_id)


# ✅ IMPORTANT: this MUST be at module level (indentation fixed)
def _cid_to_int(cid) -> int:
    if isinstance(cid, int):
        return cid
    s = str(cid)
    if s == "outliers":
        return -1
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else -1


def build_finetuning_results_snapshot(backend) -> dict:
    """
    EXACT mirror of clustering results dict.
    Writes SSOT labels into metadata['topic_keywords'] using current cluster_name(s).
    """
    entries = backend.getAllEntries()
    clusters = backend.getAllClusters()

    sorted_eids = sorted(entries.keys(), key=lambda x: str(x))
    texts, topics, probabilities = [], [], []
    for eid in sorted_eids:
        e = entries[eid]
        txt = e.get("entry_text") or e.get("original_text") or ""
        cid = _cid_to_int(e.get("clusterID", -1))
        p = float(e.get("probability", 0.0) or 0.0)
        texts.append(txt)
        topics.append(cid)
        probabilities.append(p)

    total_texts = len(texts)
    outliers = sum(1 for t in topics if t == -1)
    clustered = total_texts - outliers
    n_clusters = len({t for t in topics if t != -1})
    success_rate = (clustered / total_texts * 100.0) if total_texts else 0.0

    high = sum(1 for p in probabilities if p >= 0.7)
    med = sum(1 for p in probabilities if 0.3 <= p < 0.7)
    low = sum(1 for p in probabilities if p < 0.3)
    avg_conf = (sum(probabilities) / len(probabilities)) if probabilities else 0.0

    topic_keywords = {}
    for cid_str, c in clusters.items():
        k = _cid_to_int(cid_str)
        nm = (c.get("cluster_name") or ("Outliers" if k == -1 else str(k))).strip()
        topic_keywords[k] = [nm] if nm else [("Outliers" if k == -1 else str(k))]

    params_used = (st.session_state.get("clustering_results") or {}).get("parameters_used", {})
    prev_meta = (st.session_state.get("clustering_results") or {}).get("metadata", {})
    n_features = int(prev_meta.get("n_features", 0))
    n_components = int(prev_meta.get("n_components", 0))

    original_performance = (st.session_state.get("clustering_results") or {}).get(
        "performance",
        {"total_time": 0.0, "setup_time": 0.0, "clustering_time": 0.0},
    )

    return {
        "success": True,
        "topics": topics,
        "probabilities": probabilities,
        "predictions": topics,
        "texts": texts,
        "metadata": {
            "model_type": "Manual",
            "n_features": n_features,
            "n_components": n_components,
            "topic_keywords": topic_keywords,
        },
        "statistics": {
            "n_clusters": n_clusters,
            "outliers": outliers,
            "clustered": clustered,
            "success_rate": success_rate,
            "total_texts": total_texts,
        },
        "confidence_analysis": {
            "high_confidence": high,
            "medium_confidence": med,
            "low_confidence": low,
            "avg_confidence": avg_conf,
        },
        "performance": original_performance,
        "parameters_used": params_used,
    }


def save_finetuning_results_to_session(backend) -> None:
    """Save fine-tuning results to session and mark as changed."""
    st.session_state.finetuning_results = build_finetuning_results_snapshot(backend)
    st.session_state.tab_results_complete = False
    st.session_state.permanent_progress["results"] = False


# =============================================================================
# NEW: Cursor-like Review UX Components
# =============================================================================


def show_suggestion_review_panel(session_id: str, backend) -> Dict[str, Any]:
    session_data = st.session_state.get(f"suggestion_session_{session_id}")
    if not session_data or session_data["status"] != "reviewing":
        return {"action": "none", "applied_count": 0}

    operation_type = session_data["operation_type"]
    suggestions = session_data["suggestions"]

    st.markdown("---")
    st.markdown("### 🤖 AI Suggestions Ready for Review")

    total_suggestions = _count_suggestions(suggestions, operation_type)
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        st.markdown(f"**{total_suggestions} suggestions** generated based on your clustering data")

    with col2:
        if st.button("✅ Accept All", key=f"accept_all_{session_id}", type="primary"):
            result = _apply_all_suggestions(session_id, suggestions, operation_type, backend)
            _cleanup_suggestion_session(session_id)
            return result

    with col3:
        if st.button("❌ Reject All", key=f"reject_all_{session_id}"):
            _cleanup_suggestion_session(session_id)
            return {"action": "reject", "applied_count": 0}

    with col4:
        if st.button("🔄 Regenerate", key=f"regenerate_{session_id}"):
            _cleanup_suggestion_session(session_id)
            return {"action": "regenerate", "applied_count": 0}

    st.markdown("---")

    if operation_type == "cluster_names":
        return _review_cluster_name_suggestions(session_id, suggestions, backend)
    if operation_type == "entry_moves":
        return _review_entry_move_suggestions(session_id, suggestions, backend)
    if operation_type == "cluster_operations":
        return _review_cluster_operation_suggestions(session_id, suggestions, backend)

    return {"action": "none", "applied_count": 0}


def _cleanup_suggestion_session(session_id: str):
    session_key = f"suggestion_session_{session_id}"
    if session_key in st.session_state:
        del st.session_state[session_key]

    selection_keys = [
        f"name_selections_{session_id}",
        f"move_selections_{session_id}",
        f"operation_selections_{session_id}",
    ]
    for key in selection_keys:
        if key in st.session_state:
            del st.session_state[key]

    keys_to_remove = [k for k in list(st.session_state.keys()) if session_id in str(k)]
    for key in keys_to_remove:
        try:
            del st.session_state[key]
        except Exception:
            pass


def _count_suggestions(suggestions: Any, operation_type: str) -> int:
    if operation_type == "cluster_names" and isinstance(suggestions, dict):
        return len(suggestions)
    if operation_type == "entry_moves" and isinstance(suggestions, list):
        return len(suggestions)
    if operation_type == "cluster_operations" and isinstance(suggestions, dict):
        return len(suggestions.get("merges", [])) + len(suggestions.get("splits", []))
    return 0


def _review_cluster_name_suggestions(
    session_id: str, suggestions: Dict[str, str], backend
) -> Dict[str, Any]:
    st.markdown("#### 📝 Cluster Name Suggestions")
    st.caption("Review each suggestion and choose which to apply")

    selection_key = f"name_selections_{session_id}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}

    applied_count = 0
    all_clusters = backend.getAllClusters()

    pending_suggestions = {
        cluster_id: name
        for cluster_id, name in suggestions.items()
        if cluster_id not in st.session_state[selection_key]
    }

    if not pending_suggestions:
        st.success("✅ All suggestions have been reviewed!")
        accepted = sum(1 for v in st.session_state[selection_key].values() if v == "accepted")
        rejected = sum(1 for v in st.session_state[selection_key].values() if v == "rejected")
        st.info(f"📊 Summary: {accepted} accepted, {rejected} rejected")

        if st.button("🔄 Generate New Suggestions", key=f"regenerate_after_complete_{session_id}"):
            _cleanup_suggestion_session(session_id)
            st.rerun()

        return {"action": "complete", "applied_count": accepted}

    total_suggestions = len(suggestions)
    remaining = len(pending_suggestions)
    processed = total_suggestions - remaining

    st.progress(processed / total_suggestions)
    st.caption(f"Progress: {processed}/{total_suggestions} reviewed ({remaining} remaining)")

    for cluster_id, suggested_name in pending_suggestions.items():
        if cluster_id not in all_clusters:
            continue

        current_name = all_clusters[cluster_id]["cluster_name"]

        with st.container():
            st.markdown(
                f"""
<div style="
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background-color: #fafafa;
">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="flex: 1;">
            <div style="font-size: 0.85rem; color: #666;">Cluster: {cluster_id}</div>
            <div style="margin: 8px 0;">
                <span style="color: #999; text-decoration: line-through;">{current_name}</span>
                <span style="margin: 0 8px;">→</span>
                <strong style="color: #667eea;">{suggested_name}</strong>
            </div>
        </div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns([3, 1, 1])

            with col2:
                accept_key = f"accept_name_{session_id}_{cluster_id}"
                if st.button("✅ Accept", key=accept_key, use_container_width=True):
                    success, msg = backend.changeClusterName(cluster_id, suggested_name)
                    if success:
                        st.session_state[selection_key][cluster_id] = "accepted"
                        applied_count += 1
                        st.success(f"✅ Applied: {suggested_name}")
                        save_finetuning_results_to_session(backend)
                        _force_dnd_refresh()
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            with col3:
                reject_key = f"reject_name_{session_id}_{cluster_id}"
                if st.button("❌ Reject", key=reject_key, use_container_width=True):
                    st.session_state[selection_key][cluster_id] = "rejected"
                    st.info(f"Rejected suggestion for {current_name}")
                    time.sleep(0.3)
                    st.rerun()

    return {"action": "partial", "applied_count": applied_count}


def _review_entry_move_suggestions(session_id: str, suggestions: List[Dict], backend) -> Dict[str, Any]:
    st.markdown("#### 🔄 Entry Move Suggestions")
    st.caption("Review each move and choose which to apply")

    selection_key = f"move_selections_{session_id}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}

    applied_count = 0
    all_clusters = backend.getAllClusters()
    all_entries = backend.getAllEntries()

    pending_suggestions = [
        (i, move) for i, move in enumerate(suggestions) if i not in st.session_state[selection_key]
    ]

    if not pending_suggestions:
        st.success("✅ All suggestions have been reviewed!")
        accepted = sum(1 for v in st.session_state[selection_key].values() if v == "accepted")
        rejected = sum(1 for v in st.session_state[selection_key].values() if v == "rejected")
        st.info(f"📊 Summary: {accepted} accepted, {rejected} rejected")

        if st.button("🔄 Generate New Suggestions", key=f"regenerate_after_complete_{session_id}"):
            _cleanup_suggestion_session(session_id)
            st.rerun()

        return {"action": "complete", "applied_count": accepted}

    total_suggestions = len(suggestions)
    remaining = len(pending_suggestions)
    processed = total_suggestions - remaining

    st.progress(processed / total_suggestions)
    st.caption(f"Progress: {processed}/{total_suggestions} reviewed ({remaining} remaining)")

    for i, move in pending_suggestions:
        entry_id = move.get("entry_id")
        target_cluster = move.get("target_cluster")
        reason = move.get("reason", "No reason provided")
        confidence = float(move.get("confidence", 0.0) or 0.0)

        if not entry_id or not target_cluster:
            continue

        # ✅ robust lookup: handle str/int keys
        entry = all_entries.get(entry_id)
        if entry is None and str(entry_id).isdigit():
            entry = all_entries.get(int(entry_id))  # type: ignore[arg-type]
        if not entry:
            continue

        current_cluster = entry.get("clusterID", "Unknown")
        current_cluster_name = all_clusters.get(current_cluster, {}).get("cluster_name", current_cluster)
        target_cluster_name = all_clusters.get(target_cluster, {}).get("cluster_name", target_cluster)

        with st.container():
            st.markdown(
                f"""
<div style="
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background-color: #fafafa;
">
    <div style="font-size: 0.85rem; color: #666;">Entry: {entry_id} • Confidence: {confidence:.2f}</div>
    <div style="margin: 8px 0; font-size: 0.9rem;">"{(entry.get('entry_text') or '')[:100]}..."</div>
    <div style="margin: 8px 0;">
        <span style="color: #999;">{current_cluster_name}</span>
        <span style="margin: 0 8px;">→</span>
        <strong style="color: #667eea;">{target_cluster_name}</strong>
    </div>
    <div style="font-size: 0.85rem; color: #666; font-style: italic;">
        💡 {reason}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            _, col2, col3 = st.columns([3, 1, 1])

            with col2:
                accept_key = f"accept_move_{session_id}_{i}"
                if st.button("✅ Accept", key=accept_key, use_container_width=True):
                    success, msg = backend.moveEntry(entry_id, target_cluster)
                    if success:
                        st.session_state[selection_key][i] = "accepted"
                        applied_count += 1
                        st.success(f"✅ Moved entry to {target_cluster_name}")
                        save_finetuning_results_to_session(backend)
                        _force_dnd_refresh()
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            with col3:
                reject_key = f"reject_move_{session_id}_{i}"
                if st.button("❌ Reject", key=reject_key, use_container_width=True):
                    st.session_state[selection_key][i] = "rejected"
                    st.info("Rejected move suggestion")
                    time.sleep(0.3)
                    st.rerun()

    return {"action": "partial", "applied_count": applied_count}


def _review_cluster_operation_suggestions(session_id: str, suggestions: Dict, backend) -> Dict[str, Any]:
    st.markdown("#### 🔧 Cluster Operation Suggestions")
    st.caption("Review merge and split operations")

    selection_key = f"operation_selections_{session_id}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}

    applied_count = 0
    all_clusters = backend.getAllClusters()

    merges = suggestions.get("merges", [])
    splits = suggestions.get("splits", [])

    pending_merges = [
        (i, merge) for i, merge in enumerate(merges) if f"merge_{i}" not in st.session_state[selection_key]
    ]
    pending_splits = [
        (i, split) for i, split in enumerate(splits) if f"split_{i}" not in st.session_state[selection_key]
    ]

    total_suggestions = len(merges) + len(splits)
    total_pending = len(pending_merges) + len(pending_splits)

    if total_pending == 0 and total_suggestions > 0:
        st.success("✅ All suggestions have been reviewed!")
        accepted = sum(1 for v in st.session_state[selection_key].values() if v == "accepted")
        rejected = sum(1 for v in st.session_state[selection_key].values() if v == "rejected")
        st.info(f"📊 Summary: {accepted} accepted, {rejected} rejected")

        if st.button("🔄 Generate New Suggestions", key=f"regenerate_after_complete_{session_id}"):
            _cleanup_suggestion_session(session_id)
            st.rerun()

        return {"action": "complete", "applied_count": accepted}

    if total_suggestions > 0:
        processed = total_suggestions - total_pending
        st.progress(processed / total_suggestions)
        st.caption(f"Progress: {processed}/{total_suggestions} reviewed ({total_pending} remaining)")

    if pending_merges:
        st.markdown("**🔗 Merge Suggestions**")
        for i, merge in pending_merges:
            cluster1 = merge.get("cluster1")
            cluster2 = merge.get("cluster2")
            reason = merge.get("reason", "No reason provided")
            confidence = float(merge.get("confidence", 0.0) or 0.0)

            if not cluster1 or not cluster2 or cluster1 not in all_clusters or cluster2 not in all_clusters:
                continue

            cluster1_name = all_clusters[cluster1]["cluster_name"]
            cluster2_name = all_clusters[cluster2]["cluster_name"]

            with st.container():
                st.markdown(
                    f"""
<div style="
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background-color: #fafafa;
">
    <div style="font-size: 0.85rem; color: #666;">Confidence: {confidence:.2f}</div>
    <div style="margin: 8px 0;">
        <strong style="color: #667eea;">{cluster1_name}</strong>
        <span style="margin: 0 8px;">+</span>
        <strong style="color: #667eea;">{cluster2_name}</strong>
        <span style="margin: 0 8px;">→</span>
        <span style="color: #666;">Merged Cluster</span>
    </div>
    <div style="font-size: 0.85rem; color: #666; font-style: italic;">
        💡 {reason}
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col2:
                    merge_name = st.text_input(
                        "New name (optional)",
                        key=f"merge_name_{session_id}_{i}",
                        placeholder="Auto-generated",
                    )

                with col3:
                    accept_key = f"accept_merge_{session_id}_{i}"
                    if st.button("✅ Merge", key=accept_key, use_container_width=True):
                        success, result = backend.mergeClusters(cluster1, cluster2, merge_name or None)
                        if success:
                            st.session_state[selection_key][f"merge_{i}"] = "accepted"
                            applied_count += 1
                            st.success(f"✅ Merged into: {result}")
                            save_finetuning_results_to_session(backend)
                            _force_dnd_refresh()
                            time.sleep(0.3)
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")

                with col4:
                    reject_key = f"reject_merge_{session_id}_{i}"
                    if st.button("❌ Reject", key=reject_key, use_container_width=True):
                        st.session_state[selection_key][f"merge_{i}"] = "rejected"
                        st.info("Rejected merge suggestion")
                        time.sleep(0.3)
                        st.rerun()

    if pending_splits:
        st.markdown("**✂️ Split Suggestions**")
        for i, split in pending_splits:
            cluster = split.get("cluster")
            reason = split.get("reason", "No reason provided")
            confidence = float(split.get("confidence", 0.0) or 0.0)
            suggested_sub_topics = split.get("suggested_sub_topics", [])

            if cluster not in all_clusters:
                continue

            cluster_name = all_clusters[cluster]["cluster_name"]

            with st.container():
                st.markdown(
                    f"""
<div style="
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background-color: #fafafa;
">
    <div style="font-size: 0.85rem; color: #666;">Confidence: {confidence:.2f}</div>
    <div style="margin: 8px 0;">
        <strong style="color: #667eea;">{cluster_name}</strong>
        <span style="margin: 0 8px;">→</span>
        <span style="color: #666;">Split into: {', '.join(suggested_sub_topics) if suggested_sub_topics else 'Multiple Clusters'}</span>
    </div>
    <div style="font-size: 0.85rem; color: #666; font-style: italic;">
        💡 {reason}
    </div>
    <div style="margin-top: 8px; padding: 8px; background-color: #fff3cd; border-radius: 4px;">
        ⚠️ Cluster splitting requires manual re-assignment of entries
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

                _, col2, col3 = st.columns([3, 1, 1])

                with col2:
                    acknowledge_key = f"acknowledge_split_{session_id}_{i}"
                    if st.button("👍 Acknowledge", key=acknowledge_key, use_container_width=True):
                        st.session_state[selection_key][f"split_{i}"] = "acknowledged"
                        st.info(f"Acknowledged split suggestion for {cluster_name}")
                        time.sleep(0.3)
                        st.rerun()

                with col3:
                    reject_split_key = f"reject_split_{session_id}_{i}"
                    if st.button("❌ Dismiss", key=reject_split_key, use_container_width=True):
                        st.session_state[selection_key][f"split_{i}"] = "rejected"
                        st.info("Dismissed split suggestion")
                        time.sleep(0.3)
                        st.rerun()

    return {"action": "partial", "applied_count": applied_count}


def _apply_all_suggestions(session_id: str, suggestions: Any, operation_type: str, backend) -> Dict[str, Any]:
    applied_count = 0
    try:
        if operation_type == "cluster_names" and isinstance(suggestions, dict):
            for cluster_id, new_name in suggestions.items():
                success, _ = backend.changeClusterName(cluster_id, new_name)
                if success:
                    applied_count += 1

        elif operation_type == "entry_moves" and isinstance(suggestions, list):
            for move in suggestions:
                entry_id = move.get("entry_id")
                target_cluster = move.get("target_cluster")
                if entry_id and target_cluster:
                    success, _ = backend.moveEntry(entry_id, target_cluster)
                    if success:
                        applied_count += 1

        elif operation_type == "cluster_operations" and isinstance(suggestions, dict):
            for merge in suggestions.get("merges", []):
                cluster1 = merge.get("cluster1")
                cluster2 = merge.get("cluster2")
                if cluster1 and cluster2:
                    success, _ = backend.mergeClusters(cluster1, cluster2)
                    if success:
                        applied_count += 1

        save_finetuning_results_to_session(backend)

        if applied_count > 0:
            _force_dnd_refresh()

        return {"action": "accept_all", "applied_count": applied_count}

    except Exception as e:
        st.error(f"Error applying suggestions: {e}")
        return {"action": "error", "applied_count": applied_count}


# =============================================================================
# IMPROVED LLM wrapper with better prompting and quality awareness
# =============================================================================


class LLMWrapper:
    """LLM wrapper with improved prompting, quality awareness, and operation-specific configs."""

    def __init__(self, backend=None):
        self.provider = None
        self.client = None
        self.config: Dict[str, Any] = {}
        self.initialized = False
        self.backend = backend

    def initLLM(self, provider: str = "mock", config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize LLM with given provider and configuration."""
        try:
            self.provider = (provider or "mock").lower()
            self.config = config or {}

            if self.provider == "openai":
                return self._init_openai()
            if self.provider == "mock":
                return self._init_mock()

            st.error(f"Unsupported LLM provider: {provider}")
            return False

        except Exception as e:
            st.error(f"LLM initialization failed: {e}")
            import traceback

            st.code(traceback.format_exc())
            return False

    def _init_openai(self) -> bool:
        try:
            import openai  # type: ignore

            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found. Set OPENAI_API_KEY.")
                return False
            self.client = openai.OpenAI(api_key=api_key)  # new SDK style
            self.initialized = True
            return True
        except ImportError:
            st.error("OpenAI package not installed. Run: pip install openai")
            return False

    def _init_mock(self) -> bool:
        self.initialized = True
        return True

    def callLLM(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Optional[str]:
        if not self.initialized:
            st.error("LLM not initialized. Use the AI panel or initLLM().")
            return None
        try:
            if self.provider == "openai":
                return self._call_openai(prompt, context or {}, temperature, max_tokens)
            if self.provider == "mock":
                return self._call_mock(prompt, context or {}, temperature, max_tokens)
            return None
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return None

    def _call_openai(self, prompt, context, temperature, max_tokens) -> str:
        sysmsg = self._build_system_message(context)
        resp = self.client.chat.completions.create(
            model=self.config.get("model", "gpt-3.5-turbo"),
            messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def _call_mock(self, prompt, context, temperature, max_tokens) -> str:
        p = prompt.lower()

        if "suggest" in p and "name" in p and "json" in p:
            return self._mock_cluster_names_response(context)
        if "suggest" in p and "move" in p and "json" in p:
            return self._mock_entry_moves_response(context)
        if "suggest" in p and ("merge" in p or "split" in p) and "json" in p:
            return self._mock_cluster_operations_response(context)

        if "suggest" in p and "name" in p:
            return self._mock_cluster_names_response(context)
        if "move" in p and "cluster" in p:
            return "Consider moving very short texts to Outliers—they often lack enough signal."
        if "improve" in p:
            return "Try merging overlapping small clusters, renaming with concise labels, and isolating outliers."
        return f"I understand you want help with: '{prompt}'. Here's a general suggestion..."

    def _mock_cluster_names_response(self, context) -> str:
        import json

        clusters_info = context.get("clusters", [])
        suggestions = {}

        for cluster in clusters_info:
            cid = cluster["id"]
            samples = cluster.get("sample_texts", [])
            if samples:
                words = samples[0].split()[:3]
                name = " ".join(words).title()
            else:
                name = f"Topic {cluster.get('name', cid)}"
            suggestions[cid] = name

        return json.dumps(suggestions)

    def _mock_entry_moves_response(self, context) -> str:
        import json

        clusters_info = context.get("clusters", [])
        if not clusters_info or len(clusters_info) < 2:
            return json.dumps([])

        moves = []

        if self.backend:
            try:
                all_entries = self.backend.getAllEntries()
                all_clusters = self.backend.getAllClusters()

                if not all_entries or len(all_entries) < 2:
                    return json.dumps([])

                low_confidence_entries = [
                    (eid, entry)
                    for eid, entry in all_entries.items()
                    if entry.get("probability", 1.0) < 0.5
                ]

                if not low_confidence_entries:
                    return json.dumps([])

                selected_entries = low_confidence_entries[: min(3, len(low_confidence_entries))]
                cluster_ids = list(all_clusters.keys())

                for eid, entry in selected_entries:
                    current_cluster = entry.get("clusterID")
                    available_clusters = [cid for cid in cluster_ids if cid != current_cluster]
                    if available_clusters:
                        target = available_clusters[0]
                        moves.append(
                            {
                                "entry_id": str(eid),
                                "target_cluster": str(target),
                                "reason": f"Low confidence in current cluster. Better semantic fit with {all_clusters[target]['cluster_name']}",
                                "confidence": 0.7,
                            }
                        )
            except Exception as e:
                print(f"Error generating mock moves: {e}")
                return json.dumps([])

        return json.dumps(moves)

    def _mock_cluster_operations_response(self, context) -> str:
        import json

        clusters_info = context.get("clusters", [])
        if len(clusters_info) < 2:
            return json.dumps({"merges": [], "splits": []})

        operations = {"merges": [], "splits": []}
        sorted_clusters = sorted(clusters_info, key=lambda c: c.get("count", 0))
        very_small = [c for c in sorted_clusters if c.get("count", 0) < 3]

        if len(very_small) >= 2:
            cluster1 = very_small[0]
            cluster2 = very_small[1]
            operations["merges"].append(
                {
                    "cluster1": cluster1["id"],
                    "cluster2": cluster2["id"],
                    "reason": f"Both are very small clusters ({cluster1['count']} and {cluster2['count']} items). Merging may improve coherence.",
                    "confidence": 0.75,
                }
            )

        large_clusters = [c for c in clusters_info if c.get("count", 0) > 25]
        for cluster in large_clusters[:1]:
            operations["splits"].append(
                {
                    "cluster": cluster["id"],
                    "suggested_sub_topics": ["Topic A", "Topic B"],
                    "reason": f"Very large cluster with {cluster['count']} items. May contain multiple distinct sub-topics.",
                    "confidence": 0.65,
                }
            )

        return json.dumps(operations)

    def _extract_json(self, text: str):
        import json

        try:
            return json.loads(text.strip())
        except Exception:
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        match = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        return None

    def _build_system_message(self, context: Dict[str, Any]) -> str:
        base = (
            "You are an expert in text clustering and data analysis. "
            "Your task is to analyze clustering results and provide HIGH-QUALITY suggestions. "
            "CRITICAL: Only suggest changes that would CLEARLY improve clustering quality. "
            "If the clustering is already good, return empty suggestions. "
            "IMPORTANT: Return ONLY valid JSON. Do not include explanations, markdown formatting, or code blocks. "
            "Just the raw JSON object or array."
        )
        clusters_info = (context or {}).get("clusters", [])
        if clusters_info:
            base += f"\n\nCurrent clusters: {len(clusters_info)}"
            for i, cluster in enumerate(clusters_info[:5]):
                name = cluster.get("name", f"Cluster {i+1}")
                item_count = cluster.get("count", len(cluster.get("items", [])))
                base += f"\n- {name}: {item_count} items"
        return base

    def create_suggestion_session(self, operation_type: str, params: Dict[str, Any]) -> str:
        import uuid

        session_id = f"{operation_type}_{uuid.uuid4().hex[:8]}"
        st.session_state[f"suggestion_session_{session_id}"] = {
            "operation_type": operation_type,
            "params": params,
            "suggestions": None,
            "status": "pending",
            "timestamp": time.time(),
        }
        return session_id

    def generate_suggestions_async(self, session_id: str, context: Dict[str, Any]) -> bool:
        session_data = st.session_state.get(f"suggestion_session_{session_id}")
        if not session_data:
            return False

        operation_type = session_data["operation_type"]

        try:
            if operation_type == "cluster_names":
                suggestions = self.suggest_cluster_names()

            elif operation_type == "entry_moves":
                params = session_data["params"]
                all_entries = self.backend.getAllEntries()
                entries_to_analyze = []

                filter_cluster = params.get("filter_cluster", "All clusters")
                input_confidence_filter = params.get("input_confidence_filter", 0.5)

                for eid, entry in all_entries.items():
                    if filter_cluster != "All clusters" and entry.get("clusterID") != filter_cluster:
                        continue
                    if entry.get("probability", 0) >= input_confidence_filter:
                        continue
                    entries_to_analyze.append(eid)

                suggestions = self.suggest_entry_moves(entries_to_analyze[:20])

                quality_threshold = params.get("suggestion_quality_threshold", 0.6)
                if suggestions:
                    suggestions = [s for s in suggestions if s.get("confidence", 0) >= quality_threshold]

            elif operation_type == "cluster_operations":
                suggestions = self.suggest_cluster_operations()

            else:
                return False

            session_data["suggestions"] = suggestions
            session_data["status"] = "reviewing"
            st.session_state[f"suggestion_session_{session_id}"] = session_data
            return True

        except Exception as e:
            session_data["status"] = "error"
            session_data["error"] = str(e)
            st.session_state[f"suggestion_session_{session_id}"] = session_data
            return False

    def build_context(self, include_texts: bool = False, max_samples: int = 5) -> Dict[str, Any]:
        if not self.backend:
            return {"clusters": []}

        clusters = self.backend.getAllClusters()
        all_entries = self.backend.getAllEntries()

        cluster_list = []
        for cid, c in clusters.items():
            cluster_data = {"id": cid, "name": c["cluster_name"], "count": len(c["entry_ids"])}
            if include_texts:
                sample_texts = [
                    all_entries[eid]["entry_text"][:200]
                    for eid in c["entry_ids"][:max_samples]
                    if eid in all_entries
                ]
                cluster_data["sample_texts"] = sample_texts
            else:
                cluster_data["items"] = ["_"] * len(c["entry_ids"])
            cluster_list.append(cluster_data)

        return {"clusters": cluster_list}

    def _analyze_cluster_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        clusters = context.get("clusters", [])
        if not clusters:
            return {}

        sizes = [c.get("count", 0) for c in clusters]
        avg_size = sum(sizes) / len(sizes) if sizes else 0

        return {
            "total_clusters": len(clusters),
            "small_clusters": [c["id"] for c in clusters if c.get("count", 0) < 5],
            "large_clusters": [c["id"] for c in clusters if c.get("count", 0) > 20],
            "avg_cluster_size": avg_size,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
        }

    def _format_cluster_stats(self, stats: Dict[str, Any]) -> str:
        if not stats:
            return "No statistics available"
        return f"""
- Total clusters: {stats.get('total_clusters', 0)}
- Small clusters (<5 items): {len(stats.get('small_clusters', []))}
- Large clusters (>20 items): {len(stats.get('large_clusters', []))}
- Average size: {stats.get('avg_cluster_size', 0):.1f}
- Size range: {stats.get('min_size', 0)}-{stats.get('max_size', 0)}
"""

    def _format_entries_for_analysis(self, entry_ids: List[str], context: Dict[str, Any]) -> str:
        all_entries = self.backend.getAllEntries()
        all_clusters = self.backend.getAllClusters()

        formatted = []
        for eid in entry_ids[:10]:
            entry = all_entries.get(eid)
            if not entry:
                continue

            current_cluster = entry.get("clusterID")
            cluster_info = all_clusters.get(current_cluster, {})

            formatted.append(
                f"ID: {eid} | Confidence: {entry.get('probability', 0):.2f} | "
                f"Current Cluster: {cluster_info.get('cluster_name', current_cluster)} | "
                f"Text: {entry['entry_text'][:150]}..."
            )

        return "\n".join(formatted)

    def suggest_cluster_operations(self) -> Optional[Dict[str, Any]]:
        if not self.initialized:
            return None

        context = self.build_context(include_texts=True, max_samples=3)
        clusters_info = context.get("clusters", [])
        cluster_stats = self._analyze_cluster_quality(context)

        cluster_details = []
        for cluster in clusters_info:
            cluster_details.append(
                f"- {cluster['id']}: '{cluster['name']}' ({cluster['count']} items)\n"
                f"  Sample texts: {'; '.join(cluster.get('sample_texts', [])[:2])}"
            )

        prompt = f"""Analyze {len(clusters_info)} clusters for merge/split opportunities.

**CLUSTERING QUALITY METRICS:**
{self._format_cluster_stats(cluster_stats)}

**MERGE CRITERIA (suggest ONLY if ALL apply):**
1. Both clusters have clear semantic overlap in sample texts
2. Combined size would be reasonable (<30 items total)
3. Merger would create more coherent cluster than keeping separate
4. At least one cluster is very small (<5 items) OR both clearly share the same theme

**SPLIT CRITERIA (suggest ONLY if ALL apply):**
1. Cluster has >25 items
2. Sample texts show CLEAR thematic diversity (not just minor variations)
3. Can identify 2+ distinct sub-topics in "suggested_sub_topics" field
4. Split would meaningfully improve overall clustering quality

**CRITICAL RULES:**
- DO NOT suggest merges just because clusters are small
- DO NOT suggest splits just because clusters are large
- If no operations would CLEARLY improve quality, return: {{"merges": [], "splits": []}}
- Each suggestion must include "confidence" score (0.0-1.0)
- Only include suggestions with confidence >= 0.6

**Clusters:**
{chr(10).join(cluster_details)}

**Output Format (JSON only, no markdown):**
{{
  "merges": [
    {{
      "cluster1": "cluster_X",
      "cluster2": "cluster_Y",
      "reason": "Specific overlapping theme from samples",
      "confidence": 0.75
    }}
  ],
  "splits": [
    {{
      "cluster": "cluster_Z",
      "suggested_sub_topics": ["Specific Topic A", "Specific Topic B"],
      "reason": "Specific diversity observed in samples",
      "confidence": 0.70
    }}
  ]
}}

Return ONLY the JSON. No explanation, no markdown.
"""

        response = self.callLLM(prompt, context, temperature=0.3, max_tokens=700)
        if response:
            parsed = self._extract_json(response)
            if parsed and isinstance(parsed, dict):
                merges = [m for m in parsed.get("merges", []) if m.get("confidence", 0) >= 0.6]
                splits = [s for s in parsed.get("splits", []) if s.get("confidence", 0) >= 0.6]
                return {"merges": merges, "splits": splits}

        return {"merges": [], "splits": []}

    def suggest_entry_moves(self, entries_to_analyze: List[str]) -> Optional[List[Dict[str, str]]]:
        if not self.initialized or not entries_to_analyze:
            return None

        context = self.build_context(include_texts=True, max_samples=5)
        cluster_stats = self._analyze_cluster_quality(context)

        prompt = f"""Analyze {len(entries_to_analyze)} text entries for potential cluster reassignments.

**CLUSTERING QUALITY METRICS:**
{self._format_cluster_stats(cluster_stats)}

**SUGGESTION GUIDELINES:**
Consider suggesting moves when:
- Entry has low-to-medium confidence (<0.5) AND might fit better elsewhere
- Entry's content seems mismatched with its cluster's dominant theme
- Entry contains keywords that align strongly with a different cluster
- Moving would noticeably improve both source and target cluster coherence

**Important:**
- Prioritize moves that clearly improve quality
- Low confidence alone isn't enough - entry must fit better elsewhere
- If an entry fits reasonably well, don't suggest moving it
- Rate your confidence 0.0-1.0 for each suggestion

**Entries to Analyze:**
{self._format_entries_for_analysis(entries_to_analyze, context)}

**Output Format (JSON only, no markdown):**
[
  {{"entry_id": "ID", "target_cluster": "cluster_X", "reason": "Specific reason", "confidence": 0.70}},
  ...
]

**If no moves would improve quality, return: []**
Return ONLY the JSON array. No explanation, no markdown.
"""

        response = self.callLLM(prompt, context, temperature=0.3, max_tokens=900)
        if response:
            parsed = self._extract_json(response)
            if parsed and isinstance(parsed, list):
                high_quality = [s for s in parsed if s.get("confidence", 0) >= 0.6]
                return high_quality if high_quality else None

        return None

    def suggest_cluster_names(self) -> Optional[Dict[str, str]]:
        """Generate cluster name suggestions with quality awareness."""
        if not self.initialized:
            return None

        context = self.build_context(include_texts=True, max_samples=5)
        clusters_info = context.get("clusters", [])
        if not clusters_info:
            return None

        cluster_details = []
        for cluster in clusters_info:
            samples = cluster.get("sample_texts", [])
            detail = f"- {cluster['id']}: '{cluster['name']}' ({cluster['count']} items)"
            if samples:
                detail += f"\n  Sample texts: {'; '.join(samples[:3])}"
            cluster_details.append(detail)

        prompt = f"""Analyze {len(clusters_info)} clusters and suggest better, more descriptive names based on their content.

**NAMING GUIDELINES:**
1. Names should be concise (2-4 words)
2. Names should capture the main theme from sample texts
3. Use descriptive, professional terminology
4. Avoid generic names like "Topic 1" or "Group A"
5. If current name is already good, keep it or make minor improvements

**Clusters to Analyze:**
{chr(10).join(cluster_details)}

**Output Format (JSON only, no markdown):**
{{"cluster_0": "Descriptive Name", "cluster_1": "Another Name"}}

**IMPORTANT:**
- Return ONLY the JSON object
- No explanations, no markdown code blocks
- If a cluster already has a good name, you can keep it
- Base names on actual content from samples, not guesses
"""

        response = self.callLLM(prompt, context, temperature=0.3, max_tokens=400)
        if response:
            parsed = self._extract_json(response)
            if parsed and isinstance(parsed, dict):
                valid_cluster_ids = {c["id"] for c in clusters_info}
                filtered = {k: v for k, v in parsed.items() if k in valid_cluster_ids}
                return filtered if filtered else None

        return None


# =============================================================================
# Singleton helpers (MUST be at module level, not inside class)
# =============================================================================

_llm_instance: Optional[LLMWrapper] = None


def get_llm_wrapper() -> LLMWrapper:
    """Get the singleton LLM wrapper instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMWrapper()
    return _llm_instance


def reset_llm_wrapper():
    """Reset the singleton instance (useful for testing)."""
    global _llm_instance
    _llm_instance = None


def initLLM(provider: str = "mock", config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize LLM with given provider and config."""
    return get_llm_wrapper().initLLM(provider, config)


def callLLM(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> Optional[str]:
    """Call LLM with given prompt and context."""
    return get_llm_wrapper().callLLM(prompt, context, temperature, max_tokens)


def get_llm_status() -> Dict[str, Any]:
    """Get current LLM initialization status."""
    w = get_llm_wrapper()
    return {"initialized": w.initialized, "provider": w.provider, "ready": w.initialized}


def quick_llm_test() -> bool:
    """Quick test to verify LLM is working."""
    resp = callLLM("Say 'Hello, I am working!' in exactly those words.", max_tokens=20)
    return resp is not None
