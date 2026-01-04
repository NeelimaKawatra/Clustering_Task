# frontend/frontend_finetuning.py
# Finetuning tab with Drag & Drop + LLM helper with Cursor-like review UX.

from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple

import re
import time
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
    pass


# helper function
def expander_open_once(key: str, default: bool = False) -> bool:
    """Return True once if session flag is set; then clear it."""
    return st.session_state.pop(key, default)

# --- zero-width ID codec (hide IDs in labels but keep them recoverable) ---
_ZW0 = "\u200b"   # ZWSP
_ZW1 = "\u200c"   # ZWNJ
_ZWS = "\u2060"   # WORD JOINER (delimiter)
_ZWB = "\u2062"   # INVISIBLE TIMES (start)
_ZWE = "\u2063"   # INVISIBLE SEPARATOR (end)

def _encode_hidden_id(eid: str) -> str:
    bits = "".join(f"{ord(c):08b}" for c in eid)            # 8-bit per char
    payload = "".join(_ZW1 if b == "1" else _ZW0 for b in bits)
    return f"{_ZWB}{payload}{_ZWE}"                          # invisible token

def _decode_hidden_id(label: str) -> Optional[str]:
    # Extract the invisible token and turn it back into the original string
    if _ZWB in label and _ZWE in label:
        enc = label.split(_ZWB, 1)[1].split(_ZWE, 1)[0]
        bits = "".join("1" if ch == _ZW1 else ("0" if ch == _ZW0 else "") for ch in enc)
        if bits and len(bits) % 8 == 0:
            chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
            return "".join(chars)
    return None


# =============================================================================
# MAIN TAB
# =============================================================================

def tab_finetuning(backend_available: bool):
    """Human-in-the-Loop Fine-tuning tied to backend API"""

    # --- always fetch fresh clusters for this run ---

    # Track tab visit
    if backend_available and hasattr(st.session_state, "backend") and st.session_state.backend:
        st.session_state.backend.track_activity(
            st.session_state.session_id, "tab_visit", {"tab_name": "finetuning"}
        )

    # Mark Fine-tuning as visited
    st.session_state["finetuning_ever_visited"] = True

    # Prerequisite check
    if not st.session_state.get("clustering_results") or not st.session_state.clustering_results.get("success", False):
        st.error("Please complete Clustering first.")
        st.info("Go to the Clustering tab and run the clustering analysis to see results here.")
        return

    # Initialize fine-tuning backend state from clustering outcome
    if not _initialize_backend():
        st.error("Failed to initialize fine-tuning backend.")
        return

    backend = get_finetuning_backend()

    # view, rename, delete
    # create, merge
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
        with col1: st.metric("Total Clusters", len(all_clusters))
        with col2: st.metric("Total Text Entries", len(all_entries))
        with col3: st.metric("Manual Clusters", modification_summary.get("manual_clusters_created", 0))
        with col4: st.metric("Modified Entries", f"{modification_summary.get('modification_percentage', 0):.1f}%")

    # ---- Keep/manage expander open state
    st.session_state.setdefault("exp_manage_open", False)

    # Precompute edit detection
    all_clusters = backend.getAllClusters()
    editing_any = False
    tmp_pairs = []
    for i, (cid, cdata) in enumerate(all_clusters.items()):
        key_prefix = f"{cid}_{i}"
        tmp_pairs.append((key_prefix, cdata["cluster_name"], cid, cdata))
        current_val = st.session_state.get(f"name_{key_prefix}", cdata["cluster_name"])
        if current_val.strip() != cdata["cluster_name"]:
            editing_any = True

    parent_expanded = st.session_state["exp_manage_open"] or editing_any
    with st.expander("💡 You can view, rename, check confidence, and delete clusters:",
                     expanded=parent_expanded):

        def _keep_manage_open():
            st.session_state["exp_manage_open"] = True

        for key_prefix, cluster_name, cluster_id, cluster_data in tmp_pairs:
            # Child expander sticks open while editing this cluster's name
            current_val = st.session_state.get(f"name_{key_prefix}", cluster_name)
            child_expanded = (current_val.strip() != cluster_name) or st.session_state.get(
                f"exp_manage_child_{key_prefix}", False
            )

            with st.expander(
                f"🗂️ {cluster_data['cluster_name']} ({len(cluster_data['entry_ids'])} entries)",
                expanded=child_expanded,
            ):
                col1, col2, col3 = st.columns([2, 1, 1])

                # Rename
                with col1:
                    new_name = st.text_input(
                        "Cluster name",
                        value=cluster_name,
                        key=f"name_{key_prefix}",
                        on_change=_keep_manage_open,  # keep open on Enter/typing
                    )
                    changed = new_name.strip() != cluster_name
                    if st.button("Rename", key=f"update_name_{key_prefix}", disabled=not changed):
                        ok, msg = backend.changeClusterName(cluster_id, new_name.strip())
                        # after success, collapse parent & child; on failure keep open
                        if ok:
                            st.session_state.finetuning_success_message = f"✏️ {msg}"
                            st.session_state["exp_manage_open"] = False
                            st.session_state[f"exp_manage_child_{key_prefix}"] = False
                            save_finetuning_results_to_session(backend)
                            st.rerun()
                        else:
                            st.session_state.finetuning_error_message = msg
                            st.session_state["exp_manage_open"] = True
                            st.session_state[f"exp_manage_child_{key_prefix}"] = True
                            st.rerun()

                # Stats
                with col2:
                    stats = backend.getClusterStatistics(cluster_id)
                    if stats:
                        st.metric("Avg Confidence", f"{stats['avg_probability']:.2f}")

                # Delete
                with col3:
                    if st.button("Delete Cluster", key=f"delete_{key_prefix}"):
                        ok, msg = backend.deleteCluster(cluster_id)
                        if ok:
                            st.session_state.finetuning_success_message = f"🗑️ {msg}"
                            st.session_state["exp_manage_open"] = False
                            st.session_state[f"exp_manage_child_{key_prefix}"] = False
                            save_finetuning_results_to_session(backend)
                            st.rerun()
                        else:
                            st.session_state.finetuning_error_message = msg
                            st.session_state["exp_manage_open"] = True
                            st.session_state[f"exp_manage_child_{key_prefix}"] = True
                            st.rerun()

    # ---- Create + Merge
    st.session_state.setdefault("exp_create_merge_open", False)

    def _keep_merge_open():
        st.session_state["exp_create_merge_open"] = True

    with st.expander("💡 You can create new clusters or merge existing clusters:",
                     expanded=st.session_state["exp_create_merge_open"]):

        current_clusters = backend.getAllClusters()
        col1, col2 = st.columns(2)

        # Create
        with col1:
            st.markdown("**Create New Cluster**")
            new_cluster_name = st.text_input(
                "New cluster name",
                placeholder="Type here...",
                key="new_cluster_name",
                on_change=_keep_merge_open,  # ENTER should not collapse
            )
            if st.button("Create New Cluster"):
                if new_cluster_name.strip():
                    ok, result = backend.createNewCluster(new_cluster_name.strip())
                    if ok:
                        st.session_state.finetuning_success_message = f"✅ Created cluster: '{result}'"
                        st.session_state["exp_create_merge_open"] = False  # collapse on success
                        save_finetuning_results_to_session(backend)
                        st.rerun()
                    else:
                        st.session_state.finetuning_error_message = result
                        st.session_state["exp_create_merge_open"] = True
                        st.rerun()

        # Merge
        with col2:
            st.markdown("**Merge Clusters**")
            cluster_ids = list(current_clusters.keys())
            if len(cluster_ids) >= 2:
                options = [f"{current_clusters[c]['cluster_name']} ({len(current_clusters[c]['entry_ids'])} entries)"
                           for c in cluster_ids]

                idx1 = st.selectbox(
                    "First cluster",
                    range(len(options)),
                    key="merge_cluster1",
                    format_func=lambda x: options[x],
                    on_change=_keep_merge_open,
                )
                idx2 = st.selectbox(
                    "Second cluster",
                    range(len(options)),
                    key="merge_cluster2",
                    format_func=lambda x: options[x],
                    on_change=_keep_merge_open,
                )
                merge_name = st.text_input(
                    "Merged cluster name (optional)",
                    key="merge_name",
                    on_change=_keep_merge_open,
                )

                if st.button("Merge Clusters") and idx1 != idx2:
                    c1, c2 = cluster_ids[idx1], cluster_ids[idx2]
                    ok, result = backend.mergeClusters(c1, c2, merge_name or None)
                    if ok:
                        st.session_state.finetuning_success_message = f"🔄 Merged into cluster: '{result}'"
                        st.session_state["exp_create_merge_open"] = False  # collapse on success
                        save_finetuning_results_to_session(backend)
                        st.rerun()
                    else:
                        st.session_state.finetuning_error_message = result
                        st.session_state["exp_create_merge_open"] = True
                        st.rerun()


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
        # If we haven't touched this section in this session, force filters to 'All'
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
        # IMPORTANT: Get fresh cluster data from backend to ensure we have latest state
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

                old_cluster_of[eid] = cluster_id

            header = f"{cdata.get('cluster_name', cluster_id)} ({len(shown_eids)} / {len(total_eids)} shown)"
            containers.append({"header": header, "items": items})
            orig_container_ids.append(cluster_id)
            total_shown += len(items)

        if filter_text or conf_level:
            if conf_level:
                st.caption(f'Filter "{filter_text or "*"}" + confidence={conf_level} → showing {total_shown} item(s).')
            else:
                st.caption(f'Filter "{filter_text}" → showing {total_shown} item(s).')

        # --- Render DnD (guarded)
        result = None
        filter_hash = hash((filter_text or "", conf_level or "all"))
        _sortable_key = f"dnd_board_{st.session_state.get('finetuning_refresh_token', 0)}_{filter_hash}"

        if "sort_items" in globals():
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
            if filter_text or conf_level:
                st.caption("No pending changes among the filtered items.")
            else:
                st.caption("No pending changes.")
            return

        # --- Compute pending moves (visible items only)
        def _eid_from_item(item: Any) -> Optional[str]:
            if not isinstance(item, str):
                return None
            # Preferred: decode hidden zero-width token
            hid = _decode_hidden_id(item)
            if hid:
                return hid
            # Back-compat: old "eid: text" format if any are still in session
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
                if eid and old_cluster_of.get(eid) != new_cid:
                    pending_moves.append((eid, new_cid))
        
        # Debug: Log if pending moves exist
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
                
                # Keep expander open to show updated state
                st.session_state["exp_drag_open"] = True
                save_finetuning_results_to_session(backend)
                try: st.cache_data.clear()
                except Exception: pass
                try: st.cache_resource.clear()
                except Exception: pass
                # Force widget to completely reset on next render
                st.session_state["finetuning_refresh_token"] = st.session_state.get(
                    "finetuning_refresh_token", 0
                ) + 1
                
                # Clear ALL sortable-related state to force fresh widget
                for key in list(st.session_state.keys()):
                    if "sortable" in key.lower() or key.startswith("sort_items"):
                        del st.session_state[key]
                
                # Add marker that we just applied changes to help debug
                st.session_state["_last_dnd_apply_token"] = st.session_state["finetuning_refresh_token"]
                
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

    with st.expander("💡 You can search, inspect, and move a single entry:",
                     expanded=st.session_state["exp_entry_open"]):

        all_entries = backend.getAllEntries()
        all_clusters = backend.getAllClusters()

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
                on_change=_keep_entry_open,  # keep open on Enter/typing
            )

            if search_text:
                matching_entries = [
                    eid for eid, ed in all_entries.items()
                    if search_text.lower() in ed["entry_text"].lower()
                ]
                if matching_entries:
                    selected_entry = st.selectbox(
                        "Matching entries",
                        matching_entries,
                        key="matching_entries_box",
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
                    key="all_entries_box",
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
                        st.text(f"Current Cluster: {cluster_info['cluster_name']} ({len(cluster_info['entry_ids'])} entries)")
                    else:
                        st.text(f"Current Cluster: {cluster_id}")

                    st.text(f"Confidence: {entry_data.get('probability', 0):.2f}")

                    cluster_options = list(all_clusters.keys())
                    current_cluster = entry_data.get("clusterID")
                    current_index = cluster_options.index(current_cluster) if current_cluster in cluster_options else 0

                    target_cluster = st.selectbox(
                        "Move Selected Entry to Cluster",
                        cluster_options,
                        index=current_index,
                        format_func=lambda x: f"{all_clusters[x]['cluster_name']} ({len(all_clusters[x]['entry_ids'])} entries)",
                        key=f"move_{selected_entry}",
                        on_change=_keep_entry_open,
                    )

                    if target_cluster != current_cluster:
                        if st.button(
                            f"Move to {all_clusters[target_cluster]['cluster_name']}",
                            key=f"move_btn_{selected_entry}",
                        ):
                            success, message = backend.moveEntry(selected_entry, target_cluster)
                            if success:
                                st.session_state.finetuning_success_message = f"📦 {message}"
                                st.session_state["exp_entry_open"] = False  # collapse on success
                                save_finetuning_results_to_session(backend)
                                st.rerun()
                            else:
                                st.session_state.finetuning_error_message = message
                                st.session_state["exp_entry_open"] = True
                                st.rerun()


def show_ai_assist_interface(backend):
    """AI assist with Cursor-like review UX for suggestions"""
    
    st.markdown("---")

    # ✅ ADD THIS NEW SECTION FIRST
    st.markdown("### 🤖 AI-Powered Fine-tuning Assistant")
    
    with st.expander("ℹ️ What can AI Assist do?", expanded=False):
        st.markdown("""
        **AI can help you:**
        - 🏷️ **Name clusters** based on their content
        - 🔄 **Move entries** to better-fitting clusters  
        - 🔧 **Merge/split** clusters for better organization
        
        **Cost:** Mock (free) • GPT-4o-mini ($0.01-0.05) • GPT-4o (10x more)
        
        💡 Configure LLM Settings first, then generate suggestions below.
        """)
    
    
    # Import check function
    from frontend.frontend_llm_settings import check_llm_configuration
    
    # Check if LLM is configured
    llm_status = check_llm_configuration()
    
    if not llm_status['configured']:
        st.warning(f"⚠️ **{llm_status['message']}**")
        st.markdown("""
        **AI Assist requires LLM configuration.**
        
        Go to **LLM Settings** in the sidebar to:
        - Choose your LLM provider (Mock for testing, OpenAI for production)
        - Select model and temperature
        - Configure API credentials
        """)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("⚙️ Open LLM Settings", use_container_width=True):
                st.session_state.current_page = "llm_settings"
                st.rerun()
        
        return
    
    # Get session config
    llm_config = st.session_state.llm_config
    provider = llm_config['provider']
    model = llm_config['model']
    temperature = llm_config['temperature']
    
    # Show current configuration at top
    st.success(f"✅ **AI Assistant Active**: {provider.upper()} ({model}) • Temperature: {temperature:.2f}")
    
    # Check for active review sessions
    active_sessions = [
        key.replace("suggestion_session_", "") for key in st.session_state.keys()
        if key.startswith("suggestion_session_") 
        and st.session_state[key].get("status") == "reviewing"
    ]
    
    # If there's an active review session, show it
    if active_sessions:
        session_id = active_sessions[0]
        result = show_suggestion_review_panel(session_id, backend)
        
        if result["action"] == "accept_all" and result["applied_count"] > 0:
            st.success(f"✅ Applied {result['applied_count']} suggestion(s)!")
            st.balloons()
            time.sleep(1)
            # Session already cleaned up by _cleanup_suggestion_session
            st.rerun()
        elif result["action"] == "partial" and result["applied_count"] > 0:
            # Partial application - one by one, don't clean up yet
            st.success(f"✅ Applied suggestion!")
            save_finetuning_results_to_session(backend)
            # Don't rerun - let user continue reviewing
        elif result["action"] == "reject":
            st.info("All suggestions rejected.")
            # Session already cleaned up by _cleanup_suggestion_session
            time.sleep(0.5)
            st.rerun()
        elif result["action"] == "regenerate":
            # Session already cleaned up by _cleanup_suggestion_session
            st.info("Ready to regenerate suggestions...")
            time.sleep(0.5)
            st.rerun()
        
        return
    
    # No active review - show generation interface
    with st.expander("🤖 Generate AI Suggestions", expanded=True):
        
        st.markdown("**Choose AI Operation**")
        operation = st.selectbox(
            "What would you like AI to help with?",
            ["Suggest Cluster Names", "Suggest Entry Moves", "Suggest Merges/Splits"],
            key="ft_ai_operation"
        )
        
        # Initialize LLM
        if 'llm_wrapper' not in st.session_state:
            st.session_state.llm_wrapper = LLMWrapper(backend)
        
        llm_wrapper = st.session_state.llm_wrapper
        llm_wrapper.backend = backend
        
        if provider != "mock":
            config = {"model": model, "api_key": os.getenv("OPENAI_API_KEY")}
            success = llm_wrapper.initLLM(provider=provider, config=config)
            if not success:
                st.error("❌ Failed to initialize LLM. Check your configuration in LLM Settings.")
                if st.button("⚙️ Go to LLM Settings"):
                    st.session_state.current_page = "llm_settings"
                    st.rerun()
                return
        else:
            llm_wrapper.initLLM(provider="mock", config={"model": "mock"})
        
        # Operation-specific parameters
        params = {}
        if operation == "Suggest Entry Moves":
            col1, col2 = st.columns(2)
            with col1:
                filter_cluster = st.selectbox(
                    "Analyze entries from",
                    ["All clusters"] + list(backend.getAllClusters().keys()),
                    key="ft_ai_move_cluster"
                )
                params["filter_cluster"] = filter_cluster
            
            with col2:
                min_confidence = st.slider(
                    "Min confidence threshold",
                    0.0, 1.0, 0.3, 0.1,
                    key="ft_ai_move_conf"
                )
                params["min_confidence"] = min_confidence
        
        # Generate button
        if st.button("🔮 Generate Suggestions", type="primary", use_container_width=True):
            
            # Create suggestion session
            operation_map = {
                "Suggest Cluster Names": "cluster_names",
                "Suggest Entry Moves": "entry_moves",
                "Suggest Merges/Splits": "cluster_operations"
            }
            
            operation_type = operation_map[operation]
            session_id = llm_wrapper.create_suggestion_session(operation_type, params)
            
            with st.spinner("🤖 Analyzing your data and generating suggestions..."):
                context = llm_wrapper.build_context(include_texts=True, max_samples=3)
                
                # Generate suggestions
                success = llm_wrapper.generate_suggestions_async(session_id, context)
                
                if success:
                    st.success("✅ Suggestions generated! Review them below.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Failed to generate suggestions. Try again.")
                    # Clean up failed session
                    _cleanup_suggestion_session(session_id)


def _cid_to_int(cid) -> int:
    if isinstance(cid, int):
        return cid
    s = str(cid)
    if s == "outliers":
        return -1
    m = re.search(r'(\d+)$', s)
    return int(m.group(1)) if m else -1


def build_finetuning_results_snapshot(backend) -> dict:
    """
    EXACT mirror of clustering results dict.
    Writes SSOT labels into metadata['topic_keywords'] using current cluster_name(s).
    """
    entries  = backend.getAllEntries()     # {entryID: {..., 'clusterID','probability','entry_text', ...}}
    clusters = backend.getAllClusters()    # {clusterID: {'cluster_name','entry_ids', ...}}

    # aligned lists
    sorted_eids = sorted(entries.keys(), key=lambda x: str(x))
    texts, topics, probabilities = [], [], []
    for eid in sorted_eids:
        e   = entries[eid]
        txt = e.get("entry_text") or e.get("original_text") or ""
        cid = _cid_to_int(e.get("clusterID", -1))
        p   = float(e.get("probability", 0.0) or 0.0)
        texts.append(txt); topics.append(cid); probabilities.append(p)

    total_texts = len(texts)
    outliers     = sum(1 for t in topics if t == -1)
    clustered    = total_texts - outliers
    n_clusters   = len({t for t in topics if t != -1})
    success_rate = (clustered / total_texts * 100.0) if total_texts else 0.0

    high = sum(1 for p in probabilities if p >= 0.7)
    med  = sum(1 for p in probabilities if 0.3 <= p < 0.7)
    low  = sum(1 for p in probabilities if p < 0.3)
    avg_conf = (sum(probabilities) / len(probabilities)) if probabilities else 0.0

    # SSOT labels: int cluster_id -> [display_name]
    topic_keywords = {}
    for cid_str, c in clusters.items():
        k  = _cid_to_int(cid_str)
        nm = (c.get("cluster_name") or ("Outliers" if k == -1 else str(k))).strip()
        topic_keywords[k] = [nm] if nm else [("Outliers" if k == -1 else str(k))]

    params_used = (st.session_state.get("clustering_results") or {}).get("parameters_used", {})
    prev_meta   = (st.session_state.get("clustering_results") or {}).get("metadata", {})
    n_features  = int(prev_meta.get("n_features", 0))
    n_components= int(prev_meta.get("n_components", 0))

    return {
        "success": True,
        "topics": topics,                  # list[int]
        "probabilities": probabilities,    # list[float]
        "predictions": topics,             # list[int]
        "texts": texts,                    # list[str]
        "metadata": {
            "model_type": "Manual",
            "n_features": n_features,
            "n_components": n_components,
            "topic_keywords": topic_keywords,   # <-- SSOT label map
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
        "performance": {                        # manual edits -> zeros
            "total_time": 0.0,
            "setup_time": 0.0,
            "clustering_time": 0.0,
        },
        "parameters_used": params_used,      
    }


def save_finetuning_results_to_session(backend) -> None:
    """
    Save fine-tuning results to session and mark as changed.
    This flags that Results tab needs to reload.
    """
    
    # Build fresh snapshot
    st.session_state.finetuning_results = build_finetuning_results_snapshot(backend)
    
    # ✅ Mark Results tab as incomplete - user needs to review updated results
    st.session_state.tab_results_complete = False
    st.session_state.permanent_progress['results'] = False
# =============================================================================
# NEW: Cursor-like Review UX Components
# =============================================================================

def show_suggestion_review_panel(session_id: str, backend) -> Dict[str, Any]:
    """
    Cursor-like review panel for LLM suggestions.
    Returns dict with action taken: {'action': 'accept'/'reject'/'partial', 'applied_count': N}
    """
    session_data = st.session_state.get(f"suggestion_session_{session_id}")
    if not session_data or session_data["status"] != "reviewing":
        return {"action": "none", "applied_count": 0}
    
    operation_type = session_data["operation_type"]
    suggestions = session_data["suggestions"]
    
    st.markdown("---")
    st.markdown("### 🤖 AI Suggestions Ready for Review")
    
    # Summary bar (Cursor-style)
    total_suggestions = _count_suggestions(suggestions, operation_type)
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.markdown(f"**{total_suggestions} suggestions** generated based on your clustering data")
    
    with col2:
        if st.button("✅ Accept All", key=f"accept_all_{session_id}", type="primary"):
            result = _apply_all_suggestions(session_id, suggestions, operation_type, backend)
            # FIXED: Clean up session after accept all
            _cleanup_suggestion_session(session_id)
            return result
    
    with col3:
        if st.button("❌ Reject All", key=f"reject_all_{session_id}"):
            # FIXED: Clean up session after reject all
            _cleanup_suggestion_session(session_id)
            return {"action": "reject", "applied_count": 0}
    
    with col4:
        if st.button("🔄 Regenerate", key=f"regenerate_{session_id}"):
            # FIXED: Clean up old session before regenerating
            _cleanup_suggestion_session(session_id)
            return {"action": "regenerate", "applied_count": 0}
    
    st.markdown("---")
    
    # Individual suggestion review (Cursor-style cards)
    if operation_type == "cluster_names":
        return _review_cluster_name_suggestions(session_id, suggestions, backend)
    elif operation_type == "entry_moves":
        return _review_entry_move_suggestions(session_id, suggestions, backend)
    elif operation_type == "cluster_operations":
        return _review_cluster_operation_suggestions(session_id, suggestions, backend)
    
    return {"action": "none", "applied_count": 0}

def _cleanup_suggestion_session(session_id: str):
    """
    Clean up a suggestion session completely.
    Removes the session and all related state.
    """
    # Remove main session
    session_key = f"suggestion_session_{session_id}"
    if session_key in st.session_state:
        del st.session_state[session_key]
    
    # Remove selection tracking keys
    selection_keys = [
        f"name_selections_{session_id}",
        f"move_selections_{session_id}",
        f"operation_selections_{session_id}"
    ]
    
    for key in selection_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Remove all button keys associated with this session
    keys_to_remove = []
    for key in st.session_state.keys():
        if session_id in str(key):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        try:
            del st.session_state[key]
        except:
            pass





def _count_suggestions(suggestions: Any, operation_type: str) -> int:
    """Count total number of suggestions."""
    if operation_type == "cluster_names" and isinstance(suggestions, dict):
        return len(suggestions)
    elif operation_type == "entry_moves" and isinstance(suggestions, list):
        return len(suggestions)
    elif operation_type == "cluster_operations" and isinstance(suggestions, dict):
        return len(suggestions.get("merges", [])) + len(suggestions.get("splits", []))
    return 0


def _review_cluster_name_suggestions(session_id: str, suggestions: Dict[str, str], backend) -> Dict[str, Any]:
    """
    Review cluster name suggestions with Cursor-like cards.
    """
    st.markdown("#### 📝 Cluster Name Suggestions")
    st.caption("Review each suggestion and choose which to apply")
    
    # Track selections
    selection_key = f"name_selections_{session_id}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}
    
    applied_count = 0
    all_clusters = backend.getAllClusters()
    
    for cluster_id, suggested_name in suggestions.items():
        if cluster_id not in all_clusters:
            continue
        
        current_name = all_clusters[cluster_id]["cluster_name"]
        
        # Cursor-style card
        with st.container():
            st.markdown(f"""
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
            """, unsafe_allow_html=True)
            
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
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
            
            with col3:
                reject_key = f"reject_name_{session_id}_{cluster_id}"
                if st.button("❌ Reject", key=reject_key, use_container_width=True):
                    st.session_state[selection_key][cluster_id] = "rejected"
                    st.info(f"Rejected suggestion for {current_name}")
    
    return {"action": "partial", "applied_count": applied_count}


def _review_entry_move_suggestions(session_id: str, suggestions: List[Dict], backend) -> Dict[str, Any]:
    """
    Review entry move suggestions with Cursor-like cards.
    """
    st.markdown("#### 🔄 Entry Move Suggestions")
    st.caption("Review each move and choose which to apply")
    
    selection_key = f"move_selections_{session_id}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}
    
    applied_count = 0
    all_clusters = backend.getAllClusters()
    all_entries = backend.getAllEntries()
    
    for i, move in enumerate(suggestions):
        entry_id = move.get("entry_id")
        target_cluster = move.get("target_cluster")
        reason = move.get("reason", "No reason provided")
        
        if not entry_id or not target_cluster:
            continue
        
        entry = all_entries.get(entry_id)
        if not entry:
            continue
        
        current_cluster = entry.get("clusterID", "Unknown")
        current_cluster_name = all_clusters.get(current_cluster, {}).get("cluster_name", current_cluster)
        target_cluster_name = all_clusters.get(target_cluster, {}).get("cluster_name", target_cluster)
        
        # Cursor-style card
        with st.container():
            st.markdown(f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                background-color: #fafafa;
            ">
                <div style="font-size: 0.85rem; color: #666;">Entry: {entry_id}</div>
                <div style="margin: 8px 0; font-size: 0.9rem;">"{entry['entry_text'][:100]}..."</div>
                <div style="margin: 8px 0;">
                    <span style="color: #999;">{current_cluster_name}</span>
                    <span style="margin: 0 8px;">→</span>
                    <strong style="color: #667eea;">{target_cluster_name}</strong>
                </div>
                <div style="font-size: 0.85rem; color: #666; font-style: italic;">
                    💡 {reason}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col2:
                accept_key = f"accept_move_{session_id}_{i}"
                if st.button("✅ Accept", key=accept_key, use_container_width=True):
                    success, msg = backend.moveEntry(entry_id, target_cluster)
                    if success:
                        st.session_state[selection_key][entry_id] = "accepted"
                        applied_count += 1
                        st.success(f"✅ Moved entry to {target_cluster_name}")
                        save_finetuning_results_to_session(backend)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
            
            with col3:
                reject_key = f"reject_move_{session_id}_{i}"
                if st.button("❌ Reject", key=reject_key, use_container_width=True):
                    st.session_state[selection_key][entry_id] = "rejected"
                    st.info("Rejected move suggestion")
    
    return {"action": "partial", "applied_count": applied_count}


def _review_cluster_operation_suggestions(session_id: str, suggestions: Dict, backend) -> Dict[str, Any]:
    """
    Review cluster merge/split suggestions with Cursor-like cards.
    """
    st.markdown("#### 🔧 Cluster Operation Suggestions")
    st.caption("Review merge and split operations")
    
    selection_key = f"operation_selections_{session_id}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}
    
    applied_count = 0
    all_clusters = backend.getAllClusters()
    
    # Merges
    merges = suggestions.get("merges", [])
    if merges:
        st.markdown("**🔗 Merge Suggestions**")
        
        for i, merge in enumerate(merges):
            cluster1 = merge.get("cluster1")
            cluster2 = merge.get("cluster2")
            reason = merge.get("reason", "No reason provided")
            
            if not cluster1 or not cluster2 or cluster1 not in all_clusters or cluster2 not in all_clusters:
                continue
            
            cluster1_name = all_clusters[cluster1]["cluster_name"]
            cluster2_name = all_clusters[cluster2]["cluster_name"]
            
            # Cursor-style card
            with st.container():
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 8px 0;
                    background-color: #fafafa;
                ">
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
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col2:
                    merge_name = st.text_input(
                        "New name (optional)",
                        key=f"merge_name_{session_id}_{i}",
                        placeholder="Auto-generated"
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
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")
                
                with col4:
                    reject_key = f"reject_merge_{session_id}_{i}"
                    if st.button("❌ Reject", key=reject_key, use_container_width=True):
                        st.session_state[selection_key][f"merge_{i}"] = "rejected"
                        st.info("Rejected merge suggestion")
    
    # Splits
    splits = suggestions.get("splits", [])
    if splits:
        st.markdown("**✂️ Split Suggestions**")
        
        for i, split in enumerate(splits):
            cluster = split.get("cluster")
            reason = split.get("reason", "No reason provided")
            
            if cluster in all_clusters:
                cluster_name = all_clusters[cluster]["cluster_name"]
                
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 8px 0;
                    background-color: #fafafa;
                ">
                    <div style="margin: 8px 0;">
                        <strong style="color: #667eea;">{cluster_name}</strong>
                        <span style="margin: 0 8px;">→</span>
                        <span style="color: #666;">Multiple Clusters</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #666; font-style: italic;">
                        💡 {reason}
                    </div>
                    <div style="margin-top: 8px; padding: 8px; background-color: #fff3cd; border-radius: 4px;">
                        ⚠️ Cluster splitting requires manual re-assignment of entries
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    return {"action": "partial", "applied_count": applied_count}


def _apply_all_suggestions(session_id: str, suggestions: Any, operation_type: str, backend) -> Dict[str, Any]:
    """Apply all suggestions at once."""
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
        
        # Save changes
        save_finetuning_results_to_session(backend)
        
        return {"action": "accept_all", "applied_count": applied_count}
    
    except Exception as e:
        st.error(f"Error applying suggestions: {e}")
        return {"action": "error", "applied_count": applied_count}


# =============================================================================
# Embedded lightweight LLM wrapper (with new suggestion session methods)
# =============================================================================

class LLMWrapper:
    """LLM wrapper with built-in clustering operations support and Cursor-like review."""

    def __init__(self, backend=None):
        self.provider = None
        self.client = None
        self.config = {}
        self.initialized = False
        self.backend = backend

    def initLLM(self, provider: str = "mock", config: Dict[str, Any] | None = None) -> bool:
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
        context: Dict[str, Any] | None = None,
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
        
        # Handle structured operations
        if "suggest" in p and "name" in p and "json" in p:
            return self._mock_cluster_names_response(context)
        elif "suggest" in p and "move" in p and "json" in p:
            return self._mock_entry_moves_response(context)
        elif "suggest" in p and ("merge" in p or "split" in p) and "json" in p:
            return self._mock_cluster_operations_response(context)
        
        # Fallback to original mock responses
        if "suggest" in p and "name" in p:
            return self._mock_cluster_names_response(context)
        if "move" in p and "cluster" in p:
            return "Consider moving very short texts to Outliers—they often lack enough signal."
        if "improve" in p:
            return "Try merging overlapping small clusters, renaming with concise labels, and isolating outliers."
        return f"I understand you want help with: '{prompt}'. Here's a general suggestion..."

    def _mock_cluster_names_response(self, context) -> str:
        """Generate mock cluster name suggestions."""
        clusters_info = context.get("clusters", [])
        suggestions = {}
        
        for cluster in clusters_info:
            cid = cluster['id']
            # Use first words from sample texts
            samples = cluster.get('sample_texts', [])
            if samples:
                words = samples[0].split()[:3]
                name = ' '.join(words).title()
            else:
                name = f"Topic {cluster.get('name', cid)}"
            suggestions[cid] = name
        
        import json
        return json.dumps(suggestions)

    def _mock_entry_moves_response(self, context) -> str:
        """Generate mock entry move suggestions using REAL entry IDs."""
        import json
        
        clusters_info = context.get("clusters", [])
        if not clusters_info or len(clusters_info) < 2:
            return json.dumps([])
        
        moves = []
        cluster_ids = [c['id'] for c in clusters_info]
        
        # ✅ FIX: Get REAL entry IDs from backend
        if self.backend:
            try:
                all_entries = self.backend.getAllEntries()
                all_clusters = self.backend.getAllClusters()
                
                if not all_entries or len(all_entries) < 2:
                    return json.dumps([])
                
                # Get up to 3 REAL entry IDs
                entry_ids = list(all_entries.keys())[:min(3, len(all_entries))]
                
                # Generate moves using REAL entry IDs and cluster IDs
                for i, eid in enumerate(entry_ids):
                    entry = all_entries[eid]
                    current_cluster = entry.get("clusterID")
                    
                    # Find a different cluster to move to
                    available_clusters = [cid for cid in cluster_ids if cid != current_cluster]
                    if available_clusters:
                        target = available_clusters[i % len(available_clusters)]
                        moves.append({
                            "entry_id": str(eid),  # ✅ Real ID
                            "target_cluster": str(target),  # ✅ Real cluster
                            "reason": f"Better semantic fit with {all_clusters[target]['cluster_name']}"
                        })
            except Exception as e:
                print(f"Error generating mock moves: {e}")
                return json.dumps([])
    
        return json.dumps(moves)

    def _mock_cluster_operations_response(self, context) -> str:
        """Generate mock cluster merge/split suggestions."""
        clusters_info = context.get("clusters", [])
        if len(clusters_info) < 2:
            return json.dumps({"merges": [], "splits": []})
        
        operations = {"merges": [], "splits": []}
        
        # Suggest 1-2 merges if we have enough clusters
        if len(clusters_info) >= 2:
            operations["merges"].append({
                "cluster1": clusters_info[0]['id'],
                "cluster2": clusters_info[1]['id'],
                "reason": "Similar themes and content overlap"
            })
        
        # Suggest 1 split if cluster has many items
        for cluster in clusters_info:
            if cluster['count'] > 10:
                operations["splits"].append({
                    "cluster": cluster['id'],
                    "reason": f"Large cluster ({cluster['count']} items) may contain distinct sub-topics"
                })
                break
        
        import json
        return json.dumps(operations)

    def _extract_json(self, text: str):
        """Extract JSON from text, handling markdown code blocks."""
        import json
        import re
        
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except:
            pass
        
        # Extract from markdown code blocks
        match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # Extract first JSON-like structure
        match = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        return None

    def _build_system_message(self, context: Dict[str, Any]) -> str:
        base = (
            "You are an expert in text clustering and data analysis. "
            "Suggest cluster names, merges, and item moves. "
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

    # =============================================================================
    # NEW: Suggestion Session Management Methods
    # =============================================================================
    
    def create_suggestion_session(self, operation_type: str, params: Dict[str, Any]) -> str:
        """
        Create a new suggestion session with unique ID.
        Returns session_id for tracking suggestions.
        """
        import uuid
        session_id = f"{operation_type}_{uuid.uuid4().hex[:8]}"
        
        st.session_state[f"suggestion_session_{session_id}"] = {
            "operation_type": operation_type,
            "params": params,
            "suggestions": None,
            "status": "pending",  # pending, reviewing, applied, rejected
            "timestamp": time.time()
        }
        
        return session_id
    
    def generate_suggestions_async(self, session_id: str, context: Dict[str, Any]) -> bool:
        """
        Generate suggestions and store them for review.
        Returns True if successful.
        """
        session_data = st.session_state.get(f"suggestion_session_{session_id}")
        if not session_data:
            return False
        
        operation_type = session_data["operation_type"]
        
        try:
            if operation_type == "cluster_names":
                suggestions = self.suggest_cluster_names()
            elif operation_type == "entry_moves":
                params = session_data["params"]
                
                # Get entries to analyze based on params
                all_entries = self.backend.getAllEntries()
                entries_to_analyze = []
                
                filter_cluster = params.get("filter_cluster", "All clusters")
                min_confidence = params.get("min_confidence", 0.3)
                
                for eid, entry in all_entries.items():
                    if filter_cluster != "All clusters" and entry.get("clusterID") != filter_cluster:
                        continue
                    if entry.get("probability", 0) < min_confidence:
                        continue
                    entries_to_analyze.append(eid)
                
                suggestions = self.suggest_entry_moves(entries_to_analyze[:20])
            elif operation_type == "cluster_operations":
                suggestions = self.suggest_cluster_operations()
            else:
                return False
            
            # Store suggestions for review
            session_data["suggestions"] = suggestions
            session_data["status"] = "reviewing"
            st.session_state[f"suggestion_session_{session_id}"] = session_data
            
            return True
            
        except Exception as e:
            session_data["status"] = "error"
            session_data["error"] = str(e)
            st.session_state[f"suggestion_session_{session_id}"] = session_data
            return False

    # =============================================================================
    # Clustering Operations Methods (existing, unchanged)
    # =============================================================================

    def build_context(self, include_texts=False, max_samples=5) -> Dict[str, Any]:
        """Build context for LLM operations with optional sample texts."""
        if not self.backend:
            return {"clusters": []}
            
        clusters = self.backend.getAllClusters()
        all_entries = self.backend.getAllEntries()
        
        cluster_list = []
        for cid, c in clusters.items():
            cluster_data = {
                "id": cid, 
                "name": c["cluster_name"],
                "count": len(c["entry_ids"])
            }
            if include_texts:
                sample_texts = [all_entries[eid]["entry_text"][:200] 
                              for eid in c["entry_ids"][:max_samples] 
                              if eid in all_entries]
                cluster_data["sample_texts"] = sample_texts
            else:
                cluster_data["items"] = ["_"] * len(c["entry_ids"])
            cluster_list.append(cluster_data)
        
        return {"clusters": cluster_list}
    
    def suggest_cluster_names(self) -> Optional[Dict[str, str]]:
        """Generate cluster name suggestions."""
        if not self.initialized:
            return None
        
        context = self.build_context(include_texts=True, max_samples=3)
        prompt = """Analyze the following clusters and suggest better, more descriptive names based on their content. 
        Return a JSON object with cluster IDs as keys and suggested names as values.
        
        Example: {"cluster_0": "Customer Support Issues", "cluster_1": "Product Feedback"}
        
        Clusters to analyze:"""
        
        # Add cluster context to prompt
        clusters_info = context.get("clusters", [])
        for cluster in clusters_info:
            prompt += f"\n- {cluster['id']}: {cluster['name']} ({cluster['count']} items)"
            if cluster.get("sample_texts"):
                prompt += f"\n  Sample texts: {'; '.join(cluster['sample_texts'][:3])}"
        
        response = self.callLLM(prompt, context, temperature=0.3, max_tokens=300)
        if response:
            parsed = self._extract_json(response)
            if parsed and isinstance(parsed, dict):
                return parsed
        return None
    
    def suggest_entry_moves(self, entries_to_analyze: List[str]) -> Optional[List[Dict[str, str]]]:
        """Generate entry move suggestions."""
        if not self.initialized or not entries_to_analyze:
            return None
        
        context = self.build_context(include_texts=True, max_samples=2)
        prompt = """Analyze the following entries and suggest which cluster they should be moved to. 
        Return a JSON array with objects containing entry_id, target_cluster, and reason.
        
        Example: [{"entry_id": "001", "target_cluster": "cluster_2", "reason": "Better semantic fit"}]
        
        Available clusters:"""
        
        # Add cluster context
        clusters_info = context.get("clusters", [])
        for cluster in clusters_info:
            prompt += f"\n- {cluster['id']}: {cluster['name']} ({cluster['count']} items)"
        
        prompt += f"\n\nEntries to analyze: {', '.join(entries_to_analyze[:10])}"
        
        response = self.callLLM(prompt, context, temperature=0.4, max_tokens=500)
        if response:
            parsed = self._extract_json(response)
            if parsed and isinstance(parsed, list):
                return parsed
        return None
    
    def suggest_cluster_operations(self) -> Optional[Dict[str, Any]]:
        """Generate cluster merge/split suggestions."""
        if not self.initialized:
            return None
        
        context = self.build_context(include_texts=True, max_samples=2)
        prompt = """Analyze the following clusters and suggest:
        1. Which clusters should be merged (if any)
        2. Which clusters should be split (if any)
        
        Return JSON: {"merges": [{"cluster1": "cluster_0", "cluster2": "cluster_1", "reason": "Similar themes"}], "splits": [{"cluster": "cluster_2", "reason": "Contains distinct sub-topics"}]}
        
        Clusters to analyze:"""
        
        clusters_info = context.get("clusters", [])
        for cluster in clusters_info:
            prompt += f"\n- {cluster['id']}: {cluster['name']} ({cluster['count']} items)"
            if cluster.get("sample_texts"):
                prompt += f"\n  Sample: {'; '.join(cluster['sample_texts'][:2])}"
        
        response = self.callLLM(prompt, context, temperature=0.5, max_tokens=400)
        if response:
            parsed = self._extract_json(response)
            if parsed and isinstance(parsed, dict):
                return parsed
            else:
                return {"merges": [], "splits": []}
        return None


# Singleton helpers
_llm_instance: Optional[LLMWrapper] = None

def get_llm_wrapper() -> LLMWrapper:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMWrapper()
    return _llm_instance

def initLLM(provider: str = "mock", config: Dict[str, Any] | None = None) -> bool:
    return get_llm_wrapper().initLLM(provider, config)

def callLLM(
    prompt: str,
    context: Dict[str, Any] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> Optional[str]:
    return get_llm_wrapper().callLLM(prompt, context, temperature, max_tokens)

# Optional: simple status getters (if you want to show status elsewhere)
def get_llm_status() -> Dict[str, Any]:
    w = get_llm_wrapper()
    return {"initialized": w.initialized, "provider": w.provider, "ready": w.initialized}

def quick_llm_test() -> bool:
    resp = callLLM("Say 'Hello, I am working!' in exactly those words.", max_tokens=20)
    return resp is not None