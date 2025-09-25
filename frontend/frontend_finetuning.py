# frontend/frontend_finetuning.py
# Finetuning tab with Drag & Drop + optional embedded LLM helper.

from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple

import re
import streamlit as st

# Backend entrypoint
from backend.finetuning_backend import get_finetuning_backend

# --- Optional drag & drop addon (graceful fallback if missing) ---
try:
    # pip install streamlit-sortables
    from streamlit_sortables import sort_items
except Exception:
    pass


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

    show_cluster_management_interface(backend)
    show_drag_drop_board(backend)
    st.markdown("---")

    show_entry_management_interface(backend)
    st.markdown("---")

    # ---- Optional AI assist (uses lightweight embedded wrapper) ----
    with st.expander("🤖 AI Assist (optional)"):
        col1, col2 = st.columns([2, 1])
        with col1:
            prompt = st.text_area(
                "Ask AI for help (e.g., “Suggest a clearer name for cluster_2 based on its texts”).",
                height=120,
                key="ft_ai_prompt",
            )
        with col2:
            temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.05, key="ft_ai_temp")
            model_hint = st.text_input(
                "Model hint (optional)",
                placeholder="gpt-4o-mini / claude-3-sonnet",
                key="ft_ai_model_hint",
            )

        if st.button("Ask AI", use_container_width=True):
            ctx = _build_ai_context_for_wrapper(backend)
            # If wrapper isn't initialized, default to mock so the button works out-of-the-box
            w = get_llm_wrapper()
            if not w.initialized:
                initLLM(provider="mock", config={"model": "mock"})
            answer = callLLM(prompt, context=ctx, temperature=temperature, max_tokens=500)
            if answer:
                st.markdown("**AI Suggestion:**")
                st.write(answer)


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

    text_column = st.session_state.text_column
    
    success = backend.initialize_from_clustering_results(
        clustering_results, df, text_column, subject_id_column
    )

    if success:
        st.session_state.finetuning_initialized = True
        save_finetuning_results_to_session(backend)
        return True

    return False


def show_cluster_management_interface(backend):
    """Cluster summary + rename/delete + create + merge."""

    # Display any pending success/error messages
    if "finetuning_success_message" in st.session_state:
        st.success(st.session_state.finetuning_success_message)
        del st.session_state.finetuning_success_message
    
    if "finetuning_error_message" in st.session_state:
        st.error(st.session_state.finetuning_error_message)
        del st.session_state.finetuning_error_message

    # Summary
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
            manual_clusters = modification_summary.get("manual_clusters_created", 0)
            st.metric("Manual Clusters", manual_clusters)
        with col4:
            modification_pct = modification_summary.get("modification_percentage", 0)
            st.metric("Modified Entries", f"{modification_pct:.1f}%")

    # Create + Merge
    with st.expander("💡 You can create new clusters or merge existing clusters:"):
        current_clusters = backend.getAllClusters()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Create New Cluster**")
            new_cluster_name = st.text_input("New cluster name", placeholder="Type here...")
            if st.button("Create New Cluster") and new_cluster_name.strip():
                success, result = backend.createNewCluster(new_cluster_name.strip())
                if success:
                    st.session_state.finetuning_success_message = f"✅ Created cluster: '{result}'"
                    save_finetuning_results_to_session(backend)
                    st.rerun()
                else:
                    st.session_state.finetuning_error_message = result
                    st.rerun()

        with col2:
            st.markdown("**Merge Clusters**")
            cluster_ids = list(current_clusters.keys())
            if len(cluster_ids) >= 2:
                # Create display options for selectbox
                cluster_options = []
                for cid in cluster_ids:
                    cluster_name = current_clusters[cid]['cluster_name']
                    entry_count = len(current_clusters[cid]['entry_ids'])
                    cluster_options.append(f"{cluster_name} ({entry_count} entries)")
                
                cluster1_idx = st.selectbox(
                    "First cluster",
                    range(len(cluster_options)),
                    key="merge_cluster1",
                    format_func=lambda x: cluster_options[x],
                )
                cluster2_idx = st.selectbox(
                    "Second cluster",
                    range(len(cluster_options)),
                    key="merge_cluster2",
                    format_func=lambda x: cluster_options[x],
                )
                
                cluster1 = cluster_ids[cluster1_idx]
                cluster2 = cluster_ids[cluster2_idx]
                merge_name = st.text_input("Merged cluster name (optional)", key="merge_name")

                if st.button("Merge Clusters") and cluster1 != cluster2:
                    success, result = backend.mergeClusters(cluster1, cluster2, merge_name or None)
                    if success:
                        st.session_state.finetuning_success_message = f"🔄 Merged into cluster: '{result}'"
                        save_finetuning_results_to_session(backend)
                        st.rerun()
                    else:
                        st.session_state.finetuning_error_message = result
                        st.rerun()
    
    # Rename / delete per cluster
    all_clusters = backend.getAllClusters()
    with st.expander("💡 You can rename, check confidence, and delete clusters:"):
        for i, (cluster_id, cluster_data) in enumerate(all_clusters.items()):
            key_prefix = f"{cluster_id}_{i}"
            with st.expander(
                f"🗂️ {cluster_data['cluster_name']} ({len(cluster_data['entry_ids'])} entries)",
                expanded=False,
            ):
                col1, col2, col3 = st.columns([2, 1, 1])

                # Rename
                with col1:
                    new_name = st.text_input(
                        "Cluster name", value=cluster_data["cluster_name"], key=f"name_{key_prefix}"
                    )
                    changed = new_name.strip() != cluster_data["cluster_name"]
                    clicked = st.button(
                        "Update Name", key=f"update_name_{key_prefix}", disabled=not changed
                    )
                    if clicked:
                        success, message = backend.changeClusterName(cluster_id, new_name.strip())
                        if success:
                            st.session_state.finetuning_success_message = f"✏️ {message}"
                            save_finetuning_results_to_session(backend)
                            st.rerun()
                        else:
                            st.session_state.finetuning_error_message = message
                            st.rerun()

                # Stats
                with col2:
                    stats = backend.getClusterStatistics(cluster_id)
                    if stats:
                        st.metric("Avg Confidence", f"{stats['avg_probability']:.2f}")

                # Delete
                with col3:
                    if st.button("Delete", key=f"delete_{key_prefix}"):
                        success, message = backend.deleteCluster(cluster_id)
                        if success:
                            st.session_state.finetuning_success_message = f"🗑️ {message}"
                            save_finetuning_results_to_session(backend)
                            st.rerun()
                        else:
                            st.session_state.finetuning_error_message = message
                            st.rerun()


def show_drag_drop_board(backend):
    """
    Drag & Drop board for cluster reassignment.
    """

    #  fetch fresh clusters
    clusters: Dict[str, dict] = backend.getAllClusters()
    if not clusters:
        st.info("No clusters to display.")
        return

    # Build authoritative mapping and containers
    # old_cluster_of: entryID -> clusterID
    old_cluster_of: Dict[str, str] = {}
    containers: List[dict] = []
    orig_container_ids: List[str] = []

    for cluster_id, cdata in clusters.items():
        # Expect cdata to include 'cluster_name' and 'entry_ids'
        entry_ids: List[str] = []
        items: List[str] = []
        for eid in cdata.get("entry_ids", []):
            entry = backend.getEntry(eid)
            if not entry:
                continue
            text = (entry.get("entry_text") or "").replace("\n", " ").strip()
            disp = text[:120] + ("…" if len(text) > 120 else "")
            # Visual label ONLY; logic will extract eid from the left side before ':'
            items.append(f"{eid}: {disp}")
            entry_ids.append(eid)
            old_cluster_of[eid] = cluster_id
        containers.append({
            "id": cluster_id,  # target clusterID
            "header": f"{cdata.get('cluster_name', cluster_id)} ({len(entry_ids)} entries)",
            "items": items
        })
        orig_container_ids.append(cluster_id)

    with st.expander("💡 You can drag entries across clusters. Click **Apply changes** to commit."):
        # display the DnD board
        result = sort_items(
            containers,
            multi_containers=True,
            direction="horizontal",
            #key (must not have any key parameter)
        )

        # Helper: robustly extract entryID from 'entryID: text'
        def _eid_from_item(item: Any) -> Optional[str]:
            if not isinstance(item, str) or ":" not in item:
                return None
            return item.split(":", 1)[0].strip()

        # Compute moves: (entryID -> new_cluster_id)
        pending_moves: List[Tuple[str, str]] = []
        for i, container in enumerate(result):
            new_cid = container.get("id") or (orig_container_ids[i] if i < len(orig_container_ids) else None)
            if not new_cid:
                continue
            for item in container.get("items", []):
                eid = _eid_from_item(item)
                if not eid:
                    continue
                if old_cluster_of.get(eid) != new_cid:
                    pending_moves.append((eid, new_cid))

        # Apply changes
        if pending_moves:
            if st.button("Apply changes", use_container_width=True):
                ok  = 0
                for eid, target_cid in pending_moves:
                    success, _ = backend.moveEntry(eid, target_cid)
                    ok += int(success)

                st.session_state.finetuning_success_message = f"✅ Applied {ok} change(s)."
                # save the changes
                save_finetuning_results_to_session(backend)

                # Blow away caches/snapshots if you use them anywhere
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass

                # Nudge any cached helpers
                st.session_state["finetuning_refresh_token"] = st.session_state.get("finetuning_refresh_token", 0) + 1

                # Hard refresh so all sections re-read backend state
                st.rerun()
        else:
            st.caption("No pending changes.")


def show_entry_management_interface(backend):
    """Search, inspect, and move a single entry with form controls."""
    with st.expander("💡 You can search, inspect, and move a single entry with form controls:"):

        all_entries = backend.getAllEntries()
        all_clusters = backend.getAllClusters()

        if not all_entries:
            st.info("No entries available.")
            return

        entry_ids = list(all_entries.keys())

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Find Entry**")
            search_text = st.text_input("Search in entry text", placeholder="Type to search...")

            if search_text:
                matching_entries = [
                    eid for eid, ed in all_entries.items()
                    if search_text.lower() in ed["entry_text"].lower()
                ]
                if matching_entries:
                    selected_entry = st.selectbox(
                        "Matching entries",
                        matching_entries,
                        format_func=lambda eid: f"{all_entries[eid]['subjectID']}: {all_entries[eid]['entry_text']}",
                    )
                else:
                    st.info("No entries match your search.")
                    selected_entry = None
            else:
                selected_entry = st.selectbox(
                    "Select entry",
                    entry_ids,
                    format_func=lambda eid: f"{all_entries[eid]['subjectID']}: {all_entries[eid]['entry_text']}",
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
                    )

                    if target_cluster != current_cluster:
                        if st.button(
                            f"Move to {all_clusters[target_cluster]['cluster_name']}",
                            key=f"move_btn_{selected_entry}",
                        ):
                            success, message = backend.moveEntry(selected_entry, target_cluster)
                            if success:
                                st.session_state.finetuning_success_message = f"📦 {message}"
                                save_finetuning_results_to_session(backend)
                                st.rerun()
                            else:
                                st.session_state.finetuning_error_message = message
                                st.rerun()


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
    st.session_state.finetuning_results = build_finetuning_results_snapshot(backend)


def _build_ai_context_for_wrapper(backend) -> Dict[str, Any]:
    """Small context dict for LLM suggestions (names + counts only)."""
    clusters = backend.getAllClusters()
    cluster_list = [
        {"id": cid, "name": c["cluster_name"], "items": ["_"] * len(c["entry_ids"])}
        for cid, c in clusters.items()
    ]
    return {"clusters": cluster_list}


# =============================================================================
# Embedded lightweight LLM wrapper (optional)
# =============================================================================

class LLMWrapper:
    """Tiny wrapper to abstract LLM providers without hard dependency."""

    def __init__(self):
        self.provider = None
        self.client = None
        self.config = {}
        self.initialized = False

    def initLLM(self, provider: str = "mock", config: Dict[str, Any] | None = None) -> bool:
        try:
            self.provider = (provider or "mock").lower()
            self.config = config or {}
            if self.provider == "openai":
                return self._init_openai()
            if self.provider == "anthropic":
                return self._init_anthropic()
            if self.provider == "local":
                return self._init_local()
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

    def _init_anthropic(self) -> bool:
        try:
            import anthropic  # type: ignore
            api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("Anthropic API key not found. Set ANTHROPIC_API_KEY.")
                return False
            self.client = anthropic.Anthropic(api_key=api_key)
            self.initialized = True
            return True
        except ImportError:
            st.error("Anthropic package not installed. Run: pip install anthropic")
            return False

    def _init_local(self) -> bool:
        st.info("Local LLM support (placeholder).")
        self.initialized = True
        return True

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
            if self.provider == "anthropic":
                return self._call_anthropic(prompt, context or {}, temperature, max_tokens)
            if self.provider == "local":
                return self._call_local(prompt, context or {}, temperature, max_tokens)
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

    def _call_anthropic(self, prompt, context, temperature, max_tokens) -> str:
        sysmsg = self._build_system_message(context)
        full = f"{sysmsg}\n\nUser: {prompt}"
        resp = self.client.completions.create(
            model=self.config.get("model", "claude-3-sonnet-20240229"),
            prompt=full,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.completion

    def _call_local(self, prompt, context, temperature, max_tokens) -> str:
        return "Local LLM response (not implemented)"

    def _call_mock(self, prompt, context, temperature, max_tokens) -> str:
        p = prompt.lower()
        if "suggest" in p and "name" in p:
            return "Based on the cluster content, a clearer name might be “Topic Analysis”."
        if "move" in p and "cluster" in p:
            return "Consider moving very short texts to Outliers—they often lack enough signal."
        if "improve" in p:
            return "Try merging overlapping small clusters, renaming with concise labels, and isolating outliers."
        return f"I understand you want help with: “{prompt}”. Here’s a general suggestion..."

    def _build_system_message(self, context: Dict[str, Any]) -> str:
        base = (
            "You are an expert in text clustering and data analysis. "
            "Suggest cluster names, merges, and item moves. Be concise and actionable."
        )
        clusters_info = (context or {}).get("clusters", [])
        if clusters_info:
            base += f"\n\nCurrent clusters: {len(clusters_info)}"
            for i, cluster in enumerate(clusters_info[:5]):
                name = cluster.get("name", f"Cluster {i+1}")
                item_count = len(cluster.get("items", []))
                base += f"\n- {name}: {item_count} items"
        return base


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
