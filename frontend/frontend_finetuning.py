# frontend/frontend_finetuning.py
# Finetuning tab with Drag & Drop + optional embedded LLM helper.

from __future__ import annotations

import os
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st

# Backend entrypoint
from backend.finetuning_backend import get_finetuning_backend

# --- Optional drag & drop addon (graceful fallback if missing) ---
try:
    # pip install streamlit-sortables
    from streamlit_sortables import sort_items
    _HAS_DND = True
except Exception:
    _HAS_DND = False


# =============================================================================
# MAIN TAB
# =============================================================================

def tab_finetuning(backend_available: bool):
    """Tab E: Human-in-the-Loop Fine-tuning tied to backend API"""

    # Track tab visit
    if backend_available and hasattr(st.session_state, "backend") and st.session_state.backend:
        st.session_state.backend.track_activity(
            st.session_state.session_id, "tab_visit", {"tab_name": "finetuning"}
        )

    st.header("🧩 Fine-tuning")
    st.caption("Manually adjust your clustering results, rename clusters, move entries, and export fine-tuned results.")

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

    # ---- Cluster management (rename/delete/create/merge) ----
    show_cluster_management_interface(backend)
    st.markdown("---")

    # ---- Entry management (search, inspect, move with controls) ----
    show_entry_management_interface(backend)
    st.markdown("---")

    # ---- Drag & Drop board (optional, auto-enabled if addon is installed) ----
    show_drag_drop_board(backend)
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
    text_column = st.session_state.text_column

    # Prefer user_selections to determine subject ID column (if not auto-generated)
    user_selections = st.session_state.get("user_selections", {})
    subject_id_column = None
    if not user_selections.get("id_is_auto_generated", True):
        subject_id_column = user_selections.get("id_column_choice")

    success = backend.initialize_from_clustering_results(
        clustering_results, df, text_column, subject_id_column
    )

    if success:
        st.session_state.finetuning_initialized = True
        return True

    return False


def show_cluster_management_interface(backend):
    """Cluster summary + rename/delete + create + merge."""
    st.subheader("Option 1: Cluster Management")

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
            st.metric("Manual Clusters", modification_summary.get("manual_clusters_created", 0))
        with col4:
            mod_pct = modification_summary.get("modification_percentage", 0)
            st.metric("Modified Entries %", f"{mod_pct:.1f}%")

    # Rename / delete per cluster
    all_clusters = backend.getAllClusters()
    st.markdown("**Current Clusters** (rename, check confidence, delete)")
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
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

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
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

            # Sample entries
            entries_text = backend.getAllEntriesInCluster(cluster_id)
            if entries_text:
                st.markdown("**Entries (sample):**")
                for j, text in enumerate(entries_text[:5]):
                    st.text(f"• {text[:100]}{'...' if len(text) > 100 else ''}")
                if len(entries_text) > 5:
                    st.caption(f"... and {len(entries_text) - 5} more entries")

    # Create + Merge
    current_clusters = backend.getAllClusters()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Create New Cluster**")
        new_cluster_name = st.text_input("New cluster name", placeholder="Enter cluster name")
        if st.button("Create Cluster") and new_cluster_name.strip():
            success, result = backend.createNewCluster(new_cluster_name.strip())
            if success:
                st.success(f"Created cluster: {result}")
                st.rerun()
            else:
                st.error(result)

    with col2:
        st.markdown("**Merge Clusters**")
        cluster_ids = list(current_clusters.keys())
        if len(cluster_ids) >= 2:
            cluster1 = st.selectbox(
                "First cluster",
                cluster_ids,
                key="merge_cluster1",
                format_func=lambda cid: f"{current_clusters[cid]['cluster_name']} ({len(current_clusters[cid]['entry_ids'])} entries)",
            )
            cluster2 = st.selectbox(
                "Second cluster",
                cluster_ids,
                key="merge_cluster2",
                format_func=lambda cid: f"{current_clusters[cid]['cluster_name']} ({len(current_clusters[cid]['entry_ids'])} entries)",
            )
            merge_name = st.text_input("Merged cluster name (optional)", key="merge_name")

            if st.button("Merge Clusters") and cluster1 != cluster2:
                success, result = backend.mergeClusters(cluster1, cluster2, merge_name or None)
                if success:
                    st.success(f"Merged into cluster: {result}")
                    st.rerun()
                else:
                    st.error(result)


def show_entry_management_interface(backend):
    """Search, inspect, and move a single entry with form controls."""
    st.subheader("Option 2: Text Entry Management")

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
                    format_func=lambda eid: f"{all_entries[eid]['subjectID']}: {all_entries[eid]['entry_text'][:50]}...",
                )
            else:
                st.info("No entries match your search.")
                selected_entry = None
        else:
            selected_entry = st.selectbox(
                "Select entry",
                entry_ids[:50],  # limit for performance
                format_func=lambda eid: f"{all_entries[eid]['subjectID']}: {all_entries[eid]['entry_text'][:50]}...",
            )

    with col2:
        if "selected_entry" in locals() and selected_entry:
            st.markdown("**Entry Details**")
            entry_data = backend.getEntry(selected_entry)
            if entry_data:
                st.text(f"Subject ID: {entry_data['subjectID']}")

                cluster_id = entry_data.get("clusterID", "Unassigned")
                cluster_info = all_clusters.get(cluster_id)
                if cluster_info:
                    st.text(f"Current Cluster: {cluster_info['cluster_name']} ({len(cluster_info['entry_ids'])} entries)")
                else:
                    st.text(f"Current Cluster: {cluster_id}")

                st.text(f"Confidence: {entry_data.get('probability', 0):.2f}")

                st.text_area(
                    "Complete Text Entry",
                    value=entry_data["entry_text"],
                    height=120,
                    disabled=True,
                    key=f"text_{selected_entry}",
                )

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
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)


def show_drag_drop_board(backend):
    """Drag & Drop board to move entries between clusters (optional addon)."""
    st.subheader("Option 3: Drag & Drop Board")

    if not _HAS_DND:
        st.warning("Drag & drop add-on not installed: `pip install streamlit-sortables`")
        st.info("Falling back to the manual controls above.")
        return

    clusters = backend.getAllClusters()
    if not clusters:
        st.info("No clusters to display.")
        return

    # Build original mapping: entry_id -> cluster_id
    old_cluster_of: Dict[str, str] = {}

    # Limit items per cluster for performance
    per_cluster_cap = st.number_input("Max items per cluster shown", 10, 1000, 100, 10)

    # Build containers structure for streamlit-sortables
    containers = []
    for cid, cdata in clusters.items():
        items = []
        shown = 0
        for eid in cdata["entry_ids"]:
            if shown >= per_cluster_cap:
                break
            entry = backend.getEntry(eid)
            if not entry:
                continue
            label_text = entry["entry_text"].replace("\n", " ").strip()
            # Prefix every label with the entry ID so we can parse it later
            label = f"{eid} • {label_text[:120]}{'…' if len(label_text) > 120 else ''}"
            items.append(label)
            old_cluster_of[eid] = cid
            shown += 1

        containers.append({
            "id": cid,  # keep ID to know where items end up
            "header": f"{cdata['cluster_name']} ({len(cdata['entry_ids'])} total)",
            "items": items
        })

    st.caption("Drag items between clusters. Click **Apply changes** to commit to backend.")

    result = sort_items(
    containers,
    multi_containers=True,
    direction="horizontal",          # 👈 entries laid out horizontally
    key="finetuning_dnd_board",
)

    # Compute intended moves
    pending_moves: List[tuple[str, str]] = []
    for container in result:
        new_cid = container.get("id")
        for label in container.get("items", []):
            eid = label.split(" • ", 1)[0]
            if old_cluster_of.get(eid) != new_cid:
                pending_moves.append((eid, new_cid))

    # Deduplicate
    seen = set()
    unique_moves: List[tuple[str, str]] = []
    for eid, cid in pending_moves:
        if (eid, cid) not in seen:
            seen.add((eid, cid))
            unique_moves.append((eid, cid))

    if unique_moves:
        st.info(f"Detected {len(unique_moves)} change(s).")
        if st.button("Apply changes", use_container_width=True):
            ok, fail = 0, 0
            for eid, target_cid in unique_moves:
                success, msg = backend.moveEntry(eid, target_cid)
                if success:
                    ok += 1
                else:
                    fail += 1
                    st.warning(f"Move failed for {eid}: {msg}")
            st.success(f"Applied {ok} move(s).")
            if fail:
                st.error(f"{fail} move(s) failed.")
            st.rerun()
    else:
        st.caption("No pending changes.")


def create_finetuning_report(backend) -> str:
    """Create a detailed text report of fine-tuning changes."""
    all_clusters = backend.getAllClusters()
    modification_summary = backend.getModificationSummary()

    report = f"""
FINE-TUNING SUMMARY REPORT
=========================

Generated for session: {st.session_state.get('session_id', 'unknown')}

OVERVIEW:
- Total Clusters: {len(all_clusters)}
- Manual Clusters Created: {modification_summary.get('manual_clusters_created', 0)}
- Clusters Merged: {modification_summary.get('clusters_merged', 0)}
- Total Entries: {modification_summary.get('total_entries', 0)}
- Entries in Manual Clusters: {modification_summary.get('entries_in_manual_clusters', 0)}
- Modification Percentage: {modification_summary.get('modification_percentage', 0):.1f}%

CLUSTER DETAILS:
"""
    for cluster_id, cluster_data in all_clusters.items():
        cluster_name = cluster_data["cluster_name"]
        entry_count = len(cluster_data["entry_ids"])
        is_manual = cluster_data.get("created_manually", False)

        report += f"\n{cluster_name} (ID: {cluster_id})\n"
        report += f"  - Entries: {entry_count}\n"
        report += f"  - Type: {'Manual' if is_manual else 'Original'}\n"

        if "merged_from" in cluster_data:
            report += f"  - Merged from: {', '.join(cluster_data['merged_from'])}\n"

        stats = backend.getClusterStatistics(cluster_id)
        if stats:
            report += f"  - Avg Confidence: {stats['avg_probability']:.3f}\n"
            report += f"  - Avg Text Length: {stats['avg_text_length']:.1f} chars\n"

        entries_text = backend.getAllEntriesInCluster(cluster_id)
        if entries_text:
            report += "  - Sample entries:\n"
            for i, text in enumerate(entries_text[:3]):
                report += f"    • {text[:100]}{'...' if len(text) > 100 else ''}\n"
            if len(entries_text) > 3:
                report += f"    ... and {len(entries_text) - 3} more\n"

    report += "\n" + "=" * 50
    report += "\nGenerated by Clustery Fine-tuning Module\n"
    return report


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
