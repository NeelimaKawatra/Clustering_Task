# frontend/frontend_finetuning.py
# Finetuning tab + embedded LLM wrapper + embedded LLM configuration UI.

import os
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st

# --- backend finetuning entrypoints (unchanged) ---
from backend.finetuning_backend import get_finetuning_backend


# =============================================================================
# FINETUNING TAB (your original UI)
# =============================================================================

def tab_finetuning(backend_available: bool):
    """Tab: Human-in-the-Loop Fine-tuning with proper backend API"""

    # Track tab visit
    if backend_available and hasattr(st.session_state, "backend") and st.session_state.backend:
        st.session_state.backend.track_activity(
            st.session_state.session_id, "tab_visit", {"tab_name": "finetuning"}
        )

    st.header("Fine-tuning: Manual Cluster Adjustment")
    st.caption("Manually adjust your clustering results using standardized backend functions.")

    # Optional: show AI config in sidebar (comment out if you don’t want it)
    # show_llm_config_sidebar()

    # Check prerequisites
    if not st.session_state.get("clustering_results") or not st.session_state.clustering_results.get("success", False):
        st.error("Please complete Clustering first!")
        st.info("Go to the Clustering tab and run the clustering analysis to see results here.")
        return

    # Initialize backend if needed
    if not _initialize_backend():
        st.error("Failed to initialize fine-tuning backend")
        return

    backend = get_finetuning_backend()

    # Summary
    show_finetuning_summary(backend)

    # Main interface
    show_cluster_management_interface(backend)
    show_entry_management_interface(backend)

    # Export
    #show_export_interface(backend)

    # Optional: tiny AI helper panel (uses the wrapper in this file)
    with st.expander("🤖 AI Assist (optional)"):
        col1, col2 = st.columns([2, 1])
        with col1:
            prompt = st.text_area(
                "Ask AI for help (e.g., “Suggest a better name for cluster_2 based on its texts”).",
                height=120,
                key="ft_ai_prompt",
            )
        with col2:
            temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.05, key="ft_ai_temp")
            model_hint = st.text_input("Model hint (optional)", placeholder="gpt-4o-mini / claude-3-sonnet", key="ft_ai_model_hint")

        if st.button("Ask AI"):
            ctx = _build_ai_context_for_wrapper(backend)
            # If user typed a model hint, we can set it before calling init
            if model_hint:
                # if user already initialized with another provider, we don't override provider here
                # but your wrapper chooses provider at init; you can re-init if you want.
                pass
            # The wrapper needs to be initialized via the sidebar first, or we fallback to mock:
            w = get_llm_wrapper()
            if not w.initialized:
                # safe fallback so button still works out of the box
                initLLM(provider="mock", config={"model": "mock"})
            answer = callLLM(prompt, context=ctx, temperature=temperature, max_tokens=500)
            if answer:
                st.markdown("**AI Suggestion:**")
                st.write(answer)


def _initialize_backend() -> bool:
    """Initialize fine-tuning backend from clustering results"""

    if st.session_state.get("finetuning_initialized"):
        return True

    backend = get_finetuning_backend()

    # Required data
    clustering_results = st.session_state.clustering_results
    df = st.session_state.df
    text_column = st.session_state.text_column

    # Subject ID column from user selections
    user_selections = st.session_state.get("user_selections", {})
    subject_id_column = None
    if not user_selections.get("id_is_auto_generated", True):
        subject_id_column = user_selections.get("id_column_choice")

    # Initialize backend
    success = backend.initialize_from_clustering_results(
        clustering_results, df, text_column, subject_id_column
    )

    if success:
        st.session_state.finetuning_initialized = True
        return True

    return False


def show_finetuning_summary(backend):
    """Show summary of current clustering state"""

    with st.expander("Clustering Summary", expanded=True):
        all_clusters = backend.getAllClusters()
        all_entries = backend.getAllEntries()
        modification_summary = backend.getModificationSummary()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Clusters", len(all_clusters))
        with col2:
            st.metric("Total Entries", len(all_entries))
        with col3:
            st.metric("Manual Clusters", modification_summary.get("manual_clusters_created", 0))
        with col4:
            mod_pct = modification_summary.get("modification_percentage", 0)
            st.metric("Modified %", f"{mod_pct:.1f}%")


def show_cluster_management_interface(backend):
    """Interface for managing clusters"""

    st.subheader("Cluster Management")

    all_clusters = backend.getAllClusters()

    # Cluster operations
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
        cluster_ids = list(all_clusters.keys())
        if len(cluster_ids) >= 2:
            cluster1 = st.selectbox("First cluster", cluster_ids, key="merge_cluster1")
            cluster2 = st.selectbox("Second cluster", cluster_ids, key="merge_cluster2")
            merge_name = st.text_input("Merged cluster name (optional)", key="merge_name")

            if st.button("Merge Clusters") and cluster1 != cluster2:
                success, result = backend.mergeClusters(cluster1, cluster2, merge_name or None)
                if success:
                    st.success(f"Merged into cluster: {result}")
                    st.rerun()
                else:
                    st.error(result)

    # Display clusters
    st.markdown("**Current Clusters**")

    for cluster_id, cluster_data in all_clusters.items():
        with st.expander(
            f"🗂️ {cluster_data['cluster_name']} ({len(cluster_data['entry_ids'])} entries)", expanded=False
        ):
            # Cluster name editing
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                new_name = st.text_input(
                    "Cluster name", value=cluster_data["cluster_name"], key=f"name_{cluster_id}"
                )

                if new_name != cluster_data["cluster_name"]:
                    if st.button("Update Name", key=f"update_name_{cluster_id}"):
                        success, message = backend.changeClusterName(cluster_id, new_name)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

            with col2:
                # Show cluster statistics
                stats = backend.getClusterStatistics(cluster_id)
                if stats:
                    st.metric("Avg Confidence", f"{stats['avg_probability']:.2f}")

            with col3:
                # Delete cluster option
                if st.button("Delete", key=f"delete_{cluster_id}"):
                    success, message = backend.deleteCluster(cluster_id)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

            # Show entries in cluster
            entries_text = backend.getAllEntriesInCluster(cluster_id)
            if entries_text:
                st.markdown("**Entries:**")
                for i, text in enumerate(entries_text[:5]):  # Show first 5
                    st.text(f"• {text[:100]}{'...' if len(text) > 100 else ''}")

                if len(entries_text) > 5:
                    st.caption(f"... and {len(entries_text) - 5} more entries")


def show_entry_management_interface(backend):
    """Interface for managing individual entries"""

    st.subheader("Entry Management")

    all_entries = backend.getAllEntries()
    all_clusters = backend.getAllClusters()

    if not all_entries:
        st.info("No entries available")
        return

    # Entry search and selection
    entry_ids = list(all_entries.keys())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Find Entry**")

        # Search by text content
        search_text = st.text_input("Search in entry text", placeholder="Type to search...")

        if search_text:
            matching_entries = []
            for entry_id, entry_data in all_entries.items():
                if search_text.lower() in entry_data["entry_text"].lower():
                    matching_entries.append(entry_id)

            if matching_entries:
                selected_entry = st.selectbox(
                    "Matching entries",
                    matching_entries,
                    format_func=lambda x: f"{x}: {all_entries[x]['entry_text'][:50]}...",
                )
            else:
                st.info("No entries match your search")
                selected_entry = None
        else:
            # Show all entries (limit for perf)
            selected_entry = st.selectbox(
                "Select entry",
                entry_ids[:50],
                format_func=lambda x: f"{x}: {all_entries[x]['entry_text'][:50]}...",
            )

    with col2:
        if "selected_entry" in locals() and selected_entry:
            st.markdown("**Entry Details**")

            entry_data = backend.getEntry(selected_entry)
            if entry_data:
                st.text(f"Entry ID: {entry_data['entryID']}")
                st.text(f"Subject ID: {entry_data.get('subjectID', 'N/A')}")
                st.text(f"Current Cluster: {entry_data.get('clusterID', 'Unassigned')}")
                st.text(f"Confidence: {entry_data.get('probability', 0):.2f}")

                st.markdown("**Text:**")
                st.text_area("", value=entry_data["entry_text"], height=100, disabled=True, key=f"text_{selected_entry}")

                # Move entry
                st.markdown("**Move Entry**")
                cluster_options = list(all_clusters.keys())
                current_cluster = entry_data.get("clusterID")

                if current_cluster in cluster_options:
                    current_index = cluster_options.index(current_cluster)
                else:
                    current_index = 0

                target_cluster = st.selectbox(
                    "Move to cluster",
                    cluster_options,
                    index=current_index,
                    format_func=lambda x: f"{all_clusters[x]['cluster_name']} ({len(all_clusters[x]['entry_ids'])} entries)",
                    key=f"move_{selected_entry}",
                )

                if target_cluster != current_cluster:
                    if st.button(
                        f"Move to {all_clusters[target_cluster]['cluster_name']}", key=f"move_btn_{selected_entry}"
                    ):
                        success, message = backend.moveEntry(selected_entry, target_cluster)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)


def show_export_interface(backend):
    """Interface for exporting fine-tuned results"""

    st.subheader("Export Fine-tuned Results")

    modification_summary = backend.getModificationSummary()

    # Show modification summary
    with st.expander("Modification Summary", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Manual Clusters Created", modification_summary.get("manual_clusters_created", 0))
        with col2:
            st.metric("Clusters Merged", modification_summary.get("clusters_merged", 0))
        with col3:
            st.metric("Entries Modified", modification_summary.get("entries_in_manual_clusters", 0))

    # Export options
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export to CSV", use_container_width=True):
            df = st.session_state.df
            text_column = st.session_state.text_column
            user_selections = st.session_state.get("user_selections", {})
            subject_id_column = None

            if not user_selections.get("id_is_auto_generated", True):
                subject_id_column = user_selections.get("id_column_choice")

            export_df = backend.exportFineTunedResults(df, text_column, subject_id_column)

            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "Download Fine-tuned Results CSV",
                csv_data,
                "finetuned_clustering_results.csv",
                "text/csv",
                use_container_width=True,
            )

    with col2:
        if st.button("📋 Export Summary Report", use_container_width=True):
            report = create_finetuning_report(backend)
            st.download_button(
                "Download Summary Report", report, "finetuning_summary_report.txt", "text/plain", use_container_width=True
            )

    with col3:
        if st.button("🔄 Reset to Original", use_container_width=True):
            if st.session_state.get("confirm_reset"):
                # Actually reset
                st.session_state.finetuning_initialized = False
                del st.session_state["confirm_reset"]
                st.success("Reset to original clustering results")
                st.rerun()
            else:
                # Ask for confirmation
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")


def create_finetuning_report(backend) -> str:
    """Create a detailed text report of fine-tuning changes"""

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

        # Stats
        stats = backend.getClusterStatistics(cluster_id)
        if stats:
            report += f"  - Avg Confidence: {stats['avg_probability']:.3f}\n"
            report += f"  - Avg Text Length: {stats['avg_text_length']:.1f} chars\n"

        # Sample entries
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
    """Make a small context dict for LLM suggestions."""
    clusters = backend.getAllClusters()
    # Drop large stuff; keep names and counts
    cluster_list = [
        {"id": cid, "name": c["cluster_name"], "items": ["_"] * len(c["entry_ids"])}
        for cid, c in clusters.items()
    ]
    return {"clusters": cluster_list}


# =============================================================================
# EMBEDDED LLM WRAPPER  (originally utils/llm_wrapper.py)
# =============================================================================

class LLMWrapper:
    """Wrapper for accessing different LLM APIs without direct API calls"""

    def __init__(self):
        self.provider = None
        self.client = None
        self.config = {}
        self.initialized = False

    def initLLM(self, provider: str = "openai", config: Dict[str, Any] = None) -> bool:
        """Initialize LLM connection

        Args:
            provider: "openai", "anthropic", "local", "mock"
            config: Provider-specific configuration
        """
        try:
            self.provider = provider.lower()
            self.config = config or {}

            if self.provider == "openai":
                return self._init_openai()
            elif self.provider == "anthropic":
                return self._init_anthropic()
            elif self.provider == "local":
                return self._init_local()
            elif self.provider == "mock":
                return self._init_mock()
            else:
                st.error(f"Unsupported LLM provider: {provider}")
                return False

        except Exception as e:
            st.error(f"LLM initialization failed: {e}")
            return False

    def _init_openai(self) -> bool:
        """Initialize OpenAI client"""
        try:
            import openai  # type: ignore

            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                return False

            # New-style client (openai>=1.0 uses openai.OpenAI)
            self.client = openai.OpenAI(api_key=api_key)
            self.initialized = True
            return True

        except ImportError:
            st.error("OpenAI package not installed. Run: pip install openai")
            return False

    def _init_anthropic(self) -> bool:
        """Initialize Anthropic client"""
        try:
            import anthropic  # type: ignore

            api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
                return False

            self.client = anthropic.Anthropic(api_key=api_key)
            self.initialized = True
            return True

        except ImportError:
            st.error("Anthropic package not installed. Run: pip install anthropic")
            return False

    def _init_local(self) -> bool:
        """Initialize local LLM (placeholder for local models)"""
        st.info("Local LLM support coming soon")
        self.initialized = True
        return True

    def _init_mock(self) -> bool:
        """Initialize mock LLM for testing"""
        self.initialized = True
        return True

    def callLLM(
        self,
        prompt: str,
        context: Dict[str, Any] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Optional[str]:
        """Generic LLM call function"""
        if not self.initialized:
            st.error("LLM not initialized. Use the AI Configuration panel or initLLM().")
            return None

        try:
            if self.provider == "openai":
                return self._call_openai(prompt, context, temperature, max_tokens)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt, context, temperature, max_tokens)
            elif self.provider == "local":
                return self._call_local(prompt, context, temperature, max_tokens)
            elif self.provider == "mock":
                return self._call_mock(prompt, context, temperature, max_tokens)
            else:
                return None

        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return None

    def _call_openai(self, prompt: str, context: Dict[str, Any], temperature: float, max_tokens: int) -> str:
        system_message = self._build_system_message(context)
        resp = self.client.chat.completions.create(
            model=self.config.get("model", "gpt-3.5-turbo"),
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def _call_anthropic(self, prompt: str, context: Dict[str, Any], temperature: float, max_tokens: int) -> str:
        system_message = self._build_system_message(context)
        full_prompt = f"{system_message}\n\nUser: {prompt}"
        resp = self.client.completions.create(
            model=self.config.get("model", "claude-3-sonnet-20240229"),
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.completion

    def _call_local(self, prompt: str, context: Dict[str, Any], temperature: float, max_tokens: int) -> str:
        return "Local LLM response (not implemented)"

    def _call_mock(self, prompt: str, context: Dict[str, Any], temperature: float, max_tokens: int) -> str:
        p = prompt.lower()
        if "suggest" in p and "name" in p:
            return "Based on the cluster content, a clearer name might be “Topic Analysis”."
        elif "move" in p and "cluster" in p:
            return "Consider moving very short texts to Outliers—they often lack enough signal."
        elif "improve" in p:
            return "You could: (1) merge small highly-overlapping clusters, (2) rename clusters with concise labels, (3) isolate outliers."
        else:
            return f"I understand you want help with: '{prompt}'. Here’s a general suggestion..."

    def _build_system_message(self, context: Dict[str, Any]) -> str:
        base = (
            "You are an expert in text clustering and data analysis. "
            "You help users improve their clustering results by suggesting cluster names, "
            "moves, merges, and insight. Be concise and actionable."
        )
        if context:
            clusters_info = context.get("clusters", [])
            if clusters_info:
                base += f"\n\nCurrent clusters: {len(clusters_info)}"
                for i, cluster in enumerate(clusters_info[:5]):
                    name = cluster.get("name", f"Cluster {i+1}")
                    item_count = len(cluster.get("items", []))
                    base += f"\n- {name}: {item_count} items"
        return base


# module-level singleton
_llm_instance = None

def get_llm_wrapper() -> LLMWrapper:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMWrapper()
    return _llm_instance

def initLLM(provider: str = "mock", config: Dict[str, Any] = None) -> bool:
    wrapper = get_llm_wrapper()
    return wrapper.initLLM(provider, config)

def callLLM(
    prompt: str, context: Dict[str, Any] = None, temperature: float = 0.7, max_tokens: int = 500
) -> Optional[str]:
    wrapper = get_llm_wrapper()
    return wrapper.callLLM(prompt, context, temperature, max_tokens)


# =============================================================================
# EMBEDDED LLM CONFIG UI  (originally utils/llm_config.py)
# =============================================================================

def show_llm_config_sidebar():
    """Show LLM configuration in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 AI Configuration")

        provider = st.selectbox(
            "AI Provider",
            ["mock", "openai", "anthropic", "local"],
            help="Choose your AI provider",
            key="llm_provider",
        )

        # API Keys for real providers
        if provider in ["openai", "anthropic"]:
            api_key_env = f"{provider.upper()}_API_KEY"
            existing_key = os.getenv(api_key_env)

            if existing_key:
                st.success(f"✅ {provider.title()} API key found in environment")
                show_key = st.checkbox("Show/change API key")
                if show_key:
                    _ = st.text_input(
                        f"{provider.title()} API Key",
                        value=existing_key[:8] + "..." if existing_key else "",
                        type="password",
                        key=f"{provider}_api_key_input",
                    )
            else:
                st.warning(f"⚠️ No {provider.title()} API key found")
                new_key = st.text_input(
                    f"Enter {provider.title()} API Key",
                    type="password",
                    help=f"Set {api_key_env} environment variable or enter here",
                    key=f"{provider}_api_key_input",
                )
                if new_key:
                    st.session_state[f"{provider}_api_key"] = new_key

        # Model selection
        if provider == "openai":
            model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"], help="Choose OpenAI model")
        elif provider == "anthropic":
            model = st.selectbox(
                "Model",
                ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                help="Choose Anthropic model",
            )
        else:
            model = "default"

        temperature = st.slider("Creativity", 0.0, 1.0, 0.7, help="Higher = more creative, Lower = more focused")

        if st.button("🔧 Initialize AI", use_container_width=True):
            config = {"model": model, "temperature": temperature}
            if provider in ["openai", "anthropic"]:
                session_key = st.session_state.get(f"{provider}_api_key")
                if session_key:
                    config["api_key"] = session_key

            success = initLLM(provider, config)
            if success:
                st.success(f"✅ {provider.title()} initialized")
                st.session_state.llm_ready = True
            else:
                st.error("❌ Initialization failed")
                st.session_state.llm_ready = False

        # Status
        wrapper = get_llm_wrapper()
        if wrapper.initialized:
            st.info(f"🟢 AI Ready: {wrapper.provider}")
        else:
            st.info("🔴 AI Not Initialized")


def get_llm_status() -> Dict[str, Any]:
    wrapper = get_llm_wrapper()
    return {"initialized": wrapper.initialized, "provider": wrapper.provider, "ready": st.session_state.get("llm_ready", False)}

def quick_llm_test() -> bool:
    resp = callLLM("Say 'Hello, I am working!' in exactly those words.", max_tokens=20)
    return resp is not None
