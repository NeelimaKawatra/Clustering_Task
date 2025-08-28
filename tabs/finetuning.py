# pip install -q streamlit streamlit-sortables
# Run with: streamlit run app.py

import streamlit as st
from collections import Counter
import re

# --- Optional dependency for drag & drop ---
try:
    # community component enabling multi-list drag & drop + container reordering
    from streamlit_sortables import sort_items  # pip install streamlit-sortables
    HAS_SORTABLES = True
except Exception:
    HAS_SORTABLES = False

st.set_page_config(page_title="HITL", layout="wide")
st.title("Human-in-the-Loop Clusters Fine-tuning 🧩")
st.caption(
    "Drag items between clusters, reorder clusters, rename/describe them, and send an instruction prompt."
)

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------

def _init_state():
    if "clusters" not in st.session_state:
        # Made-up dataset; tweak freely
        st.session_state.clusters = [
            {
                "id": "animals",
                "name": "name_1",
                "description": "llm_suggestion_1",
                "items": ["cat","dog","bird"],
            },
            {
                "id": "vehicles",
                "name": "name_2",
                "description": "llm_suggestion_2",
                "items": ["car","truck","motorcycle"],
            },
            {
                "id": "nature",
                "name": "name_3",
                "description": "llm_suggestion_3",
                "items": ["nature","forest","river"],
            },
            {
                "id": "random",
                "name": "name_4",
                "description": "llm_suggestion_4",
                "items": ["france","germany","jupter"],
            },
        ]
    if "new_item_buffers" not in st.session_state:
        st.session_state.new_item_buffers = {}


def slugify(text: str) -> str:
    if not text or not isinstance(text, str):
        return "cluster"
    
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "cluster"


def keyword_suggest(items):
    if not items or not isinstance(items, list):
        return "General"
    
    words = []
    for s in items:
        if not isinstance(s, str):
            continue
        s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
        words.extend([w.lower() for w in s.split() if len(w) > 2])
    if not words:
        return "General"
    common = Counter(words).most_common(2)
    return " ".join([w for w, _ in common]).title()


def llm_like_action(prompt: str, clusters: list[dict]):
    """A tiny, local heuristic that *mimics* an LLM suggestion.
    You can replace this with a real LLM call if you want.
    Supported micro-intents (examples):
      - "suggest a better name for cluster 1"
      - "look through cluster 2 and move items that do not fit to cluster 3"
    """
    p = prompt.lower()
    result_msgs = []

    # Suggest name for a specific cluster
    m = re.search(r"cluster\s*(\d+)", p)
    if "suggest" in p and "name" in p and m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(clusters):
            # Ensure cluster has required keys
            if "items" not in clusters[idx]:
                clusters[idx]["items"] = []
            if "name" not in clusters[idx]:
                clusters[idx]["name"] = f"Cluster {idx+1}"
            
            suggestion = keyword_suggest(clusters[idx]["items"])
            clusters[idx]["name"] = suggestion
            result_msgs.append(f"Renamed Cluster {idx+1} → '{suggestion}'.")

    # Simple rule: if prompt asks to move outliers from A to B, move 1 shortest item
    m2 = re.search(r"move .* from cluster\s*(\d+) .* to cluster\s*(\d+)", p)
    if m2:
        a, b = int(m2.group(1)) - 1, int(m2.group(2)) - 1
        if 0 <= a < len(clusters) and 0 <= b < len(clusters):
            # Ensure clusters have required keys
            if "items" not in clusters[a]:
                clusters[a]["items"] = []
            if "items" not in clusters[b]:
                clusters[b]["items"] = []
            
            if clusters[a]["items"]:
                # pick a naive "outlier": the shortest string
                src_items = clusters[a]["items"]
                shortest = min(src_items, key=len)
                src_items.remove(shortest)
                clusters[b]["items"].append(shortest)
                result_msgs.append(
                    f"Moved a likely outlier '{shortest}' from Cluster {a+1} → Cluster {b+1}."
                )

    if not result_msgs:
        result_msgs.append("No specific action understood. Try: 'Suggest a new name for Cluster 1'.")
    return result_msgs


def ensure_unique_ids(clusters: list[dict]):
    seen = set()
    for i, c in enumerate(clusters):
        # Ensure the cluster has both name and id, with fallbacks
        if "name" not in c or not c["name"]:
            c["name"] = f"Cluster {i+1}"
        if "id" not in c or not c["id"]:
            c["id"] = f"cluster_{i+1}"
        
        # Create a unique ID based on name and index
        base = slugify(c["name"])
        uid = base
        counter = 1
        while uid in seen:
            uid = f"{base}-{counter}"
            counter += 1
        
        c["id"] = uid
        seen.add(uid)


# -------------------------------------------------------------
# App body
# -------------------------------------------------------------

_init_state()
clusters = st.session_state.clusters
ensure_unique_ids(clusters)

# --- Controls moved from sidebar to main page ---
st.markdown("## Controls")
controls = st.container()
with controls:
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Add a new cluster"):
            new_idx = len(clusters) + 1
            clusters.append(
                {
                    "id": f"cluster_{new_idx}",
                    "name": f"Cluster {new_idx}",
                    "description": "",
                    "items": ["Entry A", "Entry B", "Entry C"],
                }
            )
            # Ensure all cluster IDs remain unique after adding new one
            ensure_unique_ids(clusters)
    with col_b:
        if st.button("♻️ Reset clusters"):
            for k in list(st.session_state.keys()):
                if k.startswith("_"):
                    continue
                del st.session_state[k]
            st.rerun()

    st.markdown("** LLM (where we provide direct llm access or ask user to insert their api key):**")
    prompt = st.text_input(
        "e.g., 'Suggest a new name for Cluster 1' or 'Move items from Cluster 2 to Cluster 3'",
        key="prompt_box",
    )
    run_cols = st.columns([1,3])
    if run_cols[0].button("Run instruction") and prompt.strip():
        msgs = llm_like_action(prompt, clusters)
        for m in msgs:
            st.success(m)

st.markdown("### Drag items between clusters & reorder clusters")

if HAS_SORTABLES:
    # Build containers for the component
    containers = []
    for i, c in enumerate(clusters, start=1):
        # Ensure cluster has required keys
        if "name" not in c:
            c["name"] = f"Cluster {i}"
        if "id" not in c:
            c["id"] = f"cluster_{i}"
        if "items" not in c:
            c["items"] = []
        
        containers.append(
            {
                "id": c["id"],
                "header": f"{c['name']}",  # title shown on the board
                "items": c["items"],
                "style": {"minHeight": "240px", "padding": "6px"},
            }
        )

    st.caption("Tip: you can drag entire columns to reorder clusters, and drag bullets to move entries.")

    updated = sort_items(
        containers,
        multi_containers=True,          # allow cross-container DnD
        direction="horizontal",        # containers side-by-side
        key="board",
    )

    # Apply results back to session state: order + items
    # `updated` keeps the same dict structure but with changed order / items
    # Ensure all updated containers have required keys
    for i, c in enumerate(updated):
        if "id" not in c or not c["id"]:
            c["id"] = f"updated_container_{i}"
        if "items" not in c:
            c["items"] = []
    
    new_order_ids = [c["id"] for c in updated]
    id_to_cluster = {c["id"]: c for c in clusters}

    new_clusters = []
    for c_out in updated:
        # Find the original cluster by id, with fallback
        original = id_to_cluster.get(c_out["id"])
        if original is None:
            # Create a fallback cluster if the id doesn't match
            original = {
                "id": c_out["id"],
                "name": "Unknown Cluster",
                "description": "",
                "items": []
            }
        
        new_clusters.append(
            {
                **original,
                "items": list(c_out.get("items", [])),  # updated item order
            }
        )

    st.session_state.clusters = new_clusters
    clusters = new_clusters
    # Ensure all cluster IDs remain unique after drag-and-drop operations
    ensure_unique_ids(clusters)
else:
    st.warning(
        "Drag-and-drop requires the optional component `streamlit-sortables`.\n"
        "Install it with: `pip install streamlit-sortables`. For now, use the simple controls below."
    )

    # Fallback UI: simple move-item controls
    cols = st.columns(len(clusters))
    for i, (c, col) in enumerate(zip(clusters, cols)):
        with col:
            # Ensure cluster has required keys
            if "name" not in c:
                c["name"] = f"Cluster {i+1}"
            if "id" not in c:
                c["id"] = f"cluster_{i+1}"
            
            st.subheader(c["name"])  # header
            if c["items"]:
                to_move = st.selectbox(
                    f"Pick item to move (Cluster {i+1})",
                    options=["-"] + c["items"],
                    key=f"pick_{c['id']}",
                )
                target = st.selectbox(
                    "Move to",
                    options=[f"Cluster {j+1}" for j in range(len(clusters))],
                    key=f"target_{c['id']}",
                )
                if to_move != "-" and st.button(f"Move '{to_move}' → {target}", key=f"btn_{c['id']}"):
                    c["items"].remove(to_move)
                    t_idx = int(target.split()[-1]) - 1
                    clusters[t_idx]["items"].append(to_move)
                    st.rerun()
            else:
                st.info("No items.")

# --- Per-cluster editing (name, description, add/remove entries) ---
st.markdown("---")
st.subheader("Edit clusters")

edit_cols = st.columns(len(clusters))
for i, (c, col) in enumerate(zip(clusters, edit_cols)):
    with col:
        # Ensure cluster has required keys
        if "name" not in c:
            c["name"] = f"Cluster {i+1}"
        if "id" not in c:
            c["id"] = f"cluster_{i+1}"
        if "items" not in c:
            c["items"] = []
        
        st.markdown(f"**Cluster {i+1}**")
        new_name = st.text_input("Name", value=c["name"], key=f"nm_{c['id']}")
        new_desc = st.text_area("Description", value=c.get("description", ""), key=f"ds_{c['id']}")
        if new_name != c["name"]:
            c["name"] = new_name
        if new_desc != c.get("description", ""):
            c["description"] = new_desc

        # Add item
        buf_key = f"buf_{c['id']}"
        st.session_state.new_item_buffers.setdefault(buf_key, "")
        st.text_input("Add an entry", key=buf_key, placeholder="Type and press ⏎")
        if st.session_state.new_item_buffers[buf_key].strip():
            val = st.session_state.new_item_buffers[buf_key].strip()
            if st.button("Add", key=f"add_{c['id']}"):
                c["items"].append(val)
                st.session_state.new_item_buffers[buf_key] = ""
                st.rerun()

        # Remove item
        if c["items"]:
            rm = st.multiselect("Remove entries", c["items"], key=f"rm_{c['id']}")
            if rm and st.button("Remove selected", key=f"rm_btn_{c['id']}"):
                c["items"] = [x for x in c["items"] if x not in rm]
                st.rerun()

st.success("All changes are kept in session.")

st.info("Options to export results.")
