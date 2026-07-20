"""Streamlit dashboard for GraphRAG Intelligence Engine (Phase 5)."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.client import APIClient, APIError


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state initialization ─────────────────────────────────────────────
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None
if "client" not in st.session_state:
    st.session_state.client = APIClient()
if "documents_cache" not in st.session_state:
    st.session_state.documents_cache = []


# ─── Helper functions ─────────────────────────────────────────────────────────
def get_client() -> APIClient:
    """Get or create the API client."""
    return st.session_state.client


def handle_api_error(e: APIError, context: str = "") -> None:
    """Display API error in a user-friendly way."""
    prefix = f"{context}: " if context else ""
    if e.status_code == 0:
        st.error(
            f"{prefix}Cannot connect to API at `{st.session_state.client.base_url}`. "
            "Is the backend running?"
        )
    elif e.status_code == 404:
        st.error(f"{prefix}Resource not found (404).")
    elif e.status_code >= 500:
        st.error(f"{prefix}Server error ({e.status_code}): {e.message}")
    else:
        st.error(f"{prefix}API error ({e.status_code}): {e.message}")


def refresh_documents() -> None:
    """Refresh the documents cache."""
    try:
        st.session_state.documents_cache = get_client().list_documents()
    except APIError as e:
        handle_api_error(e, "Failed to load documents")


def set_active_doc(doc_id: str | None) -> None:
    """Set the active document and clear cache if needed."""
    st.session_state.active_doc = doc_id


def format_duration(ms: int | float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.0f} ms"
    elif ms < 60000:
        return f"{ms / 1000:.1f} s"
    else:
        return f"{ms / 60000:.1f} min"


def status_badge(status: str) -> str:
    """Return a colored badge for a status string."""
    colors = {
        "success": "🟢",
        "completed": "🟢",
        "indexed": "🟢",
        "processing": "🟡",
        "pending": "🟡",
        "failed": "🔴",
        "error": "🔴",
    }
    emoji = colors.get(status.lower(), "⚪")
    return f"{emoji} {status}"


# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 GraphRAG Intelligence")
    st.caption("Phase 5 Dashboard")

    # API connection status
    api_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    st.caption(f"API: `{api_url}`")

    if st.button("🔄 Refresh Documents", use_container_width=True):
        refresh_documents()
        st.rerun()

    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        [
            "📤 Upload",
            "📄 Documents",
            "📝 Summary",
            "💬 QA Chat",
            "🕸️ Knowledge Graph",
            "⏱️ Execution Timeline",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Active document selector (shown on all pages except Upload)
    if page != "📤 Upload":
        doc_options = ["(none)"] + [
            f"{d['filename']} ({d['id'][:8]})" for d in st.session_state.documents_cache
        ]
        doc_ids = [None] + [d["id"] for d in st.session_state.documents_cache]

        current_idx = 0
        if st.session_state.active_doc:
            try:
                current_idx = doc_ids.index(st.session_state.active_doc)
            except ValueError:
                current_idx = 0

        selected_idx = st.selectbox(
            "Active Document",
            options=range(len(doc_options)),
            format_func=lambda i: doc_options[i],
            index=current_idx,
            key="active_doc_selector",
        )
        set_active_doc(doc_ids[selected_idx])

    # Show active doc info
    if st.session_state.active_doc:
        doc = next(
            (d for d in st.session_state.documents_cache if d["id"] == st.session_state.active_doc),
            None,
        )
        if doc:
            st.caption(f"📄 **Active:** {doc['filename']}")
            st.caption(f"Status: {status_badge(doc['status'])}")


# ─── Page: Upload ─────────────────────────────────────────────────────────────
if page == "📤 Upload":
    st.header("📤 Upload PDF Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a PDF to ingest into the GraphRAG engine",
    )

    if uploaded_file is not None:
        st.info(f"Selected: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

        if st.button("🚀 Ingest Document", type="primary", use_container_width=True):
            # Save to temp file and upload
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                with st.spinner("Ingesting document... this may take a moment"):
                    result = get_client().upload_pdf(tmp_path)

                st.success("✅ Document ingested successfully!")

                # Display metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Document ID", result["doc_id"][:8] + "...")
                    st.metric("Filename", result["filename"])
                    st.metric("Status", status_badge(result["status"]))
                    st.metric("Pages", result.get("num_pages", "N/A"))
                with col2:
                    st.metric("Chunks", len(result.get("chunk_ids", [])))
                    st.metric("Entities", result.get("num_entities", 0))
                    st.metric("Graph Written", "✅ Yes" if result.get("graph_written") else "❌ No")
                    if result.get("sentiment_label"):
                        st.metric(
                            "Sentiment",
                            f"{result['sentiment_label']} ({result.get('sentiment_score', 0):.2f})",
                        )

                # Show execution log if present
                if result.get("execution_log"):
                    with st.expander("📋 Execution Log", expanded=False):
                        for entry in result["execution_log"]:
                            st.write(
                                f"**{entry.get('order', '?')}. {entry.get('agent', 'unknown')}** — "
                                f"{status_badge(entry.get('status', 'unknown'))} — "
                                f"{format_duration(entry.get('duration_ms', 0))}"
                            )
                            if entry.get("detail"):
                                st.caption(entry["detail"])

                # Button to jump to timeline
                if st.button("📊 View Execution Timeline", use_container_width=True):
                    set_active_doc(result["doc_id"])
                    st.rerun()

                # Refresh documents list
                refresh_documents()

            except APIError as e:
                handle_api_error(e, "Upload failed")
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


# ─── Page: Documents ──────────────────────────────────────────────────────────
elif page == "📄 Documents":
    st.header("📄 Document Library")

    if not st.session_state.documents_cache:
        refresh_documents()

    if not st.session_state.documents_cache:
        st.info("No documents ingested yet. Go to **Upload** to add a PDF.")
    else:
        # Build dataframe for display
        import pandas as pd

        df_data = []
        for doc in st.session_state.documents_cache:
            df_data.append(
                {
                    "ID": doc["id"][:8] + "...",
                    "Filename": doc["filename"],
                    "Status": status_badge(doc["status"]),
                    "Pages": doc.get("num_pages", "—"),
                    "Language": doc.get("language", "—"),
                    "Type": doc.get("doc_type", "—"),
                    "Entities": doc.get("num_entities", 0),
                    "Graph": "✅" if doc.get("graph_written") else "❌",
                    "Uploaded": doc.get("uploaded_at", "—"),
                }
            )

        df = pd.DataFrame(df_data)

        # Display with selection
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Handle row selection
        if event.selection.rows:
            selected_idx = event.selection.rows[0]
            selected_doc = st.session_state.documents_cache[selected_idx]
            set_active_doc(selected_doc["id"])
            st.success(f"Selected: **{selected_doc['filename']}**")
            st.rerun()


# ─── Page: Summary ────────────────────────────────────────────────────────────
elif page == "📝 Summary":
    st.header("📝 Document Summary")

    if not st.session_state.active_doc:
        st.warning("Select a document from the sidebar to view its summary.")
    else:
        try:
            with st.spinner("Loading summary..."):
                result = get_client().get_summary(st.session_state.active_doc)

            summary = result.get("summary", {})
            abstractive = summary.get("abstractive", "")
            extractive = summary.get("extractive", "")

            if not abstractive and not extractive:
                st.info("No summary available for this document yet.")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("📖 Abstractive Summary", expanded=True):
                        if abstractive:
                            st.markdown(abstractive)
                        else:
                            st.caption("*No abstractive summary generated*")

                with col2:
                    with st.expander("📋 Extractive Summary", expanded=True):
                        if extractive:
                            st.markdown(extractive)
                        else:
                            st.caption("*No extractive summary generated*")

        except APIError as e:
            handle_api_error(e, "Failed to load summary")


# ─── Page: QA Chat ────────────────────────────────────────────────────────────
elif page == "💬 QA Chat":
    st.header("💬 Question Answering")

    if not st.session_state.active_doc:
        st.warning("Select a document from the sidebar, or choose 'All Documents' below.")

    # Document selector for QA (can override active doc)
    doc_options = ["All Documents (cross-doc search)"] + [
        f"{d['filename']} ({d['id'][:8]})" for d in st.session_state.documents_cache
    ]
    doc_ids = [None] + [d["id"] for d in st.session_state.documents_cache]

    # Default to active doc
    default_idx = 0
    if st.session_state.active_doc:
        try:
            default_idx = doc_ids.index(st.session_state.active_doc)
        except ValueError:
            default_idx = 0

    selected_doc_idx = st.selectbox(
        "Search scope",
        options=range(len(doc_options)),
        format_func=lambda i: doc_options[i],
        index=default_idx,
        key="qa_doc_selector",
    )
    qa_doc_id = doc_ids[selected_doc_idx]

    # Question input
    question = st.text_area(
        "Your question",
        placeholder="Ask anything about the document(s)...",
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
    with col2:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    if ask_button and question.strip():
        try:
            with st.spinner("Thinking..."):
                result = get_client().ask(question.strip(), doc_id=qa_doc_id, top_k=top_k)

            # Display answer
            st.markdown("### Answer")
            st.markdown(result.get("answer", "*No answer generated*"))

            # Display citations
            citations = result.get("citations", [])
            if citations:
                st.markdown("### Citations")

                # Build citation table
                cite_data = []
                for c in citations:
                    source = c.get("source", "vector")
                    badge = "🔵 Vector" if source == "vector" else "🟣 Graph"
                    cite_data.append(
                        {
                            "Source": badge,
                            "Chunk / Node": c.get("chunk_id") or c.get("node_ref", "—"),
                            "Score": f"{c.get('score', 0):.3f}" if c.get("score") else "—",
                            "Excerpt": c.get("text_excerpt", "")[:200] + "..."
                            if len(c.get("text_excerpt", "")) > 200
                            else c.get("text_excerpt", ""),
                        }
                    )

                import pandas as pd

                cite_df = pd.DataFrame(cite_data)
                st.dataframe(cite_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No citations returned.")

            # Model info
            if result.get("model"):
                st.caption(f"Model: {result['model']}")

        except APIError as e:
            handle_api_error(e, "QA request failed")


# ─── Page: Knowledge Graph ────────────────────────────────────────────────────
elif page == "🕸️ Knowledge Graph":
    st.header("🕸️ Knowledge Graph Viewer")

    if not st.session_state.active_doc:
        st.warning("Select a document from the sidebar to view its knowledge graph.")
    else:
        try:
            with st.spinner("Loading graph..."):
                result = get_client().get_graph(st.session_state.active_doc)

            entities = result.get("entities", [])
            relations = result.get("relations", [])
            graph_written = result.get("graph_written", False)

            if not graph_written or (not entities and not relations):
                st.info(
                    "📭 No knowledge graph available for this document yet. "
                    "Ingest a PDF with entity extraction to build the graph."
                )
            else:
                # Build nodes and edges for streamlit-agraph
                nodes = []
                edges = []

                # Color mapping for entity labels
                label_colors = {
                    "PERSON": "#FF6B6B",
                    "ORGANIZATION": "#4ECDC4",
                    "LOCATION": "#45B7D1",
                    "DATE": "#FFA07A",
                    "MONEY": "#98D8C8",
                    "PRODUCT": "#DDA0DD",
                    "EVENT": "#F0E68C",
                }
                default_color = "#A0A0A0"

                for i, ent in enumerate(entities):
                    label = ent.get("label", "ENTITY")
                    color = label_colors.get(label.upper(), default_color)
                    nodes.append(
                        Node(
                            id=ent["text"],
                            label=ent["text"],
                            size=20,
                            color=color,
                            title=f"{ent['text']} ({label})",
                        )
                    )

                for rel in relations:
                    edges.append(
                        Edge(
                            source=rel["subject"],
                            target=rel["object"],
                            label=rel["relation"],
                            color="#888888",
                        )
                    )

                # Graph config
                config = Config(
                    width=800,
                    height=600,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#FFA500",
                    collapsible=False,
                )

                # Render graph
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.caption("Click a node to see details →")
                    selected = agraph(nodes=nodes, edges=edges, config=config)

                with col2:
                    st.markdown("### Node Details")
                    if selected:
                        # selected is a list of node IDs
                        node_id = selected[0] if isinstance(selected, list) else selected
                        node_data = next((e for e in entities if e["text"] == node_id), None)
                        if node_data:
                            st.markdown(f"**Entity:** {node_data['text']}")
                            st.markdown(f"**Label:** {node_data['label']}")

                            # Show connected relations
                            connected = [
                                r
                                for r in relations
                                if r["subject"] == node_id or r["object"] == node_id
                            ]
                            if connected:
                                st.markdown("**Relations:**")
                                for r in connected:
                                    direction = "→" if r["subject"] == node_id else "←"
                                    other = r["object"] if r["subject"] == node_id else r["subject"]
                                    st.write(f"{direction} **{r['relation']}** — {other}")
                            else:
                                st.caption("No relations")
                        else:
                            st.caption("Select a node to see details")
                    else:
                        st.caption("Click a node in the graph to see its details")

                # Stats
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Entities", len(entities))
                col2.metric("Relations", len(relations))
                col3.metric("Graph Written", "✅ Yes" if graph_written else "❌ No")

        except APIError as e:
            handle_api_error(e, "Failed to load graph")


# ─── Page: Execution Timeline ─────────────────────────────────────────────────
elif page == "⏱️ Execution Timeline":
    st.header("⏱️ Agent Execution Timeline")

    if not st.session_state.active_doc:
        st.warning("Select a document from the sidebar to view its execution timeline.")
    else:
        try:
            with st.spinner("Loading timeline..."):
                result = get_client().get_timeline(st.session_state.active_doc)

            execution_log = result.get("execution_log", [])

            if not execution_log:
                st.info("No execution log available for this document.")
            else:
                # Sort by order
                execution_log.sort(key=lambda x: x.get("order", 0))

                # Display as table
                import pandas as pd

                df_data = []
                for entry in execution_log:
                    status = entry.get("status", "unknown")
                    duration = entry.get("duration_ms", 0)
                    df_data.append(
                        {
                            "Order": entry.get("order", "?"),
                            "Agent": entry.get("agent", "unknown"),
                            "Status": status_badge(status),
                            "Duration": format_duration(duration),
                            "Duration (ms)": duration,
                            "Started": entry.get("started_at", "—"),
                            "Ended": entry.get("ended_at", "—"),
                            "Detail": entry.get("detail", ""),
                        }
                    )

                df = pd.DataFrame(df_data)

                # Color-code the dataframe
                def highlight_status(row):
                    if "🔴" in row["Status"]:
                        return ["background-color: #ffebee"] * len(row)
                    elif "🟢" in row["Status"]:
                        return ["background-color: #e8f5e9"] * len(row)
                    elif "🟡" in row["Status"]:
                        return ["background-color: #fff8e1"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df.style.apply(highlight_status, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Detail": st.column_config.TextColumn("Detail", width="large"),
                    },
                )

                # Bar chart of durations by agent
                st.divider()
                st.subheader("⏱️ Duration by Agent")

                chart_data = df.groupby("Agent")["Duration (ms)"].sum().reset_index()
                if not chart_data.empty:
                    st.bar_chart(chart_data.set_index("Agent"))

                # Total pipeline time
                total_ms = df["Duration (ms)"].sum()
                st.caption(f"Total pipeline time: **{format_duration(total_ms)}**")

        except APIError as e:
            if e.status_code == 404:
                st.info("Timeline endpoint not available on the backend yet.")
            else:
                handle_api_error(e, "Failed to load timeline")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "GraphRAG Intelligence Engine — Phase 5 Dashboard | "
    f"API: `{api_url}` | "
    "Built with Streamlit & streamlit-agraph"
)


def main() -> None:
    """Entry point for the Streamlit app."""
    # This function is called when running `streamlit run frontend/app.py`
    # The actual app logic runs at module level above
    pass


if __name__ == "__main__":
    main()