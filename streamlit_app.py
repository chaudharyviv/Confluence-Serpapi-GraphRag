import streamlit as st
from env import validate_env
from confluence_scraper import fetch_engineering_pages
from rag_engine import index_pages, ask_engineering

validate_env()

st.set_page_config(
    page_title="Engineering RAG Assistant",
    layout="wide"
)

st.title("📘 Engineering Documentation Assistant Chatbot with Google Search")
st.caption("Local RAG • External Search • Knowledge Gap Detection")

# -------------------------------
# Badge Renderer
# -------------------------------
def render_source_badge(source_type):
    styles = {
        "local": ("📘 Local RAG (Engineering Docs)", "#1d4ed8"),
        "external": ("🌐 External Search (SerpAPI)", "#047857"),
        "none": ("⚠️ Knowledge Gap", "#b91c1c"),
    }

    label, color = styles.get(source_type, ("Unknown", "#6b7280"))

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:8px 14px;
            border-radius:999px;
            background-color:{color};
            color:white;
            font-weight:600;
            margin-bottom:12px;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Index Controls")

    force = st.checkbox("Force full re-index")
    debug = st.checkbox("Show debug info")

    if st.button("🔄 Index Engineering Pages"):
        with st.spinner("Indexing..."):
            pages = fetch_engineering_pages()
            count = index_pages(pages, force_reindex=force)
        st.success(f"Indexed {count} chunks")

# -------------------------------
# Main UI
# -------------------------------
question = st.text_input(
    "Ask a question about Ceph, NetApp, Dell EMC, or Pure Storage"
)

if st.button("Ask"):
    if question.strip():
        with st.spinner("Finding the best answer..."):
            answer, sources, source_type, debug_info = ask_engineering(
                question, debug=debug
            )

        render_source_badge(source_type)

        st.markdown("### ✅ Answer")
        st.write(answer)

        if sources:
            st.markdown("### 🔗 Sources")
            for s in sources:
                st.markdown(f"- [{s}]({s})")

        if debug and debug_info:
            st.markdown("### 🧪 Debug Info")

            if debug_info.get("local_chunks"):
                st.markdown("**Local RAG Chunks:**")
                for ch in debug_info["local_chunks"][:3]:
                    st.code(ch[:700])

            if debug_info.get("google_results"):
                st.markdown("**External Search Results:**")
                for g in debug_info["google_results"][:3]:
                    st.markdown(f"- {g['title']}")
