import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title="ScriptureRAG",
    page_icon="🕉️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Clean ChatGPT Styling
st.markdown("""
<style>
    /* Dark Theme */
    .stApp {
        background-color: #212121;
        color: #ECECEC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Centered Title */
    .hero-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 10vh;
        margin-bottom: 2rem;
    }
    
    /* Chat message spacing */
    .stChatMessage {
        background-color: transparent !important;
        padding: 0.8rem 0rem;
    }
    div[data-testid="stChatMessageContent"] {
        color: #ECECEC !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Cache Pipeline Initialization
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    from src.guardrails import MahapuranaGuardrail
    from src.query_processor import ScriptureQueryProcessor
    from src.vector_db import ScriptureVectorDB
    from src.reranker import ScriptureReranker
    from src.llm_engine import ScriptureLLMEngine
    from src.validation import ScriptureOutputGuardrail

    tmp_chroma = "/tmp/chroma_db"
    parts_dir = "data/chroma_db_parts"
    sqlite_target = os.path.join(tmp_chroma, "chroma.sqlite3")

    # If /tmp/chroma_db is missing or empty, stitch split parts together
    if os.path.exists(parts_dir) and (not os.path.exists(sqlite_target) or os.path.getsize(sqlite_target) < 1000000):
        os.makedirs(tmp_chroma, exist_ok=True)
        print("[VectorDB] Reconstructing ChromaDB from split parts...")
        part_files = sorted([f for f in os.listdir(parts_dir) if f.startswith("part_")])
        with open(sqlite_target, "wb") as outfile:
            for part in part_files:
                part_path = os.path.join(parts_dir, part)
                with open(part_path, "rb") as infile:
                    outfile.write(infile.read())
        print(f"[VectorDB] Reconstructed database size: {os.path.getsize(sqlite_target)/(1024*1024):.2f} MB")

    chroma_dir = tmp_chroma if os.path.exists(tmp_chroma) else "data/chroma_db"
    vector_db = ScriptureVectorDB(persist_dir=chroma_dir)

    return {
        "guardrail": MahapuranaGuardrail,
        "processor": ScriptureQueryProcessor,
        "vector_db": vector_db,
        "reranker": ScriptureReranker(),
        "llm": ScriptureLLMEngine(),
        "validator": ScriptureOutputGuardrail()
    }

with st.spinner("🕉️ Initializing ScriptureRAG knowledge base..."):
    pipeline = load_rag_pipeline()

# Session State for Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for New Chat
with st.sidebar:
    st.markdown("### 🕉️ ScriptureRAG")
    st.markdown("Vedic wisdom grounded in the 18 Ashtadasha Mahapuranas.")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Welcome Title (only when no conversation yet)
if len(st.session_state.messages) == 0:
    st.markdown('<div class="hero-title">Where should we begin?</div>', unsafe_allow_html=True)

# Render Conversation History
for msg in st.session_state.messages:
    avatar = "🕉️" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat Input Box
user_query = st.chat_input("Ask ScriptureRAG about the Mahapuranas...")

if user_query:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Process and Display Assistant Response
    with st.chat_message("assistant", avatar="🕉️"):
        with st.spinner("Consulting sacred Mahapuranas..."):
            try:
                # Stage 1: Guardrail Check
                guard_res = pipeline["guardrail"].check_query(user_query)

                if not guard_res.allowed:
                    blocked_msg = f"I am dedicated exclusively to providing spiritual and philosophical guidance from the 18 Ashtadasha Mahapuranas. I cannot fulfill requests outside of Vedic scriptures ({guard_res.reason})."
                    st.markdown(blocked_msg)
                    st.session_state.messages.append({"role": "assistant", "content": blocked_msg})
                else:
                    # Stage 2: Query Expansion
                    expanded = pipeline["processor"].process_query(user_query)

                    # Stage 3: Vector Search (across all 10,582 chunks)
                    candidates = pipeline["vector_db"].search(
                        query=expanded.expanded_query,
                        top_k=6
                    )

                    # Stage 4: Cross-Encoder Re-Ranking
                    top_passages = pipeline["reranker"].rerank(
                        query=user_query,
                        retrieved_docs=candidates,
                        top_k=3
                    )

                    # Stage 5: Grounded LLM Generation
                    llm_res = pipeline["llm"].generate_response(
                        query=user_query,
                        retrieved_passages=top_passages
                    )

                    # Stage 6: Output Guardrail Validation
                    val_report = pipeline["validator"].validate_output(
                        query=user_query,
                        response_text=llm_res["response_text"],
                        retrieved_passages=top_passages
                    )

                    # Output Response
                    st.markdown(val_report.sanitized_response)

                    # Save to History
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": val_report.sanitized_response
                    })
            except Exception as err:
                err_msg = f"An error occurred while processing your query: `{str(err)}`"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
