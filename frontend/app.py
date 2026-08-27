import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components

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
    
    /* Hide Default Header and Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
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

    return {
        "guardrail": MahapuranaGuardrail,
        "processor": ScriptureQueryProcessor,
        "vector_db": ScriptureVectorDB(),
        "reranker": ScriptureReranker(),
        "llm": ScriptureLLMEngine(),
        "validator": ScriptureOutputGuardrail()
    }

with st.spinner("Loading..."):
    pipeline = load_rag_pipeline()

# Session State for Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for New Chat Only
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Welcome Title (only when no conversation yet)
if len(st.session_state.messages) == 0:
    st.markdown('<div class="hero-title">Where should we begin?</div>', unsafe_allow_html=True)

# Render Conversation
for msg in st.session_state.messages:
    avatar = "🕉️" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --- HTML5 Speech-to-Text Microphone Integration ---
components.html("""
<script>
function startVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Speech Recognition is not supported in this browser. Please use Chrome or Edge.");
        return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    
    const micBtn = document.getElementById("mic-btn");
    micBtn.innerText = "🔴 Listening...";
    
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        micBtn.innerText = "🎙️ Speak Question";
        
        // Find Streamlit's chat input textarea
        const parentDoc = window.parent.document;
        const chatInput = parentDoc.querySelector("textarea[data-testid='stChatInputTextArea']");
        if (chatInput) {
            chatInput.value = transcript;
            chatInput.dispatchEvent(new Event('input', { bubbles: true }));
            const submitBtn = parentDoc.querySelector("button[data-testid='stChatInputSubmitButton']");
            if (submitBtn) {
                setTimeout(() => submitBtn.click(), 300);
            }
        }
    };
    
    recognition.onerror = function() {
        micBtn.innerText = "🎙️ Speak Question";
    };
    
    recognition.onend = function() {
        micBtn.innerText = "🎙️ Speak Question";
    };
    
    recognition.start();
}
</script>
<div style="display: flex; justify-content: center; margin-bottom: 8px;">
    <button id="mic-btn" onclick="startVoiceInput()" style="
        background-color: #2f2f2f;
        color: #ECECEC;
        border: 1px solid #424242;
        border-radius: 20px;
        padding: 6px 18px;
        font-size: 0.9rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
    ">
        🎙️ Speak Question
    </button>
</div>
""", height=48)

# Chat Input Box
user_query = st.chat_input("Ask ScriptureRAG...")

if user_query:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Assistant Response
    with st.chat_message("assistant", avatar="🕉️"):
        with st.spinner("Thinking..."):
            # Stage 1: Guardrail Check
            guard_res = pipeline["guardrail"].check_query(user_query)

            if not guard_res.allowed:
                blocked_msg = f"I am dedicated exclusively to providing spiritual and philosophical guidance from the 18 Ashtadasha Mahapuranas. I cannot fulfill requests outside of Vedic scriptures ({guard_res.reason})."
                st.markdown(blocked_msg)
                st.session_state.messages.append({"role": "assistant", "content": blocked_msg})
            else:
                # Stage 2: Query Expansion
                expanded = pipeline["processor"].process_query(user_query)

                # Stage 3: Vector Search
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



