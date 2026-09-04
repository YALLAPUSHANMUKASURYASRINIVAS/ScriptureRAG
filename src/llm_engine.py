import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Reconfigure stdout/stderr for Unicode safety in Windows PowerShell (CP1252 fix)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Anchor project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def get_gemini_api_key() -> str:
    """Loads Gemini API key reliably from .env, Streamlit secrets, or Environment."""
    # 1. Try python-dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv()
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    except ImportError:
        pass

    # 2. Check os.environ
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    # 3. Streamlit Secrets
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # 4. Direct .env file parser (checks root and src dirs)
    possible_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            k = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if k:
                                return k
            except Exception:
                pass

    return ""


class ScriptureLLMEngine:
    """
    Architecture Block 6: Grounded LLM Generation Engine (v2 Production).
    
    Synthesizes authentic responses strictly grounded in retrieved Mahapurana passages.
    Uses 'gemini-flash-latest' for fast, citation-backed generation.
    Enforces verse citations and zero-hallucination guardrails.
    """

    DEFAULT_MODEL = "models/gemini-3.6-flash"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.api_key = get_gemini_api_key()
        self.model = None
        self.model_name = model_name

        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                
                # Active production candidates in priority order
                preferred_models = [
                    "models/gemini-3.6-flash",
                    "gemini-3.6-flash",
                    "models/gemini-flash-latest",
                    "gemini-flash-latest",
                    "models/gemini-pro-latest"
                ]

                # Handshake test to find the working model
                for candidate in preferred_models:
                    try:
                        test_model = genai.GenerativeModel(candidate)
                        test_model.generate_content("ping", generation_config={"max_output_tokens": 2})
                        self.model = test_model
                        self.model_name = candidate
                        print(f"[LLMEngine] Successfully connected to active Gemini model: {self.model_name}")
                        break
                    except Exception:
                        continue

                if not self.model:
                    self.model_name = "models/gemini-3.6-flash"
                    self.model = genai.GenerativeModel(self.model_name)
                    print(f"[LLMEngine] Connected to default model: {self.model_name}")

            except Exception as e:
                print(f"[LLMEngine] Warning: Gemini initialization failed ({e})")
        else:
            print("[LLMEngine] Running in fallback mode (Gemini API key not configured).")

    def _clean_ocr(self, text: str) -> str:
        """Strips corrupted Cyrillic/Russian OCR characters from scanned Sanskrit PDFs."""
        import re
        # Remove non-ASCII/Cyrillic garbled characters
        cleaned = re.sub(r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F]+', '', text)
        # Clean repetitive punctuation and normalize whitespace
        cleaned = re.sub(r'[?\'!]{2,}', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def _build_system_prompt(self) -> str:
        return """You are ScriptureRAG, an authoritative AI scholar and spiritual counselor on the 18 Ashtadasha Mahapuranas.

YOUR CORE MANDATES:
1. STRICT GROUNDING: Answer the user's question relying strictly on Vedic & Puranic philosophy and the provided Mahapurana Context Passages.
2. CITATION REQUIREMENT: Include source citations in brackets, e.g., [Vishnu Purana, Page X] or [Bhagavata Purana, Canto Y].
3. STRUCTURED WISDOM:
   - Direct Counsel: Clear, compassionate philosophical explanation addressing the user's query.
   - Puranic Evidence: Narrative or philosophical teachings from the retrieved scriptures.
   - Ethical / Dharmic Principle: How this applies to righteous living, Karma, Bhakti, or inner peace."""

    def _format_context(self, passages: List[Dict[str, Any]]) -> str:
        formatted = []
        for idx, p in enumerate(passages, 1):
            purana = p.get("purana_name", "Mahapurana")
            story = p.get("story_title", "")
            source = p.get("content_source", "")
            meta_label = f"{purana} | {story}" if story else f"{purana} | {source}"
            formatted.append(f"[Passage {idx}] Source: {meta_label}\n{self._clean_ocr(p.get('text_content', ''))}\n")
        return "\n".join(formatted)

    def generate_response(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        is_short: bool = False
    ) -> Dict[str, Any]:
        """
        Generates grounded LLM response with citations.
        Supports both short concise summaries and detailed full counsel.
        """
        if not retrieved_passages:
            return {
                "response_text": "No relevant Mahapurana scripture passages were found for this query.",
                "citations": [],
                "model_used": "none",
                "status": "empty_context"
            }

        # Extract citation list for metadata
        citations = []
        for p in retrieved_passages:
            purana = p.get("purana_name", "Unknown Purana")
            story = p.get("story_title", "")
            citations.append(f"{purana} ({story})" if story else purana)

        # Build context block
        context_str = self._format_context(retrieved_passages)

        if is_short:
            length_instruction = (
                "STYLE INSTRUCTION: Provide a SHORT, CONCISE answer (2-3 concise paragraphs or bullet points). "
                "Explain the core essence clearly and directly. Include brief scripture citations in brackets."
            )
            max_tokens = 600
        else:
            length_instruction = (
                "STYLE INSTRUCTION: Provide a DETAILED, COMPREHENSIVE spiritual counsel structured in 3 sections: "
                "(1) Direct Counsel, (2) Puranic Evidence with citations, (3) Ethical / Dharmic Principle."
            )
            max_tokens = 2048

        user_prompt = f"""USER QUERY:
{query}

{length_instruction}

RETRIEVED MAHAPURANA CONTEXT PASSAGES:
{context_str}

Please synthesize an authentic, citation-backed response based on the scripture passages above:"""

        # Call Gemini if available
        if self.model:
            for attempt_model in [self.model_name, "models/gemini-3.6-flash", "gemini-3.6-flash", "models/gemini-flash-latest"]:
                try:
                    active_m = genai.GenerativeModel(attempt_model)
                    full_prompt = f"{self._build_system_prompt()}\n\n{user_prompt}"
                    response = active_m.generate_content(
                        full_prompt,
                        generation_config={"temperature": 0.2, "max_output_tokens": max_tokens}
                    )
                    if response and response.text:
                        return {
                            "response_text": response.text.strip(),
                            "citations": citations,
                            "model_used": attempt_model,
                            "status": "success"
                        }
                except Exception as e:
                    print(f"[LLMEngine] Error with {attempt_model}: {e}")

        # Clean structured Fallback if all API calls fail or API key is not configured
        p1 = self._clean_ocr(retrieved_passages[0].get('text_content', ''))
        p2 = self._clean_ocr(retrieved_passages[1].get('text_content', '')) if len(retrieved_passages) > 1 else ""
        
        if is_short:
            short_p1 = ". ".join(p1.split(". ")[:2]).strip()
            if short_p1 and not short_p1.endswith("."):
                short_p1 += "."
            fallback_text = (
                f"### Concise Puranic Summary\n\n"
                f"{short_p1}\n\n"
                f"**Citations:** {', '.join(citations[:2])}"
            )
        else:
            fallback_text = (
                f"### Authentic Puranic Counsel\n\n"
                f"Based on the canonical teachings preserved in **{', '.join(citations[:2])}**:\n\n"
                f"1. **Core Scripture Teaching:**\n{p1.strip()}\n\n"
            )
            if p2:
                fallback_text += f"2. **Further Scriptural Context:**\n{p2.strip()}\n\n"
            fallback_text += f"**Citations:** {', '.join(citations)}"
        
        return {
            "response_text": fallback_text,
            "citations": citations,
            "model_used": "fallback_extractor",
            "status": "fallback"
        }


# -----------------------------------------------------------------------
# End-to-End Pipeline Test (Phase 4 -> 5 -> 6 -> 7)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Phase 7: Grounded LLM Generation Engine Test ---\n")

    # Import full pipeline components
    from src.query_processor import ScriptureQueryProcessor
    from src.vector_db import ScriptureVectorDB
    from src.reranker import ScriptureReranker

    # Initialize Pipeline
    print("[Pipeline] Initializing VectorDB, Reranker, and LLM Engine...")
    vdb = ScriptureVectorDB()
    reranker = ScriptureReranker()
    llm = ScriptureLLMEngine()

    test_queries = [
        "What does the scripture teach about overcoming fear and material anxiety?",
        "Tell me about the churning of the ocean (Samudra Manthan) and Lord Vishnu."
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"USER QUERY: '{q}'")
        print("=" * 80)

        # 1. Query Expansion (Phase 4)
        processed = ScriptureQueryProcessor.process_query(q)
        print(f"[Phase 4] Expanded Query: {processed.expanded_query}\n")

        # 2. Vector Search (Phase 5 - Top 5)
        candidates = vdb.search(query=processed.expanded_query, top_k=5)
        print(f"[Phase 5] Retrieved {len(candidates)} candidate vectors.")

        # 3. Cross-Encoder Re-Ranking (Phase 6 - Top 3)
        top_passages = reranker.rerank(query=q, retrieved_docs=candidates, top_k=3)
        print(f"[Phase 6] Re-ranked to Top-{len(top_passages)} high-precision passages.\n")

        # 4. LLM Generation (Phase 7)
        print("[Phase 7] Generating grounded answer with Gemini...")
        result = llm.generate_response(query=q, retrieved_passages=top_passages)

        print("\n" + "-" * 40 + " SYNTHESIZED RESPONSE " + "-" * 40)
        print(result["response_text"])
        print("\nCitations :", result["citations"])
        print("Status    :", result["status"], f"({result['model_used']})")
        print("-" * 102)