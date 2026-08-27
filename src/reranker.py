import sys
import os
from pathlib import Path

# Anchor project root to sys.path so 'src.vector_db' imports seamlessly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class ScriptureReranker:
    """
    Architecture Block 5: Cross-Encoder Re-Ranking Layer (v2 Production).
    
    Takes candidate passages retrieved by VectorDB (Phase 5) and applies
    deep full-token cross-attention scoring using ms-marco-MiniLM-L-6-v2.
    Re-ranks and filters to the Top-K most faithful scripture passages.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Reranker] Initializing Cross-Encoder on device: {self.device.upper()}")
        print(f"[Reranker] Loading model: {model_name}...")
        self.model = CrossEncoder(model_name, device=self.device)
        print("[Reranker] Cross-Encoder loaded successfully.")

    def rerank(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of retrieved documents using cross-attention scoring.
        
        Args:
            query: User's original or expanded query.
            retrieved_docs: Candidate passages returned from VectorDB search.
            top_k: Number of highest-scoring passages to return (default: 3).
            
        Returns:
            List of re-ordered documents with guaranteed 'rerank_score'.
        """
        if not retrieved_docs:
            return []

        # Safe fallback maintaining contract schema
        if not query or not query.strip():
            return [{**doc, "rerank_score": 0.0} for doc in retrieved_docs[:top_k]]

        # 1. Prepare (Query, Passage Text) pairs with safe None handling
        clean_query = query.strip()
        pairs = [
            [clean_query, (doc.get("text_content") or "").strip()]
            for doc in retrieved_docs
        ]

        # 2. Compute cross-attention relevance scores with error safety
        try:
            raw_scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"[Reranker] Warning: Prediction failed ({e}), preserving vector retrieval order.")
            return [{**doc, "rerank_score": float(doc.get("similarity_score", 0.0))} for doc in retrieved_docs[:top_k]]

        # 3. Attach scores and sort documents descending
        scored_docs = []
        for idx, doc in enumerate(retrieved_docs):
            doc_copy = dict(doc)
            # Cross-encoder raw logit (higher = more relevant)
            doc_copy["rerank_score"] = round(float(raw_scores[idx]), 4)
            scored_docs.append(doc_copy)

        # Sort by cross-encoder score descending (stable sort preserves vector rank on tie)
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_docs[:top_k]


# -----------------------------------------------------------------------
# Verification / Test Suite (End-to-End Phase 5 + Phase 6 Pipeline)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Phase 6: Cross-Encoder Re-Ranking Layer Test (v2 Production) ---\n")
    
    # Import VectorDB from Phase 5
    from src.vector_db import ScriptureVectorDB

    # Initialize VectorDB & Reranker
    vdb = ScriptureVectorDB()
    reranker = ScriptureReranker()

    test_queries = [
        "What does Vishnu Purana teach about Karma and Grihastha duties?",
        "Tell me about Samudra Manthan and the churning of the ocean",
        "How to overcome fear and despair according to Prahlada?"
    ]

    for q in test_queries:
        print("\n" + "=" * 75)
        print(f"QUERY: '{q}'")
        print("=" * 75)

        # Step A: Bi-Encoder Vector Search (Top-5 candidates)
        candidates = vdb.search(query=q, top_k=5)
        print(f"\n[Phase 5: Bi-Encoder] Retrieved {len(candidates)} candidate passages.")

        # Step B: Cross-Encoder Re-Ranking (Top-3 most relevant)
        top_passages = reranker.rerank(query=q, retrieved_docs=candidates, top_k=3)
        print(f"[Phase 6: Cross-Encoder] Re-ranked into Top-{len(top_passages)} passages:\n")

        for rank, item in enumerate(top_passages, 1):
            print(f"  Rank #{rank} | Rerank Score: {item['rerank_score']:>7.4f} | Vector Sim: {item['similarity_score']:.4f}")
            print(f"  Scripture: {item['purana_name']} ({item['content_source']})")
            print(f"  Chapter/Section: {item['story_title']}")
            preview = (item['text_content'] or '')[:200].replace('\n', ' ')
            print(f"  Passage: \"{preview}...\"\n")