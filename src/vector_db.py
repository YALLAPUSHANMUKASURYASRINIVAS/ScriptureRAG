import os
import json
import torch
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class ScriptureVectorDB:
    """
    Architecture Block 4: ChromaDB Persistent Vector Database Manager (v2 Production).
    
    - Model: BAAI/bge-base-en-v1.5 (768-dim dense embeddings)
    - Distance Metric: Cosine Similarity
    - Persistence Directory: data/chroma_db/
    - Collection: mahapuranas_collection
    - Global Cross-Batch De-duplication: Prevents silent ChromaDB overwrites
    """

    COLLECTION_NAME = "mahapuranas_collection"
    EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        chunks_json_path: str = "data/preprocessed_master_chunks.json"
    ):
        self.persist_dir = os.path.abspath(persist_dir)
        self.chunks_json_path = os.path.abspath(chunks_json_path)
        
        # 1. Detect Device (CUDA GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VectorDB] Initializing on device: {self.device.upper()}")

        # 2. Load Embedding Model
        print(f"[VectorDB] Loading embedding model: {self.EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME, device=self.device)

        # 3. Initialize Persistent ChromaDB Client
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # 4. Get or Create Collection with Cosine Space
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorDB] ChromaDB connected. Existing records: {self.collection.count()}")

    def index_dataset(self, batch_size: int = 64, force_reindex: bool = False) -> int:
        """
        Indexes all preprocessed Mahapurana chunks into ChromaDB in batches.
        Guarantees 100% unique IDs across all 10,582 chunks.
        """
        current_count = self.collection.count()
        if current_count > 0 and not force_reindex:
            print(f"[VectorDB] Database already populated with {current_count} vectors. Skipping indexing.")
            return current_count

        if force_reindex and current_count > 0:
            print("[VectorDB] Force reindex requested. Clearing existing collection...")
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

        # Load chunks from JSON
        if not os.path.exists(self.chunks_json_path):
            raise FileNotFoundError(
                f"[VectorDB] Chunks file not found at: {self.chunks_json_path}. "
                "Please run 'python src/preprocessing.py' first."
            )

        print(f"[VectorDB] Loading chunks from {self.chunks_json_path}...")
        with open(self.chunks_json_path, "r", encoding="utf-8") as f:
            chunks: List[Dict[str, Any]] = json.load(f)

        total_chunks = len(chunks)
        print(f"[VectorDB] Starting batch embedding of {total_chunks} chunks (Batch Size: {batch_size})...")

        # Global seen-ID tracker across ALL batches
        seen_global_ids = set()

        # Process and upsert in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]

            batch_ids: List[str] = []
            batch_texts: List[str] = []
            batch_metadatas: List[Dict[str, Any]] = []

            for idx, chunk in enumerate(batch):
                global_idx = i + idx
                raw_id = str(chunk.get("chunk_id", f"chunk_{global_idx}")).strip()

                if raw_id in seen_global_ids:
                    unique_id = f"{raw_id}_G{global_idx}"
                else:
                    unique_id = raw_id

                seen_global_ids.add(unique_id)

                text = str(chunk.get("text_content", "")).strip()
                if not text:
                    continue

                batch_ids.append(unique_id)
                batch_texts.append(text)
                batch_metadatas.append({
                    "purana_name": str(chunk.get("purana_name", "Unknown")),
                    "skandha_book": int(chunk.get("skandha_book", 1)),
                    "chapter": int(chunk.get("chapter", 1)),
                    "story_title": str(chunk.get("story_title", "")),
                    "content_source": str(chunk.get("content_source", "Unknown"))
                })

            if not batch_texts:
                continue

            # Generate dense embeddings (BGE model)
            embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()

            # Upsert into ChromaDB
            self.collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )

            processed = min(i + batch_size, total_chunks)
            print(f"[VectorDB] Indexed [{processed}/{total_chunks}] chunks ({processed / total_chunks * 100:.1f}%)")

        final_count = self.collection.count()
        print(f"[VectorDB] Indexing complete! Total vectors in ChromaDB: {final_count}")
        return final_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_purana: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves Top-K most semantically similar scripture chunks for a query.
        """
        if not query or not query.strip():
            return []

        # Encode query to vector
        query_vector = self.embedding_model.encode(
            [query.strip()],
            normalize_embeddings=True
        ).tolist()

        # Build where filter if specified
        where_clause = None
        if filter_purana:
            where_clause = {"purana_name": filter_purana}

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            where=where_clause
        )

        formatted_results: List[Dict[str, Any]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return formatted_results

        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for i in range(len(docs)):
            cosine_dist = distances[i] if i < len(distances) else 0.0
            similarity = max(0.0, 1.0 - cosine_dist)

            formatted_results.append({
                "chunk_id": ids[i],
                "similarity_score": round(similarity, 4),
                "text_content": docs[i],
                "purana_name": metadatas[i].get("purana_name", "Unknown"),
                "skandha_book": metadatas[i].get("skandha_book", 1),
                "chapter": metadatas[i].get("chapter", 1),
                "story_title": metadatas[i].get("story_title", ""),
                "content_source": metadatas[i].get("content_source", "Unknown")
            })

        return formatted_results


# -----------------------------------------------------------------------
# Verification / Test Suite
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Phase 5: ChromaDB Vector Indexing & Search Test ---\n")
    
    # 1. Initialize Vector Database
    vdb = ScriptureVectorDB()

    # 2. Index Dataset (runs once; skips automatically if already indexed)
    vdb.index_dataset(batch_size=64)

    # 3. Test Retrieval with Sample Queries
    test_queries = [
        "What does Vishnu Purana teach about Karma and Grihastha duties?",
        "Tell me about Samudra Manthan and the churning of the ocean",
        "How to overcome fear and despair according to Prahlada?"
    ]

    for q in test_queries:
        print(f"\nSearching for: '{q}'")
        results = vdb.search(query=q, top_k=2)
        
        for rank, res in enumerate(results, 1):
            print(f"  [Result {rank}] Similarity: {res['similarity_score']} | Purana: {res['purana_name']}")
            print(f"  Source: {res['content_source']} | Story/Section: {res['story_title']}")
            preview = res['text_content'][:200].replace('\n', ' ')
            print(f"  Passage: \"{preview}...\"\n")
        print("-" * 75)