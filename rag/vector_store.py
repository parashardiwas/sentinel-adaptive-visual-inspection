"""
Persistent Vector Store for RAG
Supports ChromaDB when available, and includes a lightweight native vector store fallback.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from rag.config import RAGConfig

class VectorStore:
    def __init__(self, persist_dir: Path = None):
        self.persist_dir = persist_dir or RAGConfig.VECTOR_CACHE_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cache_fingerprint: Optional[str] = None

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adds document chunks and corresponding embeddings to the store."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        self.chunks.extend(chunks)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

    def persist(self, fingerprint: Optional[str] = None):
        """Saves documents and embeddings to disk."""
        meta_file = self.persist_dir / "chunks_metadata.json"
        embed_file = self.persist_dir / "embeddings.npy"
        manifest_file = self.persist_dir / "index_manifest.json"

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

        if self.embeddings is not None:
            np.save(embed_file, self.embeddings)
        if fingerprint:
            manifest_file.write_text(json.dumps({
                "schema_version": 1,
                "fingerprint": fingerprint,
                "chunk_count": len(self.chunks),
                "embedding_shape": list(self.embeddings.shape) if self.embeddings is not None else None,
            }, indent=2), encoding="utf-8")
            self.cache_fingerprint = fingerprint

    def load(self, expected_fingerprint: Optional[str] = None) -> bool:
        """Loads cached documents and embeddings from disk if they exist."""
        meta_file = self.persist_dir / "chunks_metadata.json"
        embed_file = self.persist_dir / "embeddings.npy"
        manifest_file = self.persist_dir / "index_manifest.json"

        if meta_file.exists() and embed_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                if expected_fingerprint and manifest.get("fingerprint") != expected_fingerprint:
                    return False
                if manifest.get("schema_version") != 1:
                    return False
                with open(meta_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                embeddings = np.load(embed_file, allow_pickle=False)
                expected_shape = manifest.get("embedding_shape")
                if len(chunks) != len(embeddings) or (expected_shape and list(embeddings.shape) != expected_shape):
                    return False
                self.chunks = chunks
                self.embeddings = embeddings.astype(np.float32, copy=False)
                self.cache_fingerprint = manifest.get("fingerprint")
                return True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return False
        return False

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 3,
        category: Optional[str] = None,
        defect_code: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarities, applies metadata filtering,
        and returns the top_k matching document chunks.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Cosine similarity (query_vec and embeddings are L2 normalized)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        scores = np.dot(self.embeddings, query_vec)

        # Apply metadata filters
        filtered_indices = []
        for idx, chunk in enumerate(self.chunks):
            # Category filter (allow 'general' documents to be matched for any category)
            if category and chunk.get("category") not in [category, "general"]:
                continue
            # Defect code filter
            if defect_code and chunk.get("defect_code") not in [defect_code, "general"]:
                continue
            filtered_indices.append(idx)

        if not filtered_indices:
            # Relax defect filter if too strict
            for idx, chunk in enumerate(self.chunks):
                if category and chunk.get("category") not in [category, "general"]:
                    continue
                filtered_indices.append(idx)

        if not filtered_indices:
            filtered_indices = list(range(len(self.chunks)))

        filtered_scores = scores[filtered_indices]
        ranked_order = np.argsort(filtered_scores)[::-1][:top_k]

        results = []
        for rank in ranked_order:
            orig_idx = filtered_indices[rank]
            item = dict(self.chunks[orig_idx])
            item["similarity_score"] = float(scores[orig_idx])
            results.append(item)

        return results
