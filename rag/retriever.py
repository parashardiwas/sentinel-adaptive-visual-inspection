"""
Retriever Module for Hybrid Semantic & Metadata Search
"""

from typing import List, Dict, Any, Optional
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.config import RAGConfig

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        defect_code: Optional[str] = None,
        top_k: int = RAGConfig.SIMILARITY_TOP_K
    ) -> List[Dict[str, Any]]:
        """Retrieve semantically similar chunks, preferring specific SOP metadata on close matches."""
        query_vec = self.embedding_model.encode_one(query)
        # The SOP index is small, so rank the complete metadata-filtered result set.
        # This prevents a generic policy chunk from crowding out the exact defect clause.
        candidate_k = max(len(self.vector_store.chunks), top_k)
        candidates = self.vector_store.query(
            query_vec=query_vec,
            top_k=candidate_k,
            category=category,
            defect_code=defect_code
        )
        if not candidates:
            return []

        # General SOP chunks often share vocabulary with every query. Use the
        # existing metadata filters as a small ranking preference so a relevant
        # category/defect SOP is not routinely hidden behind a generic policy page.
        for chunk in candidates:
            base_score = float(chunk.get("similarity_score", 0.0))
            category_match = bool(category and str(chunk.get("category", "")).lower() == category.lower())
            defect_match = bool(
                defect_code
                and str(chunk.get("defect_code", "")).lower() == defect_code.lower()
                and str(chunk.get("defect_code", "")).lower() != "general"
            )
            chunk["_retrieval_rank"] = base_score + (0.06 if category_match else 0.0) + (0.20 if defect_match else 0.0)

        candidates.sort(key=lambda chunk: chunk["_retrieval_rank"], reverse=True)
        results = candidates[:top_k]
        for chunk in results:
            chunk.pop("_retrieval_rank", None)
        return results

    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into an authoritative prompt context block with citations."""
        if not chunks:
            return "No specific SOP documentation matched this query."

        context_blocks = []
        for i, c in enumerate(chunks, 1):
            block = (
                f"[Source Ref {i}: {c.get('doc_id', 'SOP')} - {c.get('section_title', 'General')}]\n"
                f"Category: {c.get('category', 'all').upper()} | Defect Code: {c.get('defect_code', 'general')}\n"
                f"{c.get('content', '').strip()}\n"
            )
            context_blocks.append(block)

        return "\n----------------------------------------\n".join(context_blocks)
