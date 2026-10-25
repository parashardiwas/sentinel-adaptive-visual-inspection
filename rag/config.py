"""
Configuration settings for Step 3: NLP / RAG Module
"""

from pathlib import Path
import os

class RAGConfig:
    # Base paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    RAG_DIR = REPO_ROOT / "rag"
    KNOWLEDGE_BASE_DIR = RAG_DIR / "knowledge_base"
    VECTOR_CACHE_DIR = RAG_DIR / "vector_cache"
    
    # Colab Drive standard paths vs Local Fallback
    COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/Integrated")
    LOCAL_DRIVE_ROOT = REPO_ROOT / "local_test"
    
    @classmethod
    def get_outputs_dir(cls) -> Path:
        """Locates the outputs directory, checking Colab Drive first, then local repo."""
        if cls.COLAB_DRIVE_ROOT.exists():
            return cls.COLAB_DRIVE_ROOT / "outputs"
        elif (cls.REPO_ROOT / "outputs").exists():
            return cls.REPO_ROOT / "outputs"
        elif cls.LOCAL_DRIVE_ROOT.exists():
            return cls.LOCAL_DRIVE_ROOT / "outputs"
        return cls.REPO_ROOT / "outputs"

    @classmethod
    def get_vision_outputs_dir(cls) -> Path:
        return cls.get_outputs_dir() / "vision"

    @classmethod
    def get_scheduler_outputs_dir(cls) -> Path:
        return cls.get_outputs_dir() / "scheduler_outputs"

    @classmethod
    def get_rag_outputs_dir(cls) -> Path:
        rag_out = cls.get_outputs_dir() / "rag_outputs"
        rag_out.mkdir(parents=True, exist_ok=True)
        return rag_out

    # Model parameters
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SIMILARITY_TOP_K = 3
    SIMILARITY_THRESHOLD = 0.35

    # Categories
    CATEGORIES = ["grid", "zipper", "hazelnut", "leather", "toothbrush"]
