"""
Document Ingestion & Chunking for Manufacturing SOP Knowledge Base
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from rag.config import RAGConfig

class KnowledgeBaseIngestor:
    def __init__(self, kb_dir: Path = None):
        self.kb_dir = kb_dir or RAGConfig.KNOWLEDGE_BASE_DIR

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Reads all markdown files in the knowledge base directory and parses them
        into structured semantic chunks with rich metadata.
        """
        if not self.kb_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found at {self.kb_dir}")

        md_files = list(self.kb_dir.glob("*.md"))
        chunks = []

        for file_path in md_files:
            file_chunks = self._parse_markdown_file(file_path)
            chunks.extend(file_chunks)

        return chunks

    def _parse_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc_id = "UNKNOWN"
        category = "general"

        # Detect Document ID and Category
        doc_id_match = re.search(r"-\s*\*\*Document ID\*\*:\s*([^\n\r]+)", content)
        if doc_id_match:
            doc_id = doc_id_match.group(1).strip()
        else:
            doc_id_header = re.search(r"^#\s*([A-Z0-9\-]+):", content, re.MULTILINE)
            if doc_id_header:
                doc_id = doc_id_header.group(1).strip()

        cat_match = re.search(r"-\s*\*\*Category\*\*:\s*`?([a-zA-Z0-9_\-]+)`?", content)
        if cat_match:
            category = cat_match.group(1).strip().lower()
        else:
            # Fallback infer from file name
            stem = file_path.stem.lower()
            for cat in RAGConfig.CATEGORIES:
                if cat in stem:
                    category = cat
                    break

        # Split by section headers (## or ###)
        sections = re.split(r"(?=(?:^|\n)#{2,3}\s+)", content)
        chunks = []

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            # Extract header and body
            header_match = re.match(r"^#{2,3}\s+([^\n]+)", sec)
            if header_match:
                header = header_match.group(1).strip()
            else:
                header = f"Overview ({doc_id})"

            # Extract defect code if present in header or body
            defect_code = "general"
            defect_match = re.search(r"Defect Code:\s*`?([a-zA-Z0-9_\-]+)`?", header, re.IGNORECASE)
            if defect_match:
                defect_code = defect_match.group(1).strip().lower()
            else:
                for d in [
                    "bent", "broken", "glue", "metal_contamination", "thread",
                    "broken_teeth", "split_teeth", "rough", "fabric_border",
                    "fabric_interior", "squeezed_teeth", "crack", "cut", "hole",
                    "print", "color", "fold", "poke", "defective"
                ]:
                    if re.search(rf"\b{d}\b", header, re.IGNORECASE):
                        defect_code = d
                        break

            chunk = {
                "id": f"{doc_id}_{len(chunks) + 1}",
                "doc_id": doc_id,
                "category": category,
                "defect_code": defect_code,
                "section_title": header,
                "content": sec,
                "source_file": file_path.name
            }
            chunks.append(chunk)

        return chunks
