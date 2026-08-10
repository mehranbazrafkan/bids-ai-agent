import os
from typing import Dict, List

class Retriever:
    def __init__(self, data_dir: str = "./") -> None:
        self.data_dir = data_dir

    def retrieve(self, query: str, top_k: int = 2) -> str:
        """Reads local .txt and .md files and returns relevant content based on simple keyword matching."""
        if not os.path.exists(self.data_dir):
            return "No local documentation directory found."

        docs: Dict[str, str] = {}
        # 1. Load files (.txt, .md)
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith((".txt", ".md")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            docs[file] = f.read()
                    except Exception as e:
                        continue

        if not docs:
            return "No local documentation files available."

        # 2. Simple keyword matching scoring
        query_words = set(query.lower().split())
        scored_docs: List[tuple[int, str]] = []

        for filename, content in docs.items():
            content_lower = content.lower()
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored_docs.append((score, f"--- Source: {filename} ---\n{content}"))

        # Sort by match score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        if not scored_docs:
            return "No relevant documentation found for the query."

        # Return top_k matches joined together
        selected = [doc_text for _, doc_text in scored_docs[:top_k]]
        return "\n\n".join(selected)