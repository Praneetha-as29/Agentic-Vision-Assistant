
from pathlib import Path
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    documents = []

    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        numbered_items = re.findall(
            r"(?m)^\s*\d+\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)",
            text,
            flags=re.DOTALL
        )

        if numbered_items:
            documents.extend(item.strip() for item in numbered_items)
        else:
            documents.extend(
                paragraph.strip()
                for paragraph in text.split("\n\n")
                if paragraph.strip()
            )

    if not documents:
        raise ValueError("No documents found in the docs folder.")

    return documents

class SafetyKnowledgeBase:
    def __init__(self):
        self.documents = load_documents()
        self.model = SentenceTransformer(MODEL_NAME)
        embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        if not query or not query.strip():
            return []

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_vector = np.asarray(query_vector, dtype="float32")
        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.documents))
        )

        return [
            {
                "text": self.documents[int(index)],
                "similarity": round(float(score), 4)
            }
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

if __name__ == "__main__":
    knowledge_base = SafetyKnowledgeBase()
    query = input("Ask a traffic safety question: ").strip()

    if not query:
        print("Please enter a question.")
    else:
        for result in knowledge_base.search(query):
            print(result)
