"""
rag.py — a small, dependency-light RAG pipeline.

Pipeline: chunk local policy docs -> embed with Gemini (gemini-embedding-001)
-> store vectors in a flat numpy array on disk -> cosine similarity at query
time -> return the top-k chunks with their source doc + policy id.

No hosted vector DB, no other network calls. Just files + numpy.
"""

import os
import json
import glob
import time
import numpy as np
import google.generativeai as genai

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "kb_index.json")
VECS_PATH = os.path.join(INDEX_DIR, "kb_vectors.npy")

EMBED_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150


def _configure():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it before running the app "
            "or building the index."
        )
    genai.configure(api_key=api_key)


def _chunk_text(text, size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """Simple sliding-window chunker on whitespace-normalized text."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _load_documents():
    """Read every .md file in data/ and split into (doc_name, chunk_text) pairs."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.md"))):
        name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        for i, chunk in enumerate(_chunk_text(raw)):
            docs.append({
                "doc": name,
                "chunk_id": f"{name}::chunk{i}",
                "text": chunk,
            })
    return docs


def _embed_batch(texts, task_type):
    """Embed a list of strings with Gemini, one call per text (keeps it simple
    and robust; batch endpoints vary by SDK version)."""
    vectors = []
    for t in texts:
        for attempt in range(3):
            try:
                resp = genai.embed_content(
                    model=EMBED_MODEL,
                    content=t,
                    task_type=task_type,
                )
                vectors.append(resp["embedding"])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
    return np.array(vectors, dtype=np.float32)


def build_index():
    """Embed every KB chunk and write the index to disk. Run this once
    offline (python build_index.py) and COMMIT the resulting index/ files,
    so app startup doesn't have to re-embed the whole KB inside the 90s
    startup budget."""
    _configure()
    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = _load_documents()
    if not docs:
        raise RuntimeError(f"No documents found in {DATA_DIR}")

    texts = [d["text"] for d in docs]
    vectors = _embed_batch(texts, task_type="retrieval_document")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    np.save(VECS_PATH, vectors)

    print(f"Indexed {len(docs)} chunks from {len(set(d['doc'] for d in docs))} documents.")
    return docs, vectors


def _load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(VECS_PATH)):
        # No committed index found — build it now. This is the fallback path;
        # for a fast startup, commit the prebuilt index instead (see README).
        return build_index()
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    vectors = np.load(VECS_PATH)
    return docs, vectors


class Retriever:
    """Loads the index once and answers cosine-similarity queries."""

    def __init__(self):
        _configure()
        self.docs, self.vectors = _load_index()
        # Pre-normalize for fast cosine similarity via dot product.
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self._unit_vectors = self.vectors / norms

    def retrieve(self, query, top_k=4):
        q_vec = _embed_batch([query], task_type="retrieval_query")[0]
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        scores = self._unit_vectors @ q_norm
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for i in top_idx:
            results.append({
                "doc": self.docs[i]["doc"],
                "chunk_id": self.docs[i]["chunk_id"],
                "text": self.docs[i]["text"],
                "score": float(scores[i]),
            })
        return results


if __name__ == "__main__":
    build_index()
