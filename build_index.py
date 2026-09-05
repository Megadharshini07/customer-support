"""
Run this once, locally, with GEMINI_API_KEY set:

    export GEMINI_API_KEY=your_key_here
    python build_index.py

It embeds every document in data/ and writes index/kb_index.json +
index/kb_vectors.npy. Commit both files to your repo — that way the
Devfolio judges' `python app.py` doesn't need to re-embed the whole
knowledge base inside the 90-second startup window.
"""
from rag import build_index

if __name__ == "__main__":
    build_index()
