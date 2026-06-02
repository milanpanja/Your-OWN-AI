
# VectorDB — Python Edition

### Setup & Run Guide (for Interview Demo)

---

## ✅ Step 1 — Install Python (if not installed)

Download from: https://www.python.org/downloads/

* Choose **Python 3.10 or later**
* On Windows: **check "Add Python to PATH"** during install

Verify in terminal:

```
python --version
```

---

## ✅ Step 2 — Install the ONE dependency

```bash
pip install requests
```

That's it. Everything else (HNSW, KD-Tree, BruteForce, HTTP server) is built from scratch in `main.py` using only Python standard library.

---

## ✅ Step 3 — Install Ollama (for AI features)

1. Go to https://ollama.com → Download for your OS
2. Open terminal and pull models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

> **Note:** Ollama is optional. The demo search (Tab 1) works without it.
> You only need Ollama for the Documents tab and Ask AI tab.

---

## ✅ Step 4 — Run the Server

Put all 3 files in one folder:

```
your_folder/
├── main.py
├── index.html
└── requirements.txt
```

Then run:

```bash
python main.py
```

You should see:

```
=== VectorDB Engine (Python Edition) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW + KD-Tree + BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text   gen model: llama3.2
```

Open your browser: **http://localhost:8080**

---

## 🎯 What to show in the interview

### Tab 1 — Search

* Type: `binary tree`, `sushi`, `basketball`, `calculus`
* Switch algorithms: HNSW, KD-Tree, BruteForce
* Click **COMPARE ALL** to benchmark all 3 and show speed differences
* Point to the 2D scatter plot — "see how semantic categories cluster"

### Tab 2 — Documents

* Paste any text (lecture notes, Wikipedia article)
* Click **EMBED & INSERT** — Ollama converts text to 768D vector

### Tab 3 — Ask AI (RAG)

* Type a question about your inserted documents
* Explain: embed question → HNSW search → top chunks → LLM generates answer

---

## 💬 Interview Q&A Cheat Sheet

**Q: What is a Vector Database?**

> A database that stores vectors (arrays of numbers) and lets you find the most semantically similar items. Instead of exact keyword matching, it finds things that are conceptually close — even if the words are different.

**Q: What is HNSW?**

> Hierarchical Navigable Small World — a multilayer graph. Layer 0 has all nodes densely connected. Higher layers have fewer nodes with long-range shortcuts (like a highway). Search starts at the top and zooms in, giving O(log N) speed instead of O(N) brute force.

**Q: Why is KD-Tree bad at high dimensions?**

> KD-Tree prunes subtrees using axis-aligned distance bounds. In high dimensions, almost all space is near the boundary of the hypersphere — no subtrees get pruned, and it degenerates to O(N). HNSW doesn't have this "curse of dimensionality" problem.

**Q: What is RAG?**

> Retrieval-Augmented Generation. Instead of relying only on the LLM's training data, we: (1) embed the user's question, (2) find the most relevant document chunks via HNSW, (3) inject those chunks as context, (4) let the LLM generate a grounded answer.

**Q: What is cosine similarity?**

> Measures the angle between two vectors, ignoring magnitude. cosine_distance = 1 - (dot product / product of magnitudes). A distance of 0 means identical direction (most similar), 1 means perpendicular (unrelated), 2 means opposite.

**Q: How does your Python implementation differ from C++?**

> Logic is identical — same HNSW insert/search algorithm, same KD-Tree recursion, same distance formulas. Python uses built-in `heapq` for priority queues and `threading.Lock` for thread safety. The HTTP server uses Python's standard `http.server`. No external vector libraries — everything is implemented from scratch.

**Q: What are the time complexities?**

> * BruteForce: O(N·d) search — checks every vector
> * KD-Tree:    O(log N) average, O(N) worst case (high dims)
> * HNSW:       O(log N) average for approximate nearest neighbor

**Q: How does text chunking work?**

> Long documents are split into overlapping chunks (250 words each, 30-word overlap). The overlap prevents losing information at boundaries. Each chunk gets its own embedding and HNSW entry.

---

## 📁 File Structure

```
main.py       ← All algorithms + HTTP server (~450 lines, no frameworks)
index.html    ← Frontend (PCA scatter plot, search UI, RAG chat)
requirements.txt ← Just "requests" for Ollama HTTP calls
```
