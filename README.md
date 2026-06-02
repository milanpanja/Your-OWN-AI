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

## 📁 File Structure

```
main.py       ← All algorithms + HTTP server (~450 lines, no frameworks)
index.html    ← Frontend (PCA scatter plot, search UI, RAG chat)
requirements.txt ← Just "requests" for Ollama HTTP calls
```
