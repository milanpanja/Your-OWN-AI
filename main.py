"""
VectorDB — Python Edition
A Vector Database built from scratch in Python.
Implements HNSW, KD-Tree, and Brute Force search + RAG pipeline via Ollama.

Run:  python main.py
Open: http://localhost:8080
"""

import math
import time
import random
import threading
import json
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple, Dict, Optional, Callable
import heapq

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)

def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric: str) -> Callable:
    if metric == "cosine":    return cosine
    if metric == "manhattan": return manhattan
    return euclidean

# =====================================================================
#  DATA TYPE
# =====================================================================

class VectorItem:
    def __init__(self, id: int, metadata: str, category: str, emb: List[float]):
        self.id       = id
        self.metadata = metadata
        self.category = category
        self.emb      = emb

# =====================================================================
#  BRUTE FORCE  — O(N·d) exact search
# =====================================================================

class BruteForce:
    """
    Simplest possible search: compare query against every stored vector.
    Time complexity: O(N * d) where N = number of vectors, d = dimensions.
    Always returns exact results. Used as ground-truth baseline.
    """
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, item: VectorItem):
        self.items.append(item)

    def knn(self, query: List[float], k: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        # Compute distance from query to every stored vector
        scored = [(dist_fn(query, item.emb), item.id) for item in self.items]
        scored.sort()          # sort ascending by distance
        return scored[:k]      # return k nearest

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE  — O(log N) average exact search
# =====================================================================

class KDNode:
    """One node in the KD-Tree binary space partition."""
    def __init__(self, item: VectorItem):
        self.item  = item
        self.left  = None   # points with emb[axis] < this node's emb[axis]
        self.right = None   # points with emb[axis] >= this node's emb[axis]

class KDTree:
    """
    Binary space partitioning tree. Each level splits on one dimension
    (cycling through all dimensions). Prunes subtrees when the closest
    possible point in that subtree can't beat the current best neighbor.

    Works great for low dimensions (≤20D). Degrades at high dims
    (curse of dimensionality — almost nothing gets pruned).
    """
    def __init__(self, dims: int):
        self.dims = dims
        self.root = None

    def _insert(self, node: Optional[KDNode], item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left  = self._insert(node.left,  item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item: VectorItem):
        self.root = self._insert(self.root, item, 0)

    def _knn(self, node: Optional[KDNode], query: List[float], k: int,
             depth: int, dist_fn: Callable, heap: list):
        """
        Recursive KNN search. heap is a max-heap of (-distance, id)
        so heap[0] is always the WORST of the current k best.
        We prune a subtree if its closest possible point is farther
        than our current worst.
        """
        if node is None:
            return
        d = dist_fn(query, node.item.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.item.id))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.item.id))

        axis = depth % self.dims
        diff = query[axis] - node.item.emb[axis]

        # Visit the closer child first (more likely to improve answer)
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer,  query, k, depth + 1, dist_fn, heap)

        # Only visit the farther child if it MIGHT contain a better point
        # (i.e. the hyperplane distance is less than our current worst)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, query, k, depth + 1, dist_fn, heap)

    def knn(self, query: List[float], k: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        heap = []
        self._knn(self.root, query, k, 0, dist_fn, heap)
        results = [(-neg_d, id_) for neg_d, id_ in heap]
        results.sort()
        return results

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for item in items:
            self.insert(item)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World Graph
# =====================================================================

class HNSW:
    """
    HNSW: the algorithm powering Pinecone, Weaviate, Chroma, Milvus.

    STRUCTURE:
      - A multilayer graph. Layer 0 has ALL nodes (dense connections).
      - Higher layers have exponentially fewer nodes (long-range shortcuts).
      - Each node is randomly assigned a max layer level (log-distributed).

    INSERT:
      1. Randomly assign this node a max layer L.
      2. Start at entry point, greedily descend from top layer to L+1.
      3. At each layer from L down to 0:
         - Run beam search (ef_construction candidates)
         - Connect to the M nearest neighbors bidirectionally
         - Trim neighbor lists that exceed M

    SEARCH:
      1. Greedy descent from top layer to layer 1 (single candidate).
      2. At layer 0: beam search with ef candidates, return k nearest.

    WHY FAST:
      Upper layers = highway system. You quickly zoom to the right
      neighborhood, then refine at layer 0. O(log N) average.
    """

    class Node:
        def __init__(self, item: VectorItem, max_layer: int):
            self.item      = item
            self.max_layer = max_layer
            # neighbors[layer] = list of neighbor IDs at that layer
            self.neighbors: List[List[int]] = [[] for _ in range(max_layer + 1)]

    def __init__(self, M: int = 16, ef_construction: int = 200):
        self.M              = M               # max neighbors per node per layer (except layer 0)
        self.M0             = 2 * M           # max neighbors at layer 0
        self.ef_construction = ef_construction
        self.mL             = 1.0 / math.log(M)  # level generation factor
        self.graph: Dict[int, HNSW.Node] = {}
        self.entry_point    = -1
        self.top_layer      = -1
        self._rng           = random.Random(42)  # fixed seed for reproducibility

    def _random_level(self) -> int:
        """Generate a random layer level (log-distributed: most nodes at layer 0)."""
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, query: List[float], entry_id: int,
                      ef: int, layer: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        """
        Beam search within one layer.
        Returns ef nearest candidates found, sorted ascending by distance.
        Uses two heaps: candidates (min-heap) and found (max-heap of best ef).
        """
        visited = {entry_id}
        d0 = dist_fn(query, self.graph[entry_id].item.emb)

        # candidates: min-heap, pop cheapest candidate to explore next
        candidates = [(d0, entry_id)]
        # found: max-heap of best ef results found so far
        found = [(-d0, entry_id)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            worst_found  = -found[0][0]

            # If cheapest unexplored candidate is already worse than our ef-th best, stop
            if c_dist > worst_found and len(found) >= ef:
                break

            node = self.graph.get(c_id)
            if node is None or layer >= len(node.neighbors):
                continue

            for n_id in node.neighbors[layer]:
                if n_id in visited or n_id not in self.graph:
                    continue
                visited.add(n_id)
                n_dist = dist_fn(query, self.graph[n_id].item.emb)
                if len(found) < ef or n_dist < -found[0][0]:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(found, (-n_dist, n_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        results = [(-neg_d, id_) for neg_d, id_ in found]
        results.sort()
        return results

    def insert(self, item: VectorItem, dist_fn: Callable):
        """Insert a new vector into the HNSW graph."""
        level = self._random_level()
        node  = HNSW.Node(item, level)
        self.graph[item.id] = node

        if self.entry_point == -1:
            self.entry_point = item.id
            self.top_layer   = level
            return

        ep = self.entry_point

        # Phase 1: Greedy descent from top layer to level+1 (find good entry)
        for lc in range(self.top_layer, level, -1):
            if lc < len(self.graph[ep].neighbors):
                W = self._search_layer(item.emb, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        # Phase 2: Insert at each layer from min(top, level) down to 0
        for lc in range(min(self.top_layer, level), -1, -1):
            W    = self._search_layer(item.emb, ep, self.ef_construction, lc, dist_fn)
            maxM = self.M0 if lc == 0 else self.M

            # Connect new node to its nearest neighbors
            neighbors = [id_ for _, id_ in W[:maxM]]
            node.neighbors[lc] = neighbors

            # Add bidirectional connections; trim if over limit
            for n_id in neighbors:
                n_node = self.graph.get(n_id)
                if n_node is None or lc >= len(n_node.neighbors):
                    continue
                n_node.neighbors[lc].append(item.id)
                if len(n_node.neighbors[lc]) > maxM:
                    # Keep only the maxM closest neighbors
                    scored = sorted(
                        (dist_fn(n_node.item.emb, self.graph[c].item.emb), c)
                        for c in n_node.neighbors[lc]
                        if c in self.graph
                    )
                    n_node.neighbors[lc] = [id_ for _, id_ in scored[:maxM]]

            if W:
                ep = W[0][1]

        if level > self.top_layer:
            self.top_layer   = level
            self.entry_point = item.id

    def knn(self, query: List[float], k: int, ef: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors using HNSW graph traversal."""
        if self.entry_point == -1:
            return []
        ep = self.entry_point

        # Greedy descent through upper layers
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.graph[ep].neighbors):
                W = self._search_layer(query, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        # Full beam search at layer 0
        W = self._search_layer(query, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, id: int):
        if id not in self.graph:
            return
        # Remove this node from all neighbors' lists
        for node in self.graph.values():
            for layer in node.neighbors:
                if id in layer:
                    layer.remove(id)
        # Update entry point if needed
        if self.entry_point == id:
            self.entry_point = next(
                (nid for nid in self.graph if nid != id), -1
            )
        del self.graph[id]

    def get_info(self) -> dict:
        """Return graph structure info for the /hnsw-info endpoint."""
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []
        for id_, node in self.graph.items():
            nodes.append({
                "id":       id_,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr":   node.max_layer,
            })
            for lc in range(min(node.max_layer + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(node.neighbors):
                    for n_id in node.neighbors[lc]:
                        if id_ < n_id:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": n_id, "lyr": lc})
        return {
            "topLayer":      self.top_layer,
            "nodeCount":     len(self.graph),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes":         nodes,
            "edges":         edges,
        }

    def __len__(self):
        return len(self.graph)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index — holds all 3 algorithms)
# =====================================================================

class VectorDB:
    """
    Unified interface wrapping BruteForce, KDTree, and HNSW simultaneously.
    All three indexes are kept in sync; searches pick one based on 'algo' param.
    """
    def __init__(self, dims: int):
        self.dims   = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf     = BruteForce()
        self.kdt    = KDTree(dims)
        self.hnsw   = HNSW(M=16, ef_construction=200)
        self.lock   = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str,
               emb: List[float], dist_fn: Callable) -> int:
        with self.lock:
            item = VectorItem(self._next_id, metadata, category, emb)
            self._next_id += 1
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, id: int) -> bool:
        with self.lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.bf.remove(id)
            self.hnsw.remove(id)
            # KD-Tree must be fully rebuilt on deletion (no in-place delete)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, query: List[float], k: int,
               metric: str, algo: str) -> dict:
        with self.lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(query, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(query, k, dist_fn)
            else:
                raw = self.hnsw.knn(query, k, 50, dist_fn)

            latency_us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for dist_val, id_ in raw:
                if id_ in self.store:
                    v = self.store[id_]
                    hits.append({
                        "id":        v.id,
                        "metadata":  v.metadata,
                        "category":  v.category,
                        "distance":  round(dist_val, 6),
                        "embedding": v.emb,
                    })
            return {"results": hits, "latencyUs": latency_us,
                    "algo": algo, "metric": metric}

    def benchmark(self, query: List[float], k: int, metric: str) -> dict:
        with self.lock:
            dist_fn = get_dist_fn(metric)
            def time_fn(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bruteforceUs": time_fn(lambda: self.bf.knn(query, k, dist_fn)),
                "kdtreeUs":     time_fn(lambda: self.kdt.knn(query, k, dist_fn)),
                "hnswUs":       time_fn(lambda: self.hnsw.knn(query, k, 50, dist_fn)),
                "itemCount":    len(self.store),
            }

    def all_items(self) -> List[VectorItem]:
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.lock:
            return self.hnsw.get_info()

    def __len__(self):
        return len(self.store)

# =====================================================================
#  OLLAMA CLIENT  — talks to local Ollama REST API
# =====================================================================

class OllamaClient:
    """
    Wraps Ollama's HTTP API.
    Install Ollama: https://ollama.com
    Pull models:   ollama pull nomic-embed-text
                   ollama pull llama3.2
    """
    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self.host        = host
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        """Convert text to a vector using nomic-embed-text (768 dims)."""
        try:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("embedding", [])
        except Exception:
            pass
        return []

    def generate(self, prompt: str) -> str:
        """Generate a text response using llama3.2."""
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code == 200:
                return r.json().get("response", "")
        except Exception:
            pass
        return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real 768D Ollama embeddings
# =====================================================================

class DocItem:
    def __init__(self, id: int, title: str, text: str, emb: List[float]):
        self.id    = id
        self.title = title
        self.text  = text
        self.emb   = emb

class DocumentDB:
    """
    Stores real document chunks with their Ollama embeddings.
    Uses HNSW for semantic search (cosine distance).
    Falls back to BruteForce for small sets (<10 docs).
    """
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw  = HNSW(M=16, ef_construction=200)
        self.bf    = BruteForce()
        self.lock  = threading.Lock()
        self._next_id = 1
        self.dims  = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self._next_id, title, text, emb)
            self._next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, query_emb: List[float], k: int,
               max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(query_emb, k, cosine)
            else:
                raw = self.hnsw.knn(query_emb, k, 50, cosine)
            return [
                (d, self.store[id_])
                for d, id_ in raw
                if id_ in self.store and d <= max_dist
            ]

    def remove(self, id: int) -> bool:
        with self.lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.hnsw.remove(id)
            self.bf.remove(id)
            return True

    def all_docs(self) -> List[DocItem]:
        with self.lock:
            return list(self.store.values())

    def __len__(self):
        return len(self.store)

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    """
    Split a long document into overlapping chunks for better RAG retrieval.
    Overlap ensures no information is lost at chunk boundaries.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step   = chunk_words - overlap_words
    i      = 0
    while i < len(words):
        end   = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  DEMO DATA  — 20 pre-loaded 16D semantic vectors
# =====================================================================

# Dimension groups: [0-3]=CS, [4-7]=Math, [8-11]=Food, [12-15]=Sports
# Each vector has high values in its category's dims and low elsewhere.
# This creates visible semantic clusters in the PCA scatter plot.

DEMO_VECTORS = [
    # --- Computer Science ---
    ("Linked List: nodes connected by pointers", "cs",
     [0.90,0.85,0.72,0.68, 0.12,0.08,0.15,0.10, 0.05,0.08,0.06,0.09, 0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88,0.82,0.78,0.74, 0.15,0.10,0.08,0.12, 0.06,0.07,0.08,0.05, 0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82,0.76,0.88,0.80, 0.20,0.18,0.12,0.09, 0.07,0.06,0.08,0.07, 0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85,0.80,0.75,0.82, 0.18,0.14,0.10,0.08, 0.06,0.09,0.07,0.06, 0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87,0.78,0.70,0.76, 0.13,0.11,0.09,0.14, 0.08,0.07,0.06,0.08, 0.07,0.10,0.08,0.09]),
    # --- Mathematics ---
    ("Calculus: derivatives integrals and limits", "math",
     [0.12,0.15,0.18,0.10, 0.91,0.86,0.78,0.72, 0.08,0.06,0.07,0.09, 0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20,0.18,0.15,0.12, 0.88,0.90,0.82,0.76, 0.09,0.07,0.08,0.06, 0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15,0.12,0.20,0.18, 0.84,0.80,0.88,0.82, 0.07,0.08,0.06,0.10, 0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22,0.16,0.14,0.20, 0.80,0.85,0.76,0.90, 0.08,0.09,0.07,0.06, 0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18,0.20,0.16,0.14, 0.86,0.78,0.84,0.80, 0.06,0.07,0.09,0.08, 0.06,0.09,0.10,0.07]),
    # --- Food ---
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08,0.06,0.09,0.07, 0.07,0.08,0.06,0.09, 0.90,0.86,0.78,0.72, 0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06,0.08,0.07,0.09, 0.09,0.06,0.08,0.07, 0.86,0.90,0.82,0.76, 0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09,0.07,0.06,0.08, 0.08,0.09,0.07,0.06, 0.82,0.78,0.90,0.84, 0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07,0.09,0.08,0.06, 0.06,0.07,0.09,0.08, 0.78,0.82,0.86,0.90, 0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06,0.07,0.10,0.09, 0.10,0.06,0.07,0.10, 0.85,0.80,0.76,0.82, 0.09,0.07,0.10,0.06]),
    # --- Sports ---
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09,0.07,0.08,0.10, 0.08,0.09,0.07,0.06, 0.08,0.07,0.09,0.06, 0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07,0.09,0.06,0.08, 0.09,0.07,0.10,0.08, 0.07,0.09,0.08,0.07, 0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08,0.06,0.09,0.07, 0.07,0.08,0.06,0.09, 0.09,0.06,0.07,0.08, 0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25,0.20,0.22,0.18, 0.22,0.18,0.20,0.15, 0.06,0.08,0.07,0.09, 0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06,0.08,0.07,0.09, 0.08,0.06,0.09,0.07, 0.10,0.08,0.06,0.07, 0.85,0.82,0.86,0.80]),
]

def load_demo(db: VectorDB):
    dist_fn = get_dist_fn("cosine")
    for metadata, category, emb in DEMO_VECTORS:
        db.insert(metadata, category, emb, dist_fn)

# =====================================================================
#  HTTP SERVER
# =====================================================================

def make_handler(db: VectorDB, doc_db: DocumentDB, ollama: OllamaClient, DIMS: int):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, format, *args):
            pass  # suppress default access log noise

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin",  "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, data, status: int = 200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                return json.loads(raw)
            except Exception:
                return {}

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            parsed = urlparse(self.path)
            path   = parsed.path
            params = parse_qs(parsed.query)

            def p(key, default=""):
                return params.get(key, [default])[0]

            # ── Serve index.html ──────────────────────────────────────
            if path == "/" or path == "/index.html":
                try:
                    with open("index.html", "rb") as f:
                        html = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                except FileNotFoundError:
                    self.send_response(404)
                    self.end_headers()
                return

            # ── /search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw ──────
            elif path == "/search":
                try:
                    q = [float(x) for x in p("v").split(",") if x]
                except Exception:
                    q = []
                if len(q) != DIMS:
                    return self._json({"error": f"need {DIMS}D vector"}, 400)
                k      = int(p("k", "5"))
                metric = p("metric", "cosine")
                algo   = p("algo",   "hnsw")
                self._json(db.search(q, k, metric, algo))

            # ── /items ───────────────────────────────────────────────
            elif path == "/items":
                items = db.all_items()
                self._json([{
                    "id":        v.id,
                    "metadata":  v.metadata,
                    "category":  v.category,
                    "embedding": v.emb,
                } for v in items])

            # ── /benchmark?v=...&k=5&metric=cosine ───────────────────
            elif path == "/benchmark":
                try:
                    q = [float(x) for x in p("v").split(",") if x]
                except Exception:
                    q = []
                if len(q) != DIMS:
                    return self._json({"error": f"need {DIMS}D vector"}, 400)
                k      = int(p("k", "5"))
                metric = p("metric", "cosine")
                self._json(db.benchmark(q, k, metric))

            # ── /hnsw-info ───────────────────────────────────────────
            elif path == "/hnsw-info":
                self._json(db.hnsw_info())

            # ── /doc/list ────────────────────────────────────────────
            elif path == "/doc/list":
                docs = doc_db.all_docs()
                self._json([{
                    "id":      d.id,
                    "title":   d.title,
                    "preview": d.text[:120] + ("…" if len(d.text) > 120 else ""),
                    "words":   len(d.text.split()),
                } for d in docs])

            # ── /status ──────────────────────────────────────────────
            elif path == "/status":
                up = ollama.is_available()
                self._json({
                    "ollamaAvailable": up,
                    "embedModel":      ollama.embed_model,
                    "genModel":        ollama.gen_model,
                    "docCount":        len(doc_db),
                    "docDims":         doc_db.dims,
                    "demoDims":        DIMS,
                    "demoCount":       len(db),
                })

            # ── /stats ───────────────────────────────────────────────
            elif path == "/stats":
                self._json({
                    "count":      len(db),
                    "dims":       DIMS,
                    "algorithms": ["bruteforce", "kdtree", "hnsw"],
                    "metrics":    ["euclidean", "cosine", "manhattan"],
                })

            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            body = self._read_body()

            # ── /insert ──────────────────────────────────────────────
            if path == "/insert":
                meta = body.get("metadata", "")
                cat  = body.get("category", "")
                emb  = body.get("embedding", [])
                if not meta or len(emb) != DIMS:
                    return self._json({"error": "invalid body"}, 400)
                id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
                self._json({"id": id_})

            # ── /doc/insert ──────────────────────────────────────────
            elif path == "/doc/insert":
                title = body.get("title", "")
                text  = body.get("text", "")
                if not title or not text:
                    return self._json({"error": "need title and text"}, 400)

                chunks = chunk_text(text, 250, 30)
                ids    = []
                for i, chunk in enumerate(chunks):
                    emb = ollama.embed(chunk)
                    if not emb:
                        return self._json({
                            "error": "Ollama unavailable. Install from https://ollama.com "
                                     "then run: ollama pull nomic-embed-text && ollama pull llama3.2"
                        }, 503)
                    chunk_title = (f"{title} [{i+1}/{len(chunks)}]"
                                   if len(chunks) > 1 else title)
                    ids.append(doc_db.insert(chunk_title, chunk, emb))

                self._json({
                    "ids":    ids,
                    "chunks": len(chunks),
                    "dims":   doc_db.dims,
                })

            # ── /doc/search ──────────────────────────────────────────
            elif path == "/doc/search":
                question = body.get("question", "")
                k        = body.get("k", 3)
                if not question:
                    return self._json({"error": "need question"}, 400)
                q_emb = ollama.embed(question)
                if not q_emb:
                    return self._json({"error": "Ollama unavailable"}, 503)
                hits = doc_db.search(q_emb, k)
                self._json({"contexts": [
                    {"id": d.id, "title": d.title, "distance": round(dist, 4)}
                    for dist, d in hits
                ]})

            # ── /doc/ask  (full RAG pipeline) ───────────────────────
            elif path == "/doc/ask":
                question = body.get("question", "")
                k        = body.get("k", 3)
                if not question:
                    return self._json({"error": "need question"}, 400)

                # Step 1: Embed the question
                q_emb = ollama.embed(question)
                if not q_emb:
                    return self._json({"error": "Ollama unavailable"}, 503)

                # Step 2: Retrieve top-k semantically similar chunks
                hits = doc_db.search(q_emb, k)

                # Step 3: Build the RAG prompt
                context = "\n\n".join(
                    f"[{i+1}] {d.title}:\n{d.text}"
                    for i, (_, d) in enumerate(hits)
                )
                prompt = (
                    "You are a helpful assistant. Answer the user's question directly. "
                    "Use the provided context if it contains relevant information. "
                    "If it doesn't, just use your own general knowledge. "
                    "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
                    "'the context doesn't mention'. Just answer the question naturally.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\nAnswer:"
                )

                # Step 4: Generate answer via LLM
                answer = ollama.generate(prompt)

                # Step 5: Return answer + sources
                self._json({
                    "answer":   answer,
                    "model":    ollama.gen_model,
                    "contexts": [
                        {
                            "id":       d.id,
                            "title":    d.title,
                            "text":     d.text,
                            "distance": round(dist, 4),
                        }
                        for dist, d in hits
                    ],
                    "docCount": len(doc_db),
                })

            else:
                self._json({"error": "not found"}, 404)

        def do_DELETE(self):
            path = self.path

            # ── DELETE /delete/:id ────────────────────────────────────
            if path.startswith("/delete/"):
                try:
                    id_ = int(path.split("/")[-1])
                except ValueError:
                    return self._json({"error": "bad id"}, 400)
                ok = db.remove(id_)
                self._json({"ok": ok})

            # ── DELETE /doc/delete/:id ───────────────────────────────
            elif path.startswith("/doc/delete/"):
                try:
                    id_ = int(path.split("/")[-1])
                except ValueError:
                    return self._json({"error": "bad id"}, 400)
                ok = doc_db.remove(id_)
                self._json({"ok": ok})

            else:
                self._json({"error": "not found"}, 404)

    return Handler

# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    DIMS = 16

    print("=== VectorDB Engine (Python Edition) ===")
    print("Initializing data structures...")

    db     = VectorDB(DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient()

    load_demo(db)

    ollama_up = ollama.is_available()
    print(f"http://localhost:8080")
    print(f"{len(db)} demo vectors | {DIMS} dims | HNSW + KD-Tree + BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}   gen model: {ollama.gen_model}")
    print("Press Ctrl+C to stop.\n")

    Handler = make_handler(db, doc_db, ollama, DIMS)
    server  = HTTPServer(("0.0.0.0", 8080), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")