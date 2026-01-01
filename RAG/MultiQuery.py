import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama

# -----------------------------
# DEBUG: Connect to Redis
# -----------------------------
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
print("[DEBUG] Redis connected")

# -----------------------------
# Load models
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = Llama(model_path="./models/llama-2-7b.gguf", n_ctx=2048)
print("[DEBUG] Models loaded")

# -----------------------------
# Knowledge base
# -----------------------------
docs = [
    "Multi-query RAG improves recall by exploring multiple intents.",
    "Re-ranking improves precision after retrieval.",
    "Vector databases store embeddings for similarity search.",
    "Enterprise RAG systems balance recall and precision."
]

# -----------------------------
# Build vector index
# -----------------------------
embeddings = embedder.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("[DEBUG] FAISS index built")

# -----------------------------
# User query
# -----------------------------
query = "How can RAG improve retrieval quality?"
redis_client.set("original_query", query)

# -----------------------------
# Step 1: Generate multiple queries
# -----------------------------
prompt = f"""
Generate 3 different search queries that explore
different aspects of the following question:

{query}
"""

raw = llm(prompt)["choices"][0]["text"]
expanded_queries = [q.strip("- ") for q in raw.split("\n") if q.strip()]
expanded_queries.append(query)

redis_client.set("expanded_queries", str(expanded_queries))
print("[DEBUG] Expanded queries:", expanded_queries)

# -----------------------------
# Step 2: Retrieve documents
# -----------------------------
retrieved = []

for q in expanded_queries:
    q_emb = embedder.encode([q])
    _, idx = index.search(np.array(q_emb), 2)
    for i in idx[0]:
        retrieved.append(docs[i])

print("[DEBUG] Retrieved docs (pre dedup):", retrieved)

# -----------------------------
# Step 3: Deduplicate
# -----------------------------
retrieved = list(set(retrieved))
redis_client.set("dedup_docs", str(retrieved))
print("[DEBUG] Retrieved docs (deduped):", retrieved)

# -----------------------------
# Step 4: Re-rank
# -----------------------------
pairs = [[query, doc] for doc in retrieved]
scores = reranker.predict(pairs)

ranked = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)
top_docs = [doc for doc, _ in ranked[:2]]

print("[DEBUG] Re-ranked docs:", ranked)

# -----------------------------
# Step 5: Final answer
# -----------------------------
context = "\n".join(top_docs)

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

response = llm(prompt)
print("\nFinal Answer:\n", response["choices"][0]["text"])
