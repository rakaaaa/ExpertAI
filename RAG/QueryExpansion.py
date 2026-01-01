import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# -----------------------------
# DEBUG: Redis connection
# -----------------------------
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
print("[DEBUG] Redis connected")

# -----------------------------
# Load models
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path="./models/llama-2-7b.gguf", n_ctx=2048)
print("[DEBUG] Models loaded")

# -----------------------------
# Knowledge base
# -----------------------------
docs = [
    "Re-ranking improves precision in RAG systems.",
    "Query expansion improves recall for ambiguous queries.",
    "Vector search retrieves semantically similar documents.",
    "Enterprise RAG systems balance recall and precision."
]

embeddings = embedder.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("[DEBUG] Vector index created")

# -----------------------------
# User query
# -----------------------------
query = "How to improve RAG retrieval?"
redis_client.set("original_query", query)

# -----------------------------
# Step 1: Expand query using LLM
# -----------------------------
prompt = f"""
Generate 3 alternative queries that clarify intent and add context.

Original query:
{query}
"""
expanded_text = llm(prompt)["choices"][0]["text"]
expanded_queries = [q.strip("- ") for q in expanded_text.split("\n") if q.strip()]

redis_client.set("expanded_queries", str(expanded_queries))
print("[DEBUG] Expanded queries:", expanded_queries)

# -----------------------------
# Step 2: Retrieve for each query
# -----------------------------
results = set()

for q in expanded_queries + [query]:
    q_emb = embedder.encode([q])
    _, idx = index.search(np.array(q_emb), 2)
    for i in idx[0]:
        results.add(docs[i])

print("[DEBUG] Retrieved docs (pre re-ranking):", results)

# -----------------------------
# Step 3: Generate final answer
# -----------------------------
context = "\n".join(results)

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

response = llm(prompt)
print("\nFinal Answer:\n", response["choices"][0]["text"])
