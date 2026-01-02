import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
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
llm = Llama(model_path="./models/llama-2-7b.gguf", n_ctx=2048)
print("[DEBUG] Models loaded")

# -----------------------------
# Full document (critical corpus)
# -----------------------------
document = """
This contract defines the obligations of all parties.
Section 1 explains data usage.
Section 2 defines consent requirements.
Section 3 outlines penalties for violations.
"""

# -----------------------------
# Step 1: Global document context
# -----------------------------
prompt = f"""
Summarize the core intent and structure of this document:

{document}
"""
global_context = llm(prompt)["choices"][0]["text"].strip()
redis_client.set("global_context", global_context)

print("[DEBUG] Global document context generated")
print("[DEBUG] Global context:", global_context)

# -----------------------------
# Step 2: Late chunking
# -----------------------------
chunks = [
    "Section 1 explains data usage.",
    "Section 2 defines consent requirements.",
    "Section 3 outlines penalties for violations."
]

enriched_chunks = []
for i, chunk in enumerate(chunks):
    enriched = f"""
Global Context:
{global_context}

Chunk:
{chunk}
"""
    enriched_chunks.append(enriched)
    redis_client.hset("late_chunks", f"chunk_{i}", enriched)
    print(f"[DEBUG] Enriched chunk {i} stored")

# -----------------------------
# Step 3: Embed enriched chunks
# -----------------------------
embeddings = embedder.encode(enriched_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("[DEBUG] FAISS index built for late-chunked data")

# -----------------------------
# Query
# -----------------------------
query = "What happens if consent rules are violated?"
q_emb = embedder.encode([query])
_, idx = index.search(np.array(q_emb), 2)

retrieved_chunks = [enriched_chunks[i] for i in idx[0]]
print("[DEBUG] Retrieved enriched chunks:", retrieved_chunks)

# -----------------------------
# Final answer generation
# -----------------------------
prompt = f"""
Answer the question using the context below.

Context:
{retrieved_chunks}

Question:
{query}
"""

response = llm(prompt)
print("\nFinal Answer:\n", response["choices"][0]["text"])
