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
# Example structured document
# -----------------------------
document = {
    "Policy Overview": "This policy explains how customer data must be handled.",
    "User Consent": "Consent must be obtained before collecting personal data.",
    "Compliance Risks": "Violations can lead to penalties and legal action."
}

# -----------------------------
# Context-aware chunking
# -----------------------------
chunks = []
for section, text in document.items():
    chunk = f"Section: {section}\nContent: {text}"
    chunks.append(chunk)
    redis_client.hset("chunks", section, text)
    print(f"[DEBUG] Chunk created for section: {section}")

# -----------------------------
# Embed chunks
# -----------------------------
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("[DEBUG] FAISS index built with context-aware chunks")

# -----------------------------
# Query
# -----------------------------
query = "Why is user consent required?"
q_emb = embedder.encode([query])
_, idx = index.search(np.array(q_emb), 2)

retrieved_chunks = [chunks[i] for i in idx[0]]
print("[DEBUG] Retrieved chunks:", retrieved_chunks)

# -----------------------------
# Generate final answer
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
