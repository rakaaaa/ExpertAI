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
# Example document
# -----------------------------
document = """
Section 1: Data Privacy Policy
This policy defines how user data must be handled.

Section 2: User Consent
User consent is required before collecting personal data.

Section 3: Compliance Risks
Failure to follow this policy can lead to penalties.
"""

# -----------------------------
# Chunking (simple example)
# -----------------------------
chunks = [
    "This policy defines how user data must be handled.",
    "User consent is required before collecting personal data.",
    "Failure to follow this policy can lead to penalties."
]

# -----------------------------
# Generate contextual summaries
# -----------------------------
contextual_chunks = []

for i, chunk in enumerate(chunks):
    prompt = f"""
Summarize how this text fits into the overall document:

Document:
{document}

Chunk:
{chunk}
"""
    summary = llm(prompt)["choices"][0]["text"].strip()
    contextual_chunks.append(f"Context: {summary}\nChunk: {chunk}")
    print(f"[DEBUG] Context generated for chunk {i}")

# -----------------------------
# Embed enriched chunks
# -----------------------------
embeddings = embedder.encode(contextual_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("[DEBUG] FAISS index created")

# -----------------------------
# Query
# -----------------------------
query = "Why is user consent important?"
q_emb = embedder.encode([query])
_, idx = index.search(np.array(q_emb), 2)

retrieved_context = [contextual_chunks[i] for i in idx[0]]
print("[DEBUG] Retrieved contextual chunks:", retrieved_context)

# -----------------------------
# Generate final answer
# -----------------------------
prompt = f"""
Answer the question using the context below.

Context:
{retrieved_context}

Question:
{query}
"""

response = llm(prompt)
print("\nFinal Answer:\n", response["choices"][0]["text"])
