import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# -----------------------------
# DEBUG: Initialize Redis memory
# -----------------------------
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
print("[DEBUG] Redis connected")

# -----------------------------
# Load embedding model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[DEBUG] Embedding model loaded")

# -----------------------------
# Load local LLM (agent brain)
# -----------------------------
llm = Llama(model_path="./models/llama-2-7b.gguf", n_ctx=2048)
print("[DEBUG] Local LLM loaded")

# -----------------------------
# Knowledge base
# -----------------------------
documents = {
    "vector": [
        "Vector search uses embeddings to find semantically similar text.",
        "FAISS enables efficient similarity search at scale."
    ],
    "keyword": [
        "Keyword search is precise but brittle.",
        "Exact match search works well for IDs and codes."
    ]
}

# -----------------------------
# Build vector index
# -----------------------------
vectors = embedder.encode(documents["vector"])
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))
print("[DEBUG] Vector index created")

# -----------------------------
# Agent: decide retrieval strategy
# -----------------------------
def agent_decide(query):
    print("[DEBUG] Agent deciding retrieval strategy")
    prompt = f"""
Classify the query as one of:
- semantic
- keyword
- hybrid

Query: {query}
Answer with one word.
"""
    decision = llm(prompt)["choices"][0]["text"].strip().lower()
    redis_client.set("last_decision", decision)
    print(f"[DEBUG] Agent decision: {decision}")
    return decision

# -----------------------------
# Retrieval executor
# -----------------------------
def retrieve(query):
    strategy = agent_decide(query)

    if "semantic" in strategy or "hybrid" in strategy:
        q_emb = embedder.encode([query])
        _, idx = index.search(np.array(q_emb), 2)
        results = [documents["vector"][i] for i in idx[0]]
    else:
        results = documents["keyword"]

    redis_client.set("last_retrieval", str(results))
    print("[DEBUG] Retrieval completed")
    return results

# -----------------------------
# Run Agentic RAG
# -----------------------------
query = "How should I choose a retrieval method for RAG?"
context = retrieve(query)

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

response = llm(prompt)
print("\nFinal Answer:\n", response["choices"][0]["text"])
