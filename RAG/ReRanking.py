import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama

# -----------------------------
# DEBUG: Initialize Redis memory
# -----------------------------
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
print("[DEBUG] Redis connected")

# -----------------------------
# Load embedding + reranker models
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("[DEBUG] Models loaded")

# -----------------------------
# Load local LLM
# -----------------------------
llm = Llama(model_path="./models/llama-2-7b.gguf", n_ctx=2048)
print("[DEBUG] Local LLM loaded")

# -----------------------------
# Sample documents
# -----------------------------
docs = [
    "Re-ranking improves retrieval accuracy in RAG systems.",
    "Vector databases store embeddings for similarity search.",
    "Cross-encoders score query and document pairs together.",
    "Redis can be used for conversational memory."
]

# -----------------------------
# Embed documents
# -----------------------------
doc_embeddings = embedder.encode(docs)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
print("[DEBUG] FAISS index built")

# -----------------------------
# User query
# -----------------------------
query = "How does re-ranking improve RAG accuracy?"
query_embedding = embedder.encode([query])

# -----------------------------
# Step 1: Retrieve top-K
# -----------------------------
K = 4
distances, indices = index.search(np.array(query_embedding), K)
retrieved_docs = [docs[i] for i in indices[0]]

print("[DEBUG] Retrieved docs (pre re-ranking):")
for d in retrieved_docs:
    print(" -", d)

# -----------------------------
# Step 2: Re-rank using cross-encoder
# -----------------------------
pairs = [[query, doc] for doc in retrieved_docs]
scores = reranker.predict(pairs)

ranked_docs = sorted(
    zip(retrieved_docs, scores),
    key=lambda x: x[1],
    reverse=True
)

print("[DEBUG] Re-ranked docs:")
for doc, score in ranked_docs:
    print(f" - Score: {score:.4f} | {doc}")

# -----------------------------
# Step 3: Store context in Redis
# -----------------------------
top_context = "\n".join([doc for doc, _ in ranked_docs[:2]])
redis_client.set("last_context", top_context)
print("[DEBUG] Context stored in Redis")

# -----------------------------
# Step 4: Generate answer
# -----------------------------
prompt = f"""
Use the context below to answer the question.

Context:
{top_context}

Question:
{query}
"""

response = llm(prompt)
print("[DEBUG] LLM response generated")

print("\nFinal Answer:\n", response["choices"][0]["text"])

