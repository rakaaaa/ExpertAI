import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# -----------------------------
# DEBUG: Connect to Redis
# -----------------------------
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
print("[DEBUG] Connected to Redis")

# -----------------------------
# Load fine-tuned embedding model
# -----------------------------
embedder = SentenceTransformer("./models/legal-embeddings-v1")
print("[DEBUG] Fine-tuned embedding model loaded")

# -----------------------------
# Load local LLM
# -----------------------------
llm = Llama(model_path="./models/llama-2-7b.gguf", n_ctx=2048)
print("[DEBUG] Local LLM loaded")

# -----------------------------
# Domain-specific corpus
# -----------------------------
documents = [
    "Explicit consent is required before processing personal data.",
    "Consent may be withdrawn without penalty.",
    "Failure to comply may result in regulatory fines."
]

# -----------------------------
# Embed documents
# -----------------------------
doc_embeddings = embedder.encode(documents)
print("[DEBUG] Document embeddings generated")

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))
print("[DEBUG] FAISS index created")

# -----------------------------
# Store embeddings metadata
# -----------------------------
for i, doc in enumerate(documents):
    redis_client.hset("documents", f"doc_{i}", doc)
print("[DEBUG] Documents stored in Redis")

# -----------------------------
# Query
# -----------------------------
query = "What are the consequences of violating consent requirements?"
query_embedding = embedder.encode([query])
print("[DEBUG] Query embedded using fine-tuned model")

# -----------------------------
# Retrieve top-k
# -----------------------------
_, idx = index.search(np.array(query_embedding), 2)
retrieved_docs = [documents[i] for i in idx[0]]
print("[DEBUG] Retrieved documents:", retrieved_docs)

# -----------------------------
# Generate answer
# -----------------------------
prompt = f"""
Answer the question using only the context below.

Context:
{retrieved_docs}

Question:
{query}
"""

response = llm(prompt)
final_answer = response["choices"][0]["text"]

print("\nFinal Answer:\n", final_answer)
