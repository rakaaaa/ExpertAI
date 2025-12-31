import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# -----------------------------
# DEBUG: Redis for graph memory
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
# Simple knowledge graph
# -----------------------------
knowledge_graph = {
    "GDPR": ["Data Privacy", "EU Regulation"],
    "Data Privacy": ["User Consent", "PII"],
    "PII": ["Compliance Risk"]
}

# Store graph in Redis
for entity, relations in knowledge_graph.items():
    redis_client.sadd(entity, *relations)

print("[DEBUG] Knowledge graph stored in Redis")

# -----------------------------
# Vector documents
# -----------------------------
docs = [
    "GDPR regulates personal data usage in the EU.",
    "PII refers to personally identifiable information.",
    "Compliance failures can lead to legal penalties."
]

embeddings = embedder.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("[DEBUG] Vector index created")

# -----------------------------
# Hybrid retrieval
# -----------------------------
query = "How does GDPR relate to compliance risk?"

# Step 1: Vector recall
q_emb = embedder.encode([query])
_, idx = index.search(np.array(q_emb), 2)
vector_context = [docs[i] for i in idx[0]]

print("[DEBUG] Vector retrieval results:", vector_context)

# Step 2: Graph traversal
entities = ["GDPR"]
graph_context = []

for e in entities:
    relations = redis_client.smembers(e)
    graph_context.extend(relations)

print("[DEBUG] Graph traversal results:", graph_context)

# -----------------------------
# Final generation
# -----------------------------
prompt = f"""
Answer the question using both semantic context and graph relationships.

Vector Context:
{vector_context}

Graph Relationships:
{graph_context}

Question:
{query}
"""

response = llm(prompt)
print("\nFinal Answer:\n", response["choices"][0]["text"])
