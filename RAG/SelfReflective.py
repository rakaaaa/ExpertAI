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
# Load models
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path="./models/llama-2-7b.gguf", n_ctx=2048)
print("[DEBUG] Models loaded")

# -----------------------------
# Knowledge base
# -----------------------------
documents = [
    "Consent must be explicitly recorded.",
    "Consent can be withdrawn at any time.",
    "Violations may lead to penalties."
]

embeddings = embedder.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("[DEBUG] Vector index created")

# -----------------------------
# User query
# -----------------------------
query = "What happens if consent rules are broken?"
q_emb = embedder.encode([query])

# -----------------------------
# Initial retrieval
# -----------------------------
_, idx = index.search(np.array(q_emb), 2)
context = [documents[i] for i in idx[0]]
print("[DEBUG] Initial context:", context)

# -----------------------------
# Generate initial answer
# -----------------------------
answer_prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""
initial_answer = llm(answer_prompt)["choices"][0]["text"]
print("[DEBUG] Initial answer generated")

# -----------------------------
# Self-reflection step
# -----------------------------
reflection_prompt = f"""
Evaluate the answer below.
Is it complete and reliable?
Reply with CONFIDENT or UNCERTAIN and explain briefly.

Answer:
{initial_answer}
"""
reflection = llm(reflection_prompt)["choices"][0]["text"]
print("[DEBUG] Reflection output:", reflection)

# -----------------------------
# Decision loop
# -----------------------------
if "UNCERTAIN" in reflection:
    print("[DEBUG] Low confidence detected. Triggering re-retrieval.")

    _, idx = index.search(np.array(q_emb), 3)
    expanded_context = [documents[i] for i in idx[0]]

    retry_prompt = f"""
Answer again using the expanded context.

Context:
{expanded_context}

Question:
{query}
"""
    final_answer = llm(retry_prompt)["choices"][0]["text"]
else:
    print("[DEBUG] High confidence. Accepting initial answer.")
    final_answer = initial_answer

# -----------------------------
# Store result
# -----------------------------
redis_client.set("last_answer", final_answer)
print("\nFinal Answer:\n", final_answer)
