What to install?
# pip install redis sentence-transformers faiss-cpu llama-cpp-python
# docker run -p 6379:6379 redis

#Architecture Used
#Redis → Conversation memory + metadata
#FAISS → Vector similarity search
#SentenceTransformers → Embeddings + cross-encoder re-ranker
#Local LLM → llama.cpp compatible model


How to Run:
1. Start Redis
2. Download a llama.cpp compatible model
3. Run the script
4. Observe debug logs for retrieval → re-ranking → generation
