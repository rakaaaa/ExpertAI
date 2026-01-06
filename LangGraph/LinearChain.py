"""
Linear Chain Strategy using LangGraph
-----------------------------------
Pattern:
Input → Retrieve → Reason → Generate → Validate → Output

Key Properties:
- Deterministic flow
- No branching
- Single shared state
- MongoDB-backed memory
- Local LLM (Ollama)

This is a SAFE starting point for enterprise GenAI systems.
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient

# -----------------------------
# MongoDB: Long-term memory
# -----------------------------
print("[INIT] Connecting to MongoDB...")
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["langgraph_memory"]
collection = db["linear_chain_sessions"]
print("[INIT] MongoDB connected")

# -----------------------------
# State definition
# -----------------------------
class GraphState(dict):
    """
    Single shared state object.
    Keeps the workflow debuggable and auditable.
    """
    pass

# -----------------------------
# Local LLM (Ollama)
# -----------------------------
print("[INIT] Initializing local LLM...")
llm = Ollama(model="mistral")
print("[INIT] LLM ready")

# -----------------------------
# Node 1: Retrieve context
# -----------------------------
def retrieve_context(state: GraphState):
    print("[NODE] Retrieve Context started")

    session_id = state["session_id"]
    record = collection.find_one({"session_id": session_id})

    state["previous_context"] = record["context"] if record else ""
    print(f"[DEBUG] Retrieved context: {state['previous_context']}")

    return state

# -----------------------------
# Node 2: Reasoning
# -----------------------------
def reason(state: GraphState):
    print("[NODE] Reasoning started")

    prompt = f"""
    Previous Context:
    {state['previous_context']}

    User Question:
    {state['question']}

    Provide a clear and concise answer.
    """

    state["raw_answer"] = llm(prompt)
    print("[DEBUG] LLM reasoning completed")

    return state

# -----------------------------
# Node 3: Validation
# -----------------------------
def validate(state: GraphState):
    print("[NODE] Validation started")

    # Simple deterministic validation rule
    if len(state["raw_answer"].strip()) == 0:
        state["final_answer"] = "Unable to generate a valid response."
        print("[WARN] Empty response detected")
    else:
        state["final_answer"] = state["raw_answer"]

    print("[DEBUG] Validation completed")
    return state

# -----------------------------
# Node 4: Persist memory
# -----------------------------
def persist(state: GraphState):
    print("[NODE] Persisting memory")

    collection.update_one(
        {"session_id": state["session_id"]},
        {"$set": {"context": state["final_answer"]}},
        upsert=True
    )

    print("[DEBUG] Memory persisted to MongoDB")
    return state

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building LangGraph...")

graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve_context)
graph.add_node("reason", reason)
graph.add_node("validate", validate)
graph.add_node("persist", persist)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "reason")
graph.add_edge("reason", "validate")
graph.add_edge("validate", "persist")
graph.add_edge("persist", END)

app = graph.compile()

print("[INIT] LangGraph compiled successfully")

# -----------------------------
# Run the graph
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Invoking Linear Chain Strategy")

    result = app.invoke({
        "session_id": "user-001",
        "question": "Explain LangGraph in simple terms"
    })

    print("\n[RESULT] Final Answer:\n")
    print(result["final_answer"])
