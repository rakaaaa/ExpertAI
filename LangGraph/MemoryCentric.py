"""
Memory-Centric Strategy (Long-Running Agents)
--------------------------------------------
Demonstrates:
- Ephemeral memory (reasoning)
- Session memory (conversation)
- Long-term memory (MongoDB)
- Summarization before persistence
- Local LLM (Ollama)
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient
from datetime import datetime

# -----------------------------
# MongoDB Setup (Long-Term Memory)
# -----------------------------
print("[INIT] Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph_memory"]
memory_collection = db["long_term_memory"]
print("[INIT] MongoDB connected")

# -----------------------------
# LLM Setup (Local)
# -----------------------------
print("[INIT] Initializing local LLM...")
llm = Ollama(model="mistral")
print("[INIT] LLM ready")

# -----------------------------
# Graph State Definition
# -----------------------------
class GraphState(dict):
    """
    Memory-aware state object
    """
    pass

# -----------------------------
# Node 1: Load Long-Term Memory
# -----------------------------
def load_memory(state: GraphState):
    print("[MEMORY] Loading long-term memory")

    records = memory_collection.find(
        {"user_id": state["user_id"]}
    ).sort("timestamp", -1).limit(3)

    state["long_term_memory"] = [
        r["summary"] for r in records
    ]

    print(f"[DEBUG] Loaded {len(state['long_term_memory'])} memory items")
    return state

# -----------------------------
# Node 2: Reasoning (Ephemeral)
# -----------------------------
def reason(state: GraphState):
    print("[LLM] Reasoning with memory")

    context = "\n".join(state.get("long_term_memory", []))

    prompt = f"""
    You are an assistant with memory.

    Long-term memory:
    {context}

    User message:
    {state['input']}

    Respond thoughtfully.
    """

    state["response"] = llm(prompt)
    print("[DEBUG] Response generated")
    return state

# -----------------------------
# Node 3: Summarize Interaction
# -----------------------------
def summarize(state: GraphState):
    print("[MEMORY] Summarizing interaction")

    summary_prompt = f"""
    Summarize the key insight or preference from the interaction below.
    Ignore small talk.

    User: {state['input']}
    Assistant: {state['response']}
    """

    state["memory_summary"] = llm(summary_prompt)
    print("[DEBUG] Memory summary created")
    return state

# -----------------------------
# Node 4: Persist Long-Term Memory
# -----------------------------
def persist_memory(state: GraphState):
    print("[MEMORY] Persisting long-term memory")

    memory_collection.insert_one({
        "user_id": state["user_id"],
        "summary": state["memory_summary"],
        "timestamp": datetime.utcnow(),
        "schema_version": "v1"
    })

    print("[DEBUG] Memory persisted successfully")
    return state

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building Memory-Centric LangGraph")

graph = StateGraph(GraphState)

graph.add_node("load_memory", load_memory)
graph.add_node("reason", reason)
graph.add_node("summarize", summarize)
graph.add_node("persist_memory", persist_memory)

graph.set_entry_point("load_memory")

graph.add_edge("load_memory", "reason")
graph.add_edge("reason", "summarize")
graph.add_edge("summarize", "persist_memory")
graph.add_edge("persist_memory", END)

app = graph.compile()

print("[INIT] Graph compiled successfully")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Starting long-running agent")

    result = app.invoke({
        "user_id": "user_123",
        "input": "I prefer concise technical explanations with architecture diagrams."
    })

    print("\n[RESULT] Assistant Response:\n")
    print(result["response"])
