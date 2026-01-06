"""
Conditional Routing Strategy using LangGraph
--------------------------------------------
Pattern:
Input → Router → (Search | Calculator | LLM)

Key Properties:
- Deterministic routing
- Cost-optimized execution
- Logged decisions
- MongoDB-backed memory
- Local LLM via Ollama
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient
import re

# -----------------------------
# MongoDB: Memory store
# -----------------------------
print("[INIT] Connecting to MongoDB...")
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["langgraph_memory"]
collection = db["routing_sessions"]
print("[INIT] MongoDB connected")

# -----------------------------
# State definition
# -----------------------------
class GraphState(dict):
    """
    Shared state across routing paths.
    Enables auditability and debugging.
    """
    pass

# -----------------------------
# Local LLM
# -----------------------------
print("[INIT] Initializing local LLM...")
llm = Ollama(model="mistral")
print("[INIT] LLM ready")

# -----------------------------
# Router Node
# -----------------------------
def router(state: GraphState):
    print("[ROUTER] Routing decision started")

    question = state["question"].lower()

    if re.search(r"\d+[\+\-\*/]\d+", question):
        route = "calculator"
    elif "search" in question or "find" in question:
        route = "search"
    else:
        route = "llm"

    state["route"] = route
    print(f"[ROUTER] Route selected: {route}")

    return route

# -----------------------------
# Search Node (Mock)
# -----------------------------
def search_tool(state: GraphState):
    print("[NODE] Search tool invoked")

    state["answer"] = "Search result: LangGraph is a stateful agent orchestration framework."
    print("[DEBUG] Search response generated")

    return state

# -----------------------------
# Calculator Node
# -----------------------------
def calculator(state: GraphState):
    print("[NODE] Calculator invoked")

    expression = re.findall(r"\d+[\+\-\*/]\d+", state["question"])[0]
    result = eval(expression)

    state["answer"] = f"Calculation result: {result}"
    print(f"[DEBUG] Calculated result: {result}")

    return state

# -----------------------------
# LLM Node
# -----------------------------
def llm_reasoner(state: GraphState):
    print("[NODE] LLM reasoner invoked")

    state["answer"] = llm(state["question"])
    print("[DEBUG] LLM response generated")

    return state

# -----------------------------
# Persist Memory
# -----------------------------
def persist(state: GraphState):
    print("[NODE] Persisting routing decision and answer")

    collection.insert_one({
        "question": state["question"],
        "route": state["route"],
        "answer": state["answer"]
    })

    print("[DEBUG] Memory persisted to MongoDB")
    return state

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building LangGraph with conditional routing...")

graph = StateGraph(GraphState)

graph.add_node("search", search_tool)
graph.add_node("calculator", calculator)
graph.add_node("llm", llm_reasoner)
graph.add_node("persist", persist)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    router,
    {
        "search": "search",
        "calculator": "calculator",
        "llm": "llm"
    }
)

graph.add_edge("search", "persist")
graph.add_edge("calculator", "persist")
graph.add_edge("llm", "persist")
graph.add_edge("persist", END)

app = graph.compile()

print("[INIT] LangGraph compiled successfully")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Invoking Conditional Routing Strategy")

    result = app.invoke({
        "question": "What is 12+30?"
    })

    print("\n[RESULT] Final Answer:\n")
    print(result["answer"])
