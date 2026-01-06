"""
ReAct Loop Strategy using LangGraph
----------------------------------
Pattern:
Reason → Act → Observe → Reason → ...

Guardrails:
- Max iterations
- Tool whitelist
- Explicit termination
- MongoDB-backed memory
- Local LLM (Ollama)
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient
import time

# -----------------------------
# MongoDB: Long-term memory
# -----------------------------
print("[INIT] Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph_memory"]
collection = db["react_sessions"]
print("[INIT] MongoDB connected")

# -----------------------------
# Shared State
# -----------------------------
class GraphState(dict):
    """
    Central state for ReAct loop.
    Explicit state = debuggable autonomy.
    """
    pass

# -----------------------------
# Local LLM
# -----------------------------
print("[INIT] Initializing local LLM...")
llm = Ollama(model="mistral")
print("[INIT] LLM ready")

MAX_ITERATIONS = 3
ALLOWED_TOOLS = {"search", "finish"}

# -----------------------------
# Reason Node
# -----------------------------
def reason(state: GraphState):
    print(f"[REASON] Iteration {state['iteration']}")

    prompt = f"""
    Question:
    {state['question']}

    Observations so far:
    {state.get('observations', [])}

    Decide next action.
    Reply ONLY in this format:
    ACTION: search | finish
    """

    response = llm(prompt).lower()
    print(f"[DEBUG] LLM reasoning output: {response}")

    if "search" in response:
        state["action"] = "search"
    else:
        state["action"] = "finish"

    return state

# -----------------------------
# Act Node (Tool)
# -----------------------------
def act(state: GraphState):
    print(f"[ACT] Action selected: {state['action']}")

    if state["action"] not in ALLOWED_TOOLS:
        raise ValueError("Disallowed tool invoked")

    if state["action"] == "search":
        # Mock tool call
        observation = "LangGraph enables stateful agent workflows."
    else:
        observation = "No further action required."

    state.setdefault("observations", []).append(observation)
    print(f"[OBSERVE] Observation: {observation}")

    return state

# -----------------------------
# Loop Control
# -----------------------------
def should_continue(state: GraphState):
    print("[CONTROL] Checking loop conditions")

    if state["action"] == "finish":
        print("[CONTROL] Finish action detected")
        return "end"

    if state["iteration"] >= MAX_ITERATIONS:
        print("[CONTROL] Max iterations reached")
        return "end"

    state["iteration"] += 1
    return "continue"

# -----------------------------
# Persist Memory
# -----------------------------
def persist(state: GraphState):
    print("[NODE] Persisting ReAct session")

    collection.insert_one({
        "question": state["question"],
        "observations": state.get("observations", []),
        "iterations": state["iteration"]
    })

    print("[DEBUG] ReAct session stored")
    return state

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building ReAct LangGraph...")

graph = StateGraph(GraphState)

graph.add_node("reason", reason)
graph.add_node("act", act)
graph.add_node("persist", persist)

graph.set_entry_point("reason")
graph.add_edge("reason", "act")

graph.add_conditional_edges(
    "act",
    should_continue,
    {
        "continue": "reason",
        "end": "persist"
    }
)

graph.add_edge("persist", END)

app = graph.compile()

print("[INIT] LangGraph compiled")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Starting ReAct loop")

    result = app.invoke({
        "question": "Explain LangGraph briefly",
        "iteration": 1
    })

    print("\n[RESULT] Observations:\n")
    print(result.get("observations", []))
