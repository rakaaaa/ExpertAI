"""
Observability-First Strategy (Enterprise GenAI)
----------------------------------------------
Tracks per-node:
- Latency
- Tokens (estimated)
- Tool usage
- Failures
- Cost signals

MongoDB used as observability backend
Local LLM for predictable cost
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient
from datetime import datetime
import time

# -----------------------------
# MongoDB (Observability Store)
# -----------------------------
print("[INIT] Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph_observability"]
metrics = db["node_metrics"]
print("[INIT] MongoDB connected")

# -----------------------------
# Local LLM
# -----------------------------
llm = Ollama(model="mistral")

# -----------------------------
# Graph State
# -----------------------------
class GraphState(dict):
    pass

# -----------------------------
# Utility: Metric Logger
# -----------------------------
def log_metrics(node, start, state, tokens=0, tool=None, error=None):
    duration = time.time() - start

    metrics.insert_one({
        "node": node,
        "latency_sec": round(duration, 3),
        "tokens_estimated": tokens,
        "tool": tool,
        "error": error,
        "timestamp": datetime.utcnow()
    })

    print(f"[METRICS] {node} | {round(duration,3)}s | tokens={tokens}")

# -----------------------------
# Node 1: Router (Cheap)
# -----------------------------
def router(state: GraphState):
    start = time.time()
    print("[ROUTER] Routing request")

    # Cheap heuristic routing
    state["route"] = "reason"
    log_metrics("router", start, state, tokens=10)
    return state

# -----------------------------
# Node 2: Planner
# -----------------------------
def planner(state: GraphState):
    start = time.time()
    print("[PLANNER] Planning tasks")

    state["plan"] = ["analyze", "respond"]
    log_metrics("planner", start, state, tokens=20)
    return state

# -----------------------------
# Node 3: Executor
# -----------------------------
def executor(state: GraphState):
    start = time.time()
    print("[EXECUTOR] Executing reasoning")

    prompt = f"""
    Perform reasoning for the task.
    Input: {state['input']}
    """

    response = llm(prompt)

    state["response"] = response
    log_metrics("executor", start, state, tokens=150)
    return state

# -----------------------------
# Node 4: Critic
# -----------------------------
def critic(state: GraphState):
    start = time.time()
    print("[CRITIC] Validating response")

    # Simple validation placeholder
    state["validated"] = True
    log_metrics("critic", start, state, tokens=30)
    return state

# -----------------------------
# Node 5: Final Response
# -----------------------------
def final_response(state: GraphState):
    start = time.time()
    print("[FINAL] Returning response")

    log_metrics("final", start, state, tokens=5)
    return state

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building Observability-First LangGraph")

graph = StateGraph(GraphState)

graph.add_node("router", router)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("critic", critic)
graph.add_node("final", final_response)

graph.set_entry_point("router")

graph.add_edge("router", "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "critic")
graph.add_edge("critic", "final")
graph.add_edge("final", END)

app = graph.compile()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Starting observability-first workflow")

    result = app.invoke({
        "input": "Explain why observability is critical for GenAI systems"
    })

    print("\n[RESULT]")
    print(result["response"])
