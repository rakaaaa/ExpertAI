"""
Failure-Aware Strategy (Production-Grade GenAI)
----------------------------------------------
Implements:
- Retry
- Fallback LLM
- Partial output
- Graceful degradation
- MongoDB audit logging
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient
from datetime import datetime
import random

# -----------------------------
# MongoDB (Audit Logs)
# -----------------------------
print("[INIT] Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph_memory"]
audit_logs = db["failure_audit"]
print("[INIT] MongoDB connected")

# -----------------------------
# LLMs
# -----------------------------
print("[INIT] Initializing LLMs...")
primary_llm = Ollama(model="mistral")
fallback_llm = Ollama(model="mistral")  # could be smaller model
print("[INIT] LLMs ready")

# -----------------------------
# Graph State
# -----------------------------
class GraphState(dict):
    pass

# -----------------------------
# Utility: Simulated Failure
# -----------------------------
def may_fail(probability=0.4):
    return random.random() < probability

# -----------------------------
# Node 1: Primary LLM
# -----------------------------
def primary_llm_node(state: GraphState):
    print("[LLM] Calling primary LLM")

    if may_fail():
        print("[ERROR] Primary LLM failed")
        raise RuntimeError("Primary LLM failure")

    state["response"] = primary_llm(state["prompt"])
    state["used_model"] = "primary"
    return state

# -----------------------------
# Node 2: Retry
# -----------------------------
def retry_node(state: GraphState):
    print("[RETRY] Retrying primary LLM once")

    try:
        state["response"] = primary_llm(state["prompt"])
        state["used_model"] = "primary_retry"
        return state
    except Exception:
        print("[RETRY] Retry failed")
        raise

# -----------------------------
# Node 3: Fallback LLM
# -----------------------------
def fallback_node(state: GraphState):
    print("[FALLBACK] Using fallback LLM")

    try:
        state["response"] = fallback_llm(state["prompt"])
        state["used_model"] = "fallback"
        return state
    except Exception:
        print("[FALLBACK] Fallback failed")
        raise

# -----------------------------
# Node 4: Partial Output
# -----------------------------
def partial_output(state: GraphState):
    print("[PARTIAL] Generating partial response")

    state["response"] = "Partial response: Unable to complete full request."
    state["used_model"] = "partial"
    return state

# -----------------------------
# Node 5: Safe Response
# -----------------------------
def safe_response(state: GraphState):
    print("[SAFE] Returning safe response")

    state["response"] = (
        "We're currently experiencing issues. "
        "Please try again later."
    )
    state["used_model"] = "safe"
    return state

# -----------------------------
# Node 6: Audit Log
# -----------------------------
def audit_log(state: GraphState):
    print("[AUDIT] Persisting execution details")

    audit_logs.insert_one({
        "prompt": state["prompt"],
        "response": state["response"],
        "model": state.get("used_model"),
        "timestamp": datetime.utcnow()
    })

    print("[AUDIT] Log persisted")
    return state

# -----------------------------
# Build Graph
# -----------------------------
print("[INIT] Building Failure-Aware LangGraph")

graph = StateGraph(GraphState)

graph.add_node("primary", primary_llm_node)
graph.add_node("retry", retry_node)
graph.add_node("fallback", fallback_node)
graph.add_node("partial", partial_output)
graph.add_node("safe", safe_response)
graph.add_node("audit", audit_log)

graph.set_entry_point("primary")

graph.add_edge("primary", "audit")

graph.add_conditional_edges(
    "primary",
    lambda _: "retry",
    {"retry": "retry"}
)

graph.add_conditional_edges(
    "retry",
    lambda _: "fallback",
    {"fallback": "fallback"}
)

graph.add_conditional_edges(
    "fallback",
    lambda _: "partial",
    {"partial": "partial"}
)

graph.add_edge("partial", "safe")
graph.add_edge("safe", "audit")
graph.add_edge("audit", END)

app = graph.compile()

print("[INIT] Graph compiled")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Executing failure-aware workflow")

    result = app.invoke({
        "prompt": "Generate a high-level migration plan to cloud."
    })

    print("\n[RESULT]")
    print(result["response"])
