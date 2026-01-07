"""
Human-in-the-Loop (HITL) Strategy using LangGraph
------------------------------------------------
Pattern:
LLM Proposal → Human Review → Continue / Abort

Key Properties:
- Pause / Resume execution
- State persistence
- Full audit trail
- MongoDB-backed memory
- Local LLM (Ollama)
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient

# -----------------------------
# MongoDB: Audit + state store
# -----------------------------
print("[INIT] Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph_memory"]
collection = db["hitl_sessions"]
print("[INIT] MongoDB connected")

# -----------------------------
# Shared State
# -----------------------------
class GraphState(dict):
    """
    Explicit state for HITL workflows.
    Designed for audit and compliance.
    """
    pass

# -----------------------------
# Local LLM
# -----------------------------
print("[INIT] Initializing local LLM...")
llm = Ollama(model="mistral")
print("[INIT] LLM ready")

# -----------------------------
# Node 1: LLM Proposal
# -----------------------------
def llm_proposal(state: GraphState):
    print("[LLM] Generating proposal")

    prompt = f"""
    Generate a recommendation for the following request.
    Request: {state['request']}

    Keep it concise and explain rationale.
    """

    state["proposal"] = llm(prompt)
    state["status"] = "PENDING_REVIEW"

    print("[DEBUG] Proposal generated")
    return state

# -----------------------------
# Node 2: Persist for Review
# -----------------------------
def persist_for_review(state: GraphState):
    print("[NODE] Persisting proposal for human review")

    collection.insert_one({
        "request": state["request"],
        "proposal": state["proposal"],
        "status": state["status"]
    })

    print("[DEBUG] Proposal stored. Waiting for human approval.")
    return state

# -----------------------------
# Node 3: Human Decision
# -----------------------------
def human_decision(state: GraphState):
    """
    Simulates human approval.
    In real systems, this comes from UI / API / workflow tool.
    """
    print("[HUMAN] Awaiting human decision...")

    decision = input("Approve proposal? (yes/no): ").strip().lower()

    if decision == "yes":
        state["decision"] = "APPROVED"
    else:
        state["decision"] = "REJECTED"

    print(f"[HUMAN] Decision: {state['decision']}")
    return state

# -----------------------------
# Control Flow
# -----------------------------
def decision_router(state: GraphState):
    print("[CONTROL] Routing based on human decision")

    if state["decision"] == "APPROVED":
        return "continue"
    return "abort"

# -----------------------------
# Continue Path
# -----------------------------
def continue_execution(state: GraphState):
    print("[NODE] Continuing execution")

    state["final_output"] = f"APPROVED OUTPUT: {state['proposal']}"
    state["status"] = "APPROVED"

    return state

# -----------------------------
# Abort Path
# -----------------------------
def abort_execution(state: GraphState):
    print("[NODE] Aborting execution")

    state["final_output"] = "Request rejected by human reviewer"
    state["status"] = "REJECTED"

    return state

# -----------------------------
# Persist Final State
# -----------------------------
def persist_final(state: GraphState):
    print("[NODE] Persisting final decision")

    collection.update_one(
        {"request": state["request"]},
        {"$set": {
            "final_output": state["final_output"],
            "status": state["status"],
            "decision": state["decision"]
        }}
    )

    print("[DEBUG] Final state persisted")
    return state

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building HITL LangGraph...")

graph = StateGraph(GraphState)

graph.add_node("proposal", llm_proposal)
graph.add_node("persist_review", persist_for_review)
graph.add_node("human_review", human_decision)
graph.add_node("continue", continue_execution)
graph.add_node("abort", abort_execution)
graph.add_node("persist_final", persist_final)

graph.set_entry_point("proposal")

graph.add_edge("proposal", "persist_review")
graph.add_edge("persist_review", "human_review")

graph.add_conditional_edges(
    "human_review",
    decision_router,
    {
        "continue": "continue",
        "abort": "abort"
    }
)

graph.add_edge("continue", "persist_final")
graph.add_edge("abort", "persist_final")
graph.add_edge("persist_final", END)

app = graph.compile()

print("[INIT] LangGraph compiled successfully")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Starting HITL workflow")

    result = app.invoke({
        "request": "Approve a ₹5,00,000 expense for infrastructure upgrade"
    })

    print("\n[RESULT] Final Outcome:\n")
    print(result["final_output"])
