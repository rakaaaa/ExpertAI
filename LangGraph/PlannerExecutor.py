"""
Planner–Executor Strategy using LangGraph
-----------------------------------------
Pattern:
Planner → Task Executors → Aggregate → Output

Key Properties:
- Separation of planning and execution
- Task-level observability
- Easier retries
- MongoDB-backed memory
- Local LLM via Ollama
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient

# -----------------------------
# MongoDB: Task memory
# -----------------------------
print("[INIT] Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph_memory"]
collection = db["planner_executor_sessions"]
print("[INIT] MongoDB connected")

# -----------------------------
# Shared State
# -----------------------------
class GraphState(dict):
    """
    Shared state across planner and executors.
    Enables task-level observability.
    """
    pass

# -----------------------------
# Local LLM
# -----------------------------
print("[INIT] Initializing local LLM...")
llm = Ollama(model="mistral")
print("[INIT] LLM ready")

# -----------------------------
# Planner Node
# -----------------------------
def planner(state: GraphState):
    print("[PLANNER] Planning started")

    prompt = f"""
    Decompose the following task into steps:
    Task: {state['task']}

    Return a list of steps.
    """

    plan = llm(prompt)
    state["plan"] = [
        "Collect data",
        "Analyze data",
        "Generate report"
    ]  # deterministic plan for reliability

    print(f"[PLANNER] Plan created: {state['plan']}")
    return state

# -----------------------------
# Executor Nodes
# -----------------------------
def executor_1(state: GraphState):
    print("[EXECUTOR 1] Executing step: Collect data")
    state["data"] = "Sample dataset collected"
    return state

def executor_2(state: GraphState):
    print("[EXECUTOR 2] Executing step: Analyze data")
    state["analysis"] = "Key insights identified"
    return state

def executor_3(state: GraphState):
    print("[EXECUTOR 3] Executing step: Generate report")
    state["report"] = "Final report generated"
    return state

# -----------------------------
# Persist Results
# -----------------------------
def persist(state: GraphState):
    print("[NODE] Persisting planner-executor results")

    collection.insert_one({
        "task": state["task"],
        "plan": state["plan"],
        "data": state.get("data"),
        "analysis": state.get("analysis"),
        "report": state.get("report")
    })

    print("[DEBUG] Session persisted to MongoDB")
    return state

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building Planner–Executor LangGraph...")

graph = StateGraph(GraphState)

graph.add_node("planner", planner)
graph.add_node("executor_1", executor_1)
graph.add_node("executor_2", executor_2)
graph.add_node("executor_3", executor_3)
graph.add_node("persist", persist)

graph.set_entry_point("planner")

graph.add_edge("planner", "executor_1")
graph.add_edge("executor_1", "executor_2")
graph.add_edge("executor_2", "executor_3")
graph.add_edge("executor_3", "persist")
graph.add_edge("persist", END)

app = graph.compile()

print("[INIT] LangGraph compiled successfully")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Invoking Planner–Executor Strategy")

    result = app.invoke({
        "task": "Generate a business performance report"
    })

    print("\n[RESULT] Final Output:\n")
    print(result.get("report"))
