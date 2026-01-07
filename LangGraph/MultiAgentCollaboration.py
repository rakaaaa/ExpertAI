"""
Multi-Agent Collaboration Strategy using LangGraph
--------------------------------------------------
Pattern:
User → Coordinator → (Research | Critic | Writer) → Final Output

Key Properties:
- Clear agent roles
- Shared but scoped memory
- Single final authority
- MongoDB-backed memory
- Local LLM (Ollama)
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient

# -----------------------------
# MongoDB: Shared memory store
# -----------------------------
print("[INIT] Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph_memory"]
collection = db["multi_agent_sessions"]
print("[INIT] MongoDB connected")

# -----------------------------
# Shared State
# -----------------------------
class GraphState(dict):
    """
    Shared state across all agents.
    Agent outputs are scoped to avoid contamination.
    """
    pass

# -----------------------------
# Local LLM
# -----------------------------
print("[INIT] Initializing local LLM...")
llm = Ollama(model="mistral")
print("[INIT] LLM ready")

# -----------------------------
# Coordinator
# -----------------------------
def coordinator(state: GraphState):
    print("[COORDINATOR] Task received")
    print(f"[DEBUG] Topic: {state['topic']}")
    return state

# -----------------------------
# Research Agent
# -----------------------------
def research_agent(state: GraphState):
    print("[RESEARCH] Research agent started")

    prompt = f"""
    Research the following topic and provide key facts:
    Topic: {state['topic']}
    """

    state["research_notes"] = llm(prompt)
    print("[DEBUG] Research notes generated")
    return state

# -----------------------------
# Critic Agent
# -----------------------------
def critic_agent(state: GraphState):
    print("[CRITIC] Critic agent started")

    prompt = f"""
    Review the following research and identify gaps or risks:
    {state['research_notes']}
    """

    state["critic_notes"] = llm(prompt)
    print("[DEBUG] Critic feedback generated")
    return state

# -----------------------------
# Writer Agent (Final Authority)
# -----------------------------
def writer_agent(state: GraphState):
    print("[WRITER] Writer agent started (final authority)")

    prompt = f"""
    Using the research and critique below, write a final response.

    Research:
    {state['research_notes']}

    Critique:
    {state['critic_notes']}
    """

    state["final_output"] = llm(prompt)
    print("[DEBUG] Final output generated")
    return state

# -----------------------------
# Persist Session
# -----------------------------
def persist(state: GraphState):
    print("[NODE] Persisting multi-agent session")

    collection.insert_one({
        "topic": state["topic"],
        "research": state["research_notes"],
        "critique": state["critic_notes"],
        "final_output": state["final_output"]
    })

    print("[DEBUG] Session saved to MongoDB")
    return state

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building Multi-Agent LangGraph...")

graph = StateGraph(GraphState)

graph.add_node("coordinator", coordinator)
graph.add_node("research", research_agent)
graph.add_node("critic", critic_agent)
graph.add_node("writer", writer_agent)
graph.add_node("persist", persist)

graph.set_entry_point("coordinator")

graph.add_edge("coordinator", "research")
graph.add_edge("research", "critic")
graph.add_edge("critic", "writer")
graph.add_edge("writer", "persist")
graph.add_edge("persist", END)

app = graph.compile()

print("[INIT] LangGraph compiled successfully")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Starting Multi-Agent Collaboration")

    result = app.invoke({
        "topic": "Explain LangGraph for enterprise GenAI systems"
    })

    print("\n[RESULT] Final Output:\n")
    print(result["final_output"])
