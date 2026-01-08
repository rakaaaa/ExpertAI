"""
Cost-Optimized Strategy (Production GenAI)
-----------------------------------------
Techniques:
- Cheap model for routing
- Expensive model only for reasoning
- MongoDB-based cache
- Short-circuit execution
"""

from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
from pymongo import MongoClient
from datetime import datetime

# -----------------------------
# MongoDB Cache Setup
# -----------------------------
print("[INIT] Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph_memory"]
cache_collection = db["response_cache"]
print("[INIT] MongoDB connected")

# -----------------------------
# LLMs
# -----------------------------
print("[INIT] Initializing LLMs")

# Cheap / routing model
router_llm = Ollama(model="mistral")

# Expensive / reasoning model (simulated as same local model)
reasoning_llm = Ollama(model="mistral")

print("[INIT] LLMs ready")

# -----------------------------
# Graph State
# -----------------------------
class GraphState(dict):
    pass

# -----------------------------
# Node 1: Cache Lookup
# -----------------------------
def cache_lookup(state: GraphState):
    print("[CACHE] Checking cache")

    cached = cache_collection.find_one({"query": state["query"]})

    if cached:
        print("[CACHE] Cache hit → short-circuit")
        state["response"] = cached["response"]
        state["source"] = "cache"
        state["exit"] = True
    else:
        print("[CACHE] Cache miss")
        state["exit"] = False

    return state

# -----------------------------
# Node 2: Cheap Router
# -----------------------------
def router(state: GraphState):
    print("[ROUTER] Deciding if reasoning is needed")

    prompt = f"""
    Decide if the following query requires deep reasoning.
    Respond with ONLY 'YES' or 'NO'.

    Query: {state['query']}
    """

    decision = router_llm(prompt).strip().upper()

    state["needs_reasoning"] = decision == "YES"

    print(f"[ROUTER] Needs reasoning: {state['needs_reasoning']}")
    return state

# -----------------------------
# Node 3: Simple Response
# -----------------------------
def simple_response(state: GraphState):
    print("[SIMPLE] Generating simple response")

    state["response"] = (
        "This is a straightforward request. "
        "Here is a concise answer without deep reasoning."
    )
    state["source"] = "simple"
    return state

# -----------------------------
# Node 4: Expensive Reasoning
# -----------------------------
def reasoning(state: GraphState):
    print("[REASONING] Invoking expensive reasoning model")

    prompt = f"""
    Perform deep reasoning and provide a detailed answer.

    Query: {state['query']}
    """

    state["response"] = reasoning_llm(prompt)
    state["source"] = "reasoning"
    return state

# -----------------------------
# Node 5: Cache Store
# -----------------------------
def cache_store(state: GraphState):
    print("[CACHE] Storing response")

    cache_collection.insert_one({
        "query": state["query"],
        "response": state["response"],
        "source": state["source"],
        "timestamp": datetime.utcnow()
    })

    print("[CACHE] Stored successfully")
    return state

# -----------------------------
# Control Logic
# -----------------------------
def cache_router(state: GraphState):
    if state["exit"]:
        return "end"
    return "router"

def reasoning_router(state: GraphState):
    if state["needs_reasoning"]:
        return "reasoning"
    return "simple"

# -----------------------------
# Build LangGraph
# -----------------------------
print("[INIT] Building Cost-Optimized LangGraph")

graph = StateGraph(GraphState)

graph.add_node("cache_lookup", cache_lookup)
graph.add_node("router", router)
graph.add_node("simple", simple_response)
graph.add_node("reasoning", reasoning)
graph.add_node("cache_store", cache_store)

graph.set_entry_point("cache_lookup")

graph.add_conditional_edges(
    "cache_lookup",
    cache_router,
    {
        "router": "router",
        "end": END
    }
)

graph.add_conditional_edges(
    "router",
    reasoning_router,
    {
        "simple": "simple",
        "reasoning": "reasoning"
    }
)

graph.add_edge("simple", "cache_store")
graph.add_edge("reasoning", "cache_store")
graph.add_edge("cache_store", END)

app = graph.compile()

print("[INIT] Graph compiled successfully")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("[RUN] Executing cost-optimized workflow")

    result = app.invoke({
        "query": "Explain microservices vs monolith architecture"
    })

    print("\n[RESULT]")
    print(f"Source: {result['source']}")
    print(result["response"])
