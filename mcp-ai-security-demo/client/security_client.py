import subprocess
import json
import sys
from pathlib import Path
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

# Resolve server path relative to this file to avoid cwd issues
SERVER_PATH = Path(__file__).resolve().parent.parent / "server" / "security_server.py"

server = subprocess.Popen(
    [sys.executable, "-u", str(SERVER_PATH)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

def mcp_request(payload):
    server.stdin.write(json.dumps(payload) + "\n")
    server.stdin.flush()
    return json.loads(server.stdout.readline())

# Discover tools
tools = mcp_request({"method": "list_tools"})
print("Available tools:", [t["name"] for t in tools["tools"]])

# Ask LLM to reason
prompt = """
You are a SOC analyst.
Investigate suspicious outbound connections.
Look for command-and-control traffic.
"""

try:
    llm_payload = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=10,
    ).json()
    llm_response = llm_payload.get("response", llm_payload)
except Exception as exc:  # fallback if Ollama unavailable
    llm_response = f"(LLM call failed: {exc})"

print("\nLLM Reasoning:\n", llm_response)

# Execute MCP tools
logs = mcp_request({
    "method": "call_tool",
    "params": {"name": "read_security_logs"}
})

suspicious = mcp_request({
    "method": "call_tool",
    "params": {
        "name": "find_suspicious_connections",
        "arguments": {"port": "4444"}
    }
})

print("\nSecurity Logs:")
print("\n".join(logs["result"]))

print("\nSuspicious Findings:")
print("\n".join(suspicious["result"]))
