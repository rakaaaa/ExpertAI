import sys
import json
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "workspace" / "logs.txt"

TOOLS = [
    {
        "name": "read_security_logs",
        "description": "Read all security logs",
        "input_schema": {}
    },
    {
        "name": "find_suspicious_connections",
        "description": "Find connections matching a suspicious port",
        "input_schema": {
            "type": "object",
            "properties": {
                "port": {"type": "string"}
            },
            "required": ["port"]
        }
    }
]

def read_logs():
    with open(LOG_FILE, encoding="utf-8") as f:
        return f.read().splitlines()

def find_suspicious(port):
    with open(LOG_FILE, encoding="utf-8") as f:
        return [line for line in f if f":{port}" in line]

def handle_request(req):
    if req["method"] == "list_tools":
        return {"tools": TOOLS}

    if req["method"] == "call_tool":
        name = req["params"]["name"]
        args = req["params"].get("arguments", {})

        if name == "read_security_logs":
            return {"result": read_logs()}

        if name == "find_suspicious_connections":
            return {"result": find_suspicious(args["port"])}

        return {"error": "Unknown tool"}

# MCP stdio loop
for line in sys.stdin:
    req = json.loads(line)
    resp = handle_request(req)
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
