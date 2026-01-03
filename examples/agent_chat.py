#!/usr/bin/env python3
"""
Agent Chat Example

Demonstrates headless agent chat with client tools.

Usage:
    pip install inferencesh  # or: pip install -e .
    export INFERENCE_API_KEY=your-key
    python examples/agent_chat.py
"""

import sys
import json

from inferencesh import (
    inference,
    AgentRuntimeConfig,
    ToolCallInfo,
    is_message_ready,
    # Tool builders
    tool,
    internal_tools,
    string,
    number,
    enum_of,
    boolean,
)

# =============================================================================
# Define Client Tools
# =============================================================================

# Simple calculator tool
calculator_tool = (
    tool("calculator")
    .describe("Performs basic math operations")
    .param("a", number("First number"))
    .param("b", number("Second number"))
    .param("operation", enum_of(["add", "subtract", "multiply", "divide"], "Operation to perform"))
    .build()
)

# Weather lookup tool (simulated)
weather_tool = (
    tool("get_weather")
    .describe("Gets current weather for a location")
    .param("location", string("City name"))
    .param("units", enum_of(["celsius", "fahrenheit"], "Temperature units"))
    .build()
)

# File search tool (simulated)  
search_tool = (
    tool("search_files")
    .describe("Searches for files matching a pattern")
    .param("pattern", string("Search pattern (glob)"))
    .param("recursive", boolean("Search subdirectories"))
    .require_approval()  # Requires user approval (HIL)
    .build()
)


# =============================================================================
# Tool Handlers (executed client-side)
# =============================================================================

def handle_calculator(args: dict) -> str:
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    op = args.get("operation", "add")
    
    if op == "add":
        result = a + b
    elif op == "subtract":
        result = a - b
    elif op == "multiply":
        result = a * b
    elif op == "divide":
        result = a / b if b != 0 else float("nan")
    else:
        return json.dumps({"error": f"Unknown operation: {op}"})
    
    return json.dumps({
        "result": result,
        "expression": f"{a} {op} {b} = {result}"
    })


def handle_weather(args: dict) -> str:
    location = args.get("location", "Unknown")
    units = args.get("units", "celsius")
    
    # Simulated weather data
    temp = 22 if units == "celsius" else 72
    
    return json.dumps({
        "location": location,
        "temperature": temp,
        "units": units,
        "conditions": "Partly cloudy",
        "humidity": 65,
    })


def handle_search_files(args: dict) -> str:
    pattern = args.get("pattern", "*")
    recursive = args.get("recursive", False)
    
    # Simulated file search
    return json.dumps({
        "pattern": pattern,
        "recursive": recursive,
        "results": [
            {"path": "/home/user/docs/report.md", "size": 1024},
            {"path": "/home/user/docs/notes.txt", "size": 512},
        ],
    })


TOOL_HANDLERS = {
    "calculator": handle_calculator,
    "get_weather": handle_weather,
    "search_files": handle_search_files,
}


# =============================================================================
# Main
# =============================================================================

def main():
    api_key = "1nfsh-40d0xtgj90nd2tbtxjg2s96e1p"
    if not api_key:
        print("Set INFERENCE_API_KEY environment variable")
        sys.exit(1)
    
    # Create client and agent with ad-hoc config
    client = inference(api_key=api_key, base_url="https://api-dev.inference.sh")
    agent = client.agent(AgentRuntimeConfig(
        core_app_ref="infsh/claude-haiku-45@375bg07t",  # Replace with actual app reference
        name="Tool Assistant",
        system_prompt="""You are a helpful assistant with access to tools.
Available tools:
- calculator: Performs math operations
- get_weather: Gets weather for a location  
- search_files: Searches files (requires approval)

Use tools when appropriate to help the user.""",
        tools=[calculator_tool, weather_tool, search_tool],
        internal_tools=internal_tools().memory().build(),
    ))
    
    print("Agent ready. Sending message...\n")
    
    # Callback for message updates
    def on_message(msg):
        for c in msg.get("content", []):
            if c.get("type") == "text" and c.get("text"):
                print(c["text"])
    
    # Callback for tool calls
    def on_tool_call(call: ToolCallInfo):
        print(f"\n[Tool Call] {call.name}: {call.args}")
        
        handler = TOOL_HANDLERS.get(call.name)
        if handler:
            try:
                result = handler(call.args)
                print(f"[Tool Result] {result}")
                agent.submit_tool_result(call.id, result)
            except Exception as e:
                error = json.dumps({"error": str(e)})
                agent.submit_tool_result(call.id, error)
        else:
            agent.submit_tool_result(
                call.id, 
                json.dumps({"error": f"Unknown tool: {call.name}"})
            )
    
    # Send first message
    agent.send_message(
        "What is 42 * 17? Also, what's the weather in Paris?",
        on_message=on_message,
        on_tool_call=on_tool_call,
    )
    
    print(f"\n\nChat ID: {agent.chat_id}")
    
    # Continue conversation
    print("\n--- Second message ---\n")
    
    agent.send_message(
        "Now convert that temperature to Fahrenheit",
        on_message=on_message,
        on_tool_call=on_tool_call,
    )
    
    print("\n\nDone!")


def main_streaming():
    """Alternative example using manual streaming."""
    api_key = "1nfsh-40d0xtgj90nd2tbtxjg2s96e1p"
    if not api_key:
        print("Set INFERENCE_API_KEY environment variable")
        sys.exit(1)
    
    # Create client and agent
    client = inference(api_key=api_key, base_url="https://api-dev.inference.sh")
    agent = client.agent(AgentRuntimeConfig(
        core_app_ref="infsh/claude-haiku-45@375bg07t",
        name="Simple Assistant",
        system_prompt="You are a helpful assistant.",
    ))
    
    # Send message (no callbacks - we'll stream manually)
    agent.send_message("Tell me a short joke")
    
    print("Streaming messages...\n")
    
    # Manual streaming with auto-reconnect
    for message in agent.stream_messages(
        auto_reconnect=True,
        max_reconnects=3,
        reconnect_delay_ms=1000,
    ):
        content = message.get("content", [])
        for c in content:
            if c.get("type") == "text" and c.get("text"):
                print(c["text"], end="", flush=True)
        
        # Check if complete
        if is_message_ready(message.get("status")):
            break
    
    print("\n\nDone!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming", action="store_true", help="Use manual streaming example")
    args = parser.parse_args()
    
    if args.streaming:
        main_streaming()
    else:
        main()

