"""
Template Agent Example

Demonstrates using an existing agent from the workspace by namespace/name@shortid.

Usage:
    INFERENCE_API_KEY=your_key AGENT=my-org/assistant@abc123 python examples/agent_template.py

Environment variables:
    INFERENCE_API_KEY - Your API key
    AGENT - Agent reference: namespace/name@shortid (e.g., "infsh/code-assistant@abc123")
"""

import os
import sys

from inferencesh import Agent, AgentConfig, TemplateAgentOptions


def main():
    api_key = os.environ.get("INFERENCE_API_KEY")
    agent_ref = os.environ.get("AGENT")
    
    if not api_key:
        print("Set INFERENCE_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)
    
    if not agent_ref:
        print("Set AGENT environment variable", file=sys.stderr)
        print('Format: namespace/name@shortid (e.g., "infsh/code-assistant@abc123")', file=sys.stderr)
        print("Get this from your agent in the workspace: https://app.inference.sh/agents", file=sys.stderr)
        sys.exit(1)

    # Create agent from template using namespace/name@shortid format
    config = AgentConfig(api_key=api_key)
    options = TemplateAgentOptions(agent=agent_ref)
    agent = Agent(config, options)
    
    print(f"Using template agent: {agent_ref}")
    print("Sending message...\n")
    
    def on_message(msg):
        content = msg.get("content", [])
        for c in content:
            if c.get("type") == "text" and c.get("text"):
                print(c["text"], end="", flush=True)
    
    def on_tool_call(call):
        print(f"\n[Tool Call] {call.name}: {call.args}")
        # For template agents, tool handlers depend on what tools are configured
        agent.submit_tool_result(call.id, '{"status": "not_implemented"}')
    
    # Send a message
    agent.send_message(
        "Hello! What can you help me with?",
        on_message=on_message,
        on_tool_call=on_tool_call,
    )
    
    print(f"\n\nChat ID: {agent.chat_id}")
    
    # Continue the conversation
    print("\n--- Follow-up ---\n")
    
    agent.send_message(
        "Tell me more about that.",
        on_message=on_message,
    )
    
    print("\n\nDone!")


if __name__ == "__main__":
    main()
