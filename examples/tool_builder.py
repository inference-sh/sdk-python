#!/usr/bin/env python3
"""
Tool Builder Example

Demonstrates the fluent API for building agent tools.
No API key required - just shows the tool schemas.

Usage:
    pip install inferencesh  # or: pip install -e .
    python examples/tool_builder.py
"""

import json

from inferencesh import (
    tool,
    app_tool,
    agent_tool,
    webhook_tool,
    internal_tools,
    string,
    integer,
    boolean,
    enum_of,
    obj,
    array,
    optional,
)

# =============================================================================
# Client Tools (executed in SDK consumer's environment)
# =============================================================================

# Simple tool with typed parameters
scan_ui = (
    tool("scan_ui")
    .describe("Scans the UI and returns an accessibility tree")
    .display("Scan UI")
    .build()
)

print("scan_ui tool:")
print(json.dumps(scan_ui, indent=2, default=str))

# Tool with multiple parameters
fill_field = (
    tool("fill_field")
    .describe("Fills a form field by its name or label")
    .param("field", string("The field name, label, or ID"))
    .param("value", string("The value to fill"))
    .build()
)

print("\nfill_field tool:")
print(json.dumps(fill_field, indent=2, default=str))

# Tool with enum parameter
interact = (
    tool("interact")
    .describe("Performs a UI interaction")
    .param("selector", string("CSS selector for the element"))
    .param("action", enum_of(["click", "type", "select", "focus", "blur"], "Action to perform"))
    .param("text", optional(string("Text to type (for type action)")))
    .build()
)

print("\ninteract tool:")
print(json.dumps(interact, indent=2, default=str))

# Tool with complex nested parameters
create_task = (
    tool("create_task")
    .describe("Creates a new task with metadata")
    .param("title", string("Task title"))
    .param("priority", enum_of(["low", "medium", "high"], "Priority level"))
    .param("tags", array(string("Tag name"), "List of tags"))
    .param("metadata", optional(obj({
        "assignee": optional(string("Assignee email")),
        "due_date": optional(string("Due date (ISO format)")),
        "estimate": optional(integer("Time estimate in hours")),
    }, "Additional metadata")))
    .require_approval()  # Human-in-the-loop
    .build()
)

print("\ncreate_task tool (with HIL):")
print(json.dumps(create_task, indent=2, default=str))

# =============================================================================
# Server Tools (executed on inference.sh servers)
# =============================================================================

# App tool - calls another inference app
generate_image = (
    app_tool("generate_image", "infsh/flux-schnell@abc123")
    .describe("Generates an image from a text prompt")
    .param("prompt", string("Image description"))
    .param("width", optional(integer("Image width")))
    .param("height", optional(integer("Image height")))
    .require_approval()  # Costs credits
    .build()
)

print("\ngenerate_image (app tool):")
print(json.dumps(generate_image, indent=2, default=str))

# Agent tool - delegates to sub-agent
code_review = (
    agent_tool("code_review", "infsh/code-reviewer@xyz789")
    .describe("Reviews code for best practices and bugs")
    .param("code", string("Code to review"))
    .param("language", enum_of(["typescript", "python", "go", "rust"], "Programming language"))
    .build()
)

print("\ncode_review (agent tool):")
print(json.dumps(code_review, indent=2, default=str))

# Webhook tool - calls external URL
send_slack = (
    webhook_tool("send_slack", "https://hooks.slack.com/services/...")
    .describe("Sends a message to Slack")
    .secret("SLACK_WEBHOOK_SECRET")
    .param("channel", string("Channel name"))
    .param("message", string("Message text"))
    .param("urgent", optional(boolean("Mark as urgent")))
    .build()
)

print("\nsend_slack (webhook tool):")
print(json.dumps(send_slack, indent=2, default=str))

# =============================================================================
# Internal Tools Configuration
# =============================================================================

# Using fluent builder
internal_config = (
    internal_tools()
    .plan()     # Enable plan tools
    .memory()   # Enable memory tools
    .build()
)

print("\nInternal tools config:")
print(json.dumps(internal_config, indent=2))

# Or enable all
all_internal = internal_tools().all().build()
print("All internal tools:", json.dumps(all_internal, indent=2))

# Or disable all
no_internal = internal_tools().none().build()
print("No internal tools:", json.dumps(no_internal, indent=2))

# =============================================================================
# Full Agent Config Example
# =============================================================================

# Example of what you'd pass to Agent()
agent_config = {
    "core_app": "infsh/claude-sonnet-4@latest",
    "name": "Form Assistant",
    "system_prompt": "You are a helpful assistant that can interact with forms.",
    "tools": [scan_ui, fill_field, interact, generate_image],
    "internal_tools": internal_tools().memory().build(),
}

print("\n=== Full Agent Config ===")
print(json.dumps(agent_config, indent=2, default=str))

