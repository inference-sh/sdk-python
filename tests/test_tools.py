"""Tests for the tool builder fluent API."""

import pytest

from inferencesh.tools import (
    tool,
    app_tool,
    agent_tool,
    webhook_tool,
    internal_tools,
    string,
    number,
    integer,
    boolean,
    enum_of,
    obj,
    array,
    optional,
)


class TestSchemaHelpers:
    """Tests for schema helper functions."""

    def test_string_without_description(self):
        schema = string()
        assert schema == {"type": "string"}

    def test_string_with_description(self):
        schema = string("User name")
        assert schema == {"type": "string", "description": "User name"}

    def test_number(self):
        assert number() == {"type": "number"}
        assert number("Temperature") == {"type": "number", "description": "Temperature"}

    def test_integer(self):
        assert integer() == {"type": "integer"}
        assert integer("Age") == {"type": "integer", "description": "Age"}

    def test_boolean(self):
        assert boolean() == {"type": "boolean"}
        assert boolean("Is active") == {"type": "boolean", "description": "Is active"}

    def test_enum_of(self):
        schema = enum_of(["low", "medium", "high"])
        assert schema == {"type": "string", "enum": ["low", "medium", "high"]}

    def test_enum_of_with_description(self):
        schema = enum_of(["a", "b"], "Priority level")
        assert schema == {"type": "string", "enum": ["a", "b"], "description": "Priority level"}

    def test_obj(self):
        schema = obj({
            "name": string("Name"),
            "age": integer("Age"),
        })
        assert schema == {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name"},
                "age": {"type": "integer", "description": "Age"},
            },
        }

    def test_obj_with_description(self):
        schema = obj({"x": number()}, "Coordinates")
        assert schema["description"] == "Coordinates"

    def test_array(self):
        schema = array(string("Tag"))
        assert schema == {
            "type": "array",
            "items": {"type": "string", "description": "Tag"},
        }

    def test_array_with_description(self):
        schema = array(integer(), "List of IDs")
        assert schema == {
            "type": "array",
            "items": {"type": "integer"},
            "description": "List of IDs",
        }

    def test_optional(self):
        schema = optional(string("Optional field"))
        assert schema == {"type": "string", "description": "Optional field", "optional": True}

    def test_optional_preserves_properties(self):
        schema = optional(enum_of(["a", "b"], "Choice"))
        assert schema["enum"] == ["a", "b"]
        assert schema["optional"] is True


class TestClientToolBuilder:
    """Tests for client tool builder."""

    def test_minimal_tool(self):
        t = tool("my_tool").build()
        assert t["name"] == "my_tool"
        assert t["display_name"] == "my_tool"
        assert t["description"] == ""
        assert t["type"] == "client"
        assert t["client"]["input_schema"] == {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def test_tool_with_description(self):
        t = tool("greet").describe("Says hello").build()
        assert t["description"] == "Says hello"

    def test_tool_with_display_name(self):
        t = tool("get_data").display("Get Data").build()
        assert t["display_name"] == "Get Data"

    def test_tool_with_parameters(self):
        t = (
            tool("add")
            .param("a", number("First number"))
            .param("b", number("Second number"))
            .build()
        )

        assert t["client"]["input_schema"] == {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        }

    def test_optional_parameters(self):
        t = (
            tool("search")
            .param("query", string("Search query"))
            .param("limit", optional(integer("Max results")))
            .build()
        )

        assert t["client"]["input_schema"]["required"] == ["query"]
        assert "limit" in t["client"]["input_schema"]["properties"]

    def test_tool_with_require_approval(self):
        t = tool("dangerous").require_approval().build()
        assert t["require_approval"] is True

    def test_nested_object_parameters(self):
        t = (
            tool("create_user")
            .param("user", obj({
                "name": string("Name"),
                "email": string("Email"),
            }, "User data"))
            .build()
        )

        schema = t["client"]["input_schema"]
        assert schema["properties"]["user"]["type"] == "object"
        assert "name" in schema["properties"]["user"]["properties"]
        assert "email" in schema["properties"]["user"]["properties"]

    def test_array_parameters(self):
        t = (
            tool("process_items")
            .param("items", array(string("Item"), "List of items"))
            .build()
        )

        assert t["client"]["input_schema"]["properties"]["items"]["type"] == "array"
        assert t["client"]["input_schema"]["properties"]["items"]["items"]["type"] == "string"


class TestAppToolBuilder:
    """Tests for app tool builder."""

    def test_creates_app_tool(self):
        t = (
            app_tool("generate", "infsh/flux@v1.0")
            .describe("Generate image")
            .build()
        )

        assert t["type"] == "app"
        assert t["app"] == {"id": "infsh/flux", "version_id": "v1.0"}
        assert t["description"] == "Generate image"

    def test_app_tool_with_latest_version(self):
        t = app_tool("browse", "my-org/browser@latest").build()
        assert t["app"] == {"id": "my-org/browser", "version_id": "latest"}

    def test_app_tool_with_parameters(self):
        t = (
            app_tool("fetch", "infsh/fetch@v1")
            .param("url", string("URL to fetch"))
            .build()
        )

        assert t["name"] == "fetch"


class TestAgentToolBuilder:
    """Tests for agent tool builder."""

    def test_creates_agent_tool(self):
        t = (
            agent_tool("research", "acme/researcher@v2")
            .describe("Research a topic")
            .build()
        )

        assert t["type"] == "agent"
        assert t["agent"] == {"id": "acme/researcher", "version_id": "v2"}

    def test_agent_tool_with_display_and_approval(self):
        t = (
            agent_tool("coder", "infsh/code-agent@latest")
            .display("Code Assistant")
            .require_approval()
            .build()
        )

        assert t["display_name"] == "Code Assistant"
        assert t["require_approval"] is True


class TestWebhookToolBuilder:
    """Tests for webhook tool builder."""

    def test_creates_webhook_tool(self):
        t = (
            webhook_tool("notify", "https://api.example.com/webhook")
            .describe("Send notification")
            .build()
        )

        assert t["type"] == "hook"
        assert t["hook"]["url"] == "https://api.example.com/webhook"
        assert t["description"] == "Send notification"

    def test_webhook_with_secret(self):
        t = (
            webhook_tool("slack", "https://hooks.slack.com/services/xxx")
            .secret("SLACK_SECRET")
            .build()
        )

        assert t["hook"]["secret"] == "SLACK_SECRET"

    def test_webhook_generates_input_schema(self):
        t = (
            webhook_tool("send", "https://api.example.com")
            .param("message", string("Message"))
            .param("priority", optional(enum_of(["low", "high"])))
            .build()
        )

        assert t["hook"]["input_schema"] == {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message"},
                "priority": {"type": "string", "enum": ["low", "high"]},
            },
            "required": ["message"],
        }


class TestInternalToolsBuilder:
    """Tests for internal tools builder."""

    def test_empty_config_by_default(self):
        config = internal_tools().build()
        assert config == {}

    def test_enable_plan(self):
        config = internal_tools().plan().build()
        assert config == {"plan": True}

    def test_enable_memory(self):
        config = internal_tools().memory().build()
        assert config == {"memory": True}

    def test_enable_widget(self):
        config = internal_tools().widget().build()
        assert config == {"widget": True}

    def test_enable_finish(self):
        config = internal_tools().finish().build()
        assert config == {"finish": True}

    def test_chain_multiple_enables(self):
        config = internal_tools().plan().memory().widget().build()
        assert config == {"plan": True, "memory": True, "widget": True}

    def test_all_enables_everything(self):
        config = internal_tools().all().build()
        assert config == {"plan": True, "memory": True, "widget": True, "finish": True}

    def test_none_disables_everything(self):
        config = internal_tools().none().build()
        assert config == {"plan": False, "memory": False, "widget": False, "finish": False}

    def test_explicit_disable(self):
        config = internal_tools().plan(False).memory(True).build()
        assert config == {"plan": False, "memory": True}


class TestFluentAPIChaining:
    """Tests for full fluent API chains."""

    def test_full_fluent_chain(self):
        t = (
            tool("complex")
            .display("Complex Tool")
            .describe("A complex tool with many params")
            .param("name", string("Name"))
            .param("count", integer("Count"))
            .param("options", optional(obj({
                "verbose": boolean("Verbose mode"),
                "tags": array(string("Tag")),
            })))
            .require_approval()
            .build()
        )

        assert t["name"] == "complex"
        assert t["display_name"] == "Complex Tool"
        assert t["description"] == "A complex tool with many params"
        assert t["require_approval"] is True
        assert t["client"]["input_schema"]["required"] == ["name", "count"]

