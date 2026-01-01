"""
Tool Builder - Fluent API for defining agent tools
"""

from typing import Any, Dict, List, Optional
from .types import AgentTool, InternalToolsConfig, ToolType


# =============================================================================
# Schema Builders
# =============================================================================

def string(description: Optional[str] = None) -> Dict[str, Any]:
    """String parameter."""
    schema: Dict[str, Any] = {"type": "string"}
    if description:
        schema["description"] = description
    return schema


def number(description: Optional[str] = None) -> Dict[str, Any]:
    """Number parameter."""
    schema: Dict[str, Any] = {"type": "number"}
    if description:
        schema["description"] = description
    return schema


def integer(description: Optional[str] = None) -> Dict[str, Any]:
    """Integer parameter."""
    schema: Dict[str, Any] = {"type": "integer"}
    if description:
        schema["description"] = description
    return schema


def boolean(description: Optional[str] = None) -> Dict[str, Any]:
    """Boolean parameter."""
    schema: Dict[str, Any] = {"type": "boolean"}
    if description:
        schema["description"] = description
    return schema


def enum_of(values: List[str], description: Optional[str] = None) -> Dict[str, Any]:
    """String enum parameter."""
    schema: Dict[str, Any] = {"type": "string", "enum": values}
    if description:
        schema["description"] = description
    return schema


def array(items: Dict[str, Any], description: Optional[str] = None) -> Dict[str, Any]:
    """Array parameter."""
    schema: Dict[str, Any] = {"type": "array", "items": items}
    if description:
        schema["description"] = description
    return schema


def obj(properties: Dict[str, Dict[str, Any]], description: Optional[str] = None) -> Dict[str, Any]:
    """Object parameter."""
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if description:
        schema["description"] = description
    return schema


def optional(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Mark schema as optional."""
    return {**schema, "optional": True}


# =============================================================================
# JSON Schema Generator
# =============================================================================

def _to_json_schema(params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert params dict to JSON Schema."""
    properties: Dict[str, Any] = {}
    required: List[str] = []
    
    for key, schema in params.items():
        prop = {k: v for k, v in schema.items() if k != "optional"}
        properties[key] = prop
        if not schema.get("optional"):
            required.append(key)
    
    return {"type": "object", "properties": properties, "required": required}


# =============================================================================
# Tool Builders
# =============================================================================

class _ToolBuilder:
    """Base tool builder."""
    
    def __init__(self, name: str):
        self._name = name
        self._description = ""
        self._display_name: Optional[str] = None
        self._params: Dict[str, Dict[str, Any]] = {}
        self._require_approval = False
    
    def describe(self, description: str) -> "_ToolBuilder":
        """Set description."""
        self._description = description
        return self
    
    def display(self, name: str) -> "_ToolBuilder":
        """Set display name."""
        self._display_name = name
        return self
    
    def param(self, name: str, schema: Dict[str, Any]) -> "_ToolBuilder":
        """Add a parameter."""
        self._params[name] = schema
        return self
    
    def require_approval(self) -> "_ToolBuilder":
        """Require human approval (HIL)."""
        self._require_approval = True
        return self


class ClientToolBuilder(_ToolBuilder):
    """Builder for client tools."""
    
    def build(self) -> AgentTool:
        return {
            "name": self._name,
            "display_name": self._display_name or self._name,
            "description": self._description,
            "type": ToolType.CLIENT,
            "require_approval": self._require_approval or None,
            "client": {"input_schema": _to_json_schema(self._params)},
        }


class AppToolBuilder(_ToolBuilder):
    """Builder for app tools."""
    
    def __init__(self, name: str, app_ref: str):
        super().__init__(name)
        self._app_ref = app_ref
    
    def build(self) -> AgentTool:
        parts = self._app_ref.split("/")
        ns = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        app_parts = rest.split("@")
        app_name = app_parts[0]
        version = app_parts[1] if len(app_parts) > 1 else None
        
        return {
            "name": self._name,
            "display_name": self._display_name or self._name,
            "description": self._description,
            "type": ToolType.APP,
            "require_approval": self._require_approval or None,
            "app": {"id": f"{ns}/{app_name}", "version_id": version},
        }


class AgentToolBuilder(_ToolBuilder):
    """Builder for agent tools (sub-agents)."""
    
    def __init__(self, name: str, agent_ref: str):
        super().__init__(name)
        self._agent_ref = agent_ref
    
    def build(self) -> AgentTool:
        parts = self._agent_ref.split("/")
        ns = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        agent_parts = rest.split("@")
        agent_name = agent_parts[0]
        version = agent_parts[1] if len(agent_parts) > 1 else None
        
        return {
            "name": self._name,
            "display_name": self._display_name or self._name,
            "description": self._description,
            "type": ToolType.AGENT,
            "require_approval": self._require_approval or None,
            "agent": {"id": f"{ns}/{agent_name}", "version_id": version},
        }


class WebhookToolBuilder(_ToolBuilder):
    """Builder for webhook tools."""
    
    def __init__(self, name: str, url: str):
        super().__init__(name)
        self._url = url
        self._secret: Optional[str] = None
    
    def secret(self, key: str) -> "WebhookToolBuilder":
        """Set webhook secret."""
        self._secret = key
        return self
    
    def build(self) -> AgentTool:
        return {
            "name": self._name,
            "display_name": self._display_name or self._name,
            "description": self._description,
            "type": ToolType.HOOK,
            "require_approval": self._require_approval or None,
            "hook": {
                "url": self._url,
                "secret": self._secret,
                "input_schema": _to_json_schema(self._params),
            },
        }


# =============================================================================
# Public API
# =============================================================================

def tool(name: str) -> ClientToolBuilder:
    """Create a client tool (executed by SDK consumer)."""
    return ClientToolBuilder(name)


def app_tool(name: str, app_ref: str) -> AppToolBuilder:
    """Create an app tool (runs another inference app)."""
    return AppToolBuilder(name, app_ref)


def agent_tool(name: str, agent_ref: str) -> AgentToolBuilder:
    """Create an agent tool (delegates to sub-agent)."""
    return AgentToolBuilder(name, agent_ref)


def webhook_tool(name: str, url: str) -> WebhookToolBuilder:
    """Create a webhook tool (calls external URL)."""
    return WebhookToolBuilder(name, url)


# =============================================================================
# Internal Tools Builder
# =============================================================================

class InternalToolsBuilder:
    """Builder for internal tools configuration."""
    
    def __init__(self):
        self._config: InternalToolsConfig = {}
    
    def plan(self, enabled: bool = True) -> "InternalToolsBuilder":
        """Enable plan tools (Create, Update, Load)."""
        self._config["plan"] = enabled
        return self
    
    def memory(self, enabled: bool = True) -> "InternalToolsBuilder":
        """Enable memory tools (Set, Get, GetAll)."""
        self._config["memory"] = enabled
        return self
    
    def widget(self, enabled: bool = True) -> "InternalToolsBuilder":
        """Enable widget tools (UI, HTML) - top-level only."""
        self._config["widget"] = enabled
        return self
    
    def finish(self, enabled: bool = True) -> "InternalToolsBuilder":
        """Enable finish tool - sub-agents only."""
        self._config["finish"] = enabled
        return self
    
    def all(self) -> "InternalToolsBuilder":
        """Enable all internal tools."""
        self._config["plan"] = True
        self._config["memory"] = True
        self._config["widget"] = True
        self._config["finish"] = True
        return self
    
    def none(self) -> "InternalToolsBuilder":
        """Disable all internal tools."""
        self._config["plan"] = False
        self._config["memory"] = False
        self._config["widget"] = False
        self._config["finish"] = False
        return self
    
    def build(self) -> InternalToolsConfig:
        return self._config


def internal_tools() -> InternalToolsBuilder:
    """Create internal tools configuration."""
    return InternalToolsBuilder()

