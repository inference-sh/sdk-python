"""inference.sh Python SDK package."""

__version__ = "0.5.2"

from .models import (
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    BaseAppSetup,
    File,
    Metadata,
    # LLM types
    ContextMessageRole,
    Message,
    ContextMessage,
    LLMInput,
    LLMOutput,
    build_messages,
    stream_generate,
    timing_context,
    # OutputMeta types
    MetaItem,
    MetaItemType,
    TextMeta,
    ImageMeta,
    VideoMeta,
    VideoResolution,
    AudioMeta,
    RawMeta,
    OutputMeta,
)

from .utils import StorageDir, download
from .client import Inference, AsyncInference, UploadFileOptions, is_terminal_status, is_message_ready
from .types import TaskStatus, ChatMessageStatus
from .models.errors import APIError, RequirementsNotMetError, RequirementError, SetupAction

# Agent SDK (headless)
from .agent import Agent, AsyncAgent, AgentConfig, ToolCallInfo

# Tool Builder (fluent API)
from .tools import (
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
    array,
    obj,
    optional,
)

# Generated types for Agent/Chat functionality
from .types import (
    # Enums
    ChatStatus,
    ChatMessageRole,
    ChatMessageContentType,
    ToolType,
    ToolInvocationStatus,
    # Agent types
    AgentTool,
    AgentToolDTO,
    AgentRuntimeConfig,
    InternalToolsConfig,
    # Chat types
    ChatDTO,
    ChatMessageDTO,
    ChatData,
    ChatMessageContent,
    ChatTaskInput,
    ChatTaskContextMessage,
    # Tool types
    ToolCall,
    ToolCallFunction,
    ToolInvocationDTO,
    ToolResultRequest,
    Tool,
    ToolFunction,
    ToolParameters,
)


def inference(*, api_key: str, base_url: str | None = None) -> Inference:
    """Factory function for creating an Inference client (lowercase for branding).
    
    Example:
        ```python
        client = inference(api_key="your-api-key")
        ```
    """
    return Inference(api_key=api_key, base_url=base_url)


def async_inference(*, api_key: str, base_url: str | None = None) -> AsyncInference:
    """Factory function for creating an AsyncInference client (lowercase for branding).
    
    Example:
        ```python
        client = async_inference(api_key="your-api-key")
        ```
    """
    return AsyncInference(api_key=api_key, base_url=base_url)

__all__ = [
    # Base types
    "BaseApp",
    "BaseAppInput",
    "BaseAppOutput",
    "BaseAppSetup",
    "File",
    "Metadata",
    # LLM types
    "ContextMessageRole",
    "Message",
    "ContextMessage",
    "LLMInput",
    "LLMOutput",
    "build_messages",
    "stream_generate",
    "timing_context",
    # OutputMeta types
    "MetaItem",
    "MetaItemType",
    "TextMeta",
    "ImageMeta",
    "VideoMeta",
    "VideoResolution",
    "AudioMeta",
    "RawMeta",
    "OutputMeta",
    # Utils
    "StorageDir",
    "download",
    # Client
    "inference",
    "async_inference",
    "Inference",
    "AsyncInference",
    "UploadFileOptions",
    "TaskStatus",
    "ChatMessageStatus",
    "is_terminal_status",
    "is_message_ready",
    # Errors
    "APIError",
    "RequirementsNotMetError",
    "RequirementError",
    "SetupAction",
    # Generated types - Enums
    "ChatStatus",
    "ChatMessageRole",
    "ChatMessageContentType",
    "ToolType",
    "ToolInvocationStatus",
    # Generated types - Agent
    "Agent",
    "AgentTool",
    "AgentToolDTO",
    "AgentRuntimeConfig",
    # Generated types - Chat
    "ChatDTO",
    "ChatMessageDTO",
    "ChatData",
    "ChatMessageContent",
    "ChatTaskInput",
    "ChatTaskContextMessage",
    # Generated types - Tool
    "ToolCall",
    "ToolCallFunction",
    "ToolInvocationDTO",
    "ToolResultRequest",
    "Tool",
    "ToolFunction",
    "ToolParameters",
    # Agent SDK
    "Agent",
    "AsyncAgent",
    "AgentConfig",
    "AgentRuntimeConfig",
    "InternalToolsConfig",
    "ToolCallInfo",
    # Tool Builder
    "tool",
    "app_tool",
    "agent_tool",
    "webhook_tool",
    "internal_tools",
    "string",
    "number",
    "integer",
    "boolean",
    "enum_of",
    "array",
    "obj",
    "optional",
]
