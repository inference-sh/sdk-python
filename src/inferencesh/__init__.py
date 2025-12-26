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
from .client import Inference, AsyncInference, UploadFileOptions, TaskStatus
from .models.errors import APIError, RequirementsNotMetError, RequirementError, SetupAction

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
    "Inference",
    "AsyncInference",
    "UploadFileOptions",
    "TaskStatus",
    # Errors
    "APIError",
    "RequirementsNotMetError",
    "RequirementError",
    "SetupAction",
]
