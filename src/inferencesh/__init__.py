"""inference.sh Python SDK package."""

__version__ = "0.1.2"

from .models import (
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    File,
    ContextMessageRole,
    Message,
    ContextMessage,
    LLMInput,
    LLMOutput,
    build_messages,
    stream_generate,
    timing_context,
)
from .models import (
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
    "BaseApp",
    "BaseAppInput",
    "BaseAppOutput",
    "File",
    "ContextMessageRole",
    "Message",
    "ContextMessage",
    "LLMInput",
    "LLMOutput",
    "build_messages",
    "stream_generate",
    "timing_context",
    "StorageDir",
    "download",
    "Inference",
    "AsyncInference",
    "UploadFileOptions",
    "TaskStatus",
    "APIError",
    "RequirementsNotMetError",
    "RequirementError",
    "SetupAction",
    "MetaItem",
    "MetaItemType",
    "TextMeta",
    "ImageMeta",
    "VideoMeta",
    "VideoResolution",
    "AudioMeta",
    "RawMeta",
    "OutputMeta",
]