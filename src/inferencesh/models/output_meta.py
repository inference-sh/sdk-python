"""Output metadata types for pricing and usage tracking."""

from typing import Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class MetaItemType(str, Enum):
    """Type discriminator for MetaItem."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class VideoResolution(str, Enum):
    """Standard video resolution presets."""
    RES_480P = "480p"
    RES_720P = "720p"
    RES_1080P = "1080p"
    RES_1440P = "1440p"
    RES_4K = "4k"


class MetaItem(BaseModel):
    """Base class for input/output metadata items."""
    type: str  # "text", "image", "video", "audio"
    extra: Optional[dict[str, Any]] = Field(
        default=None,
        description="App-specific key-value pairs for custom pricing factors"
    )


class TextMeta(MetaItem):
    """Metadata for text inputs/outputs (e.g., LLM tokens)."""
    type: str = MetaItemType.TEXT.value
    tokens: int = Field(
        default=0,
        description="Token count - in inputs[] = input tokens, in outputs[] = output tokens"
    )


class ImageMeta(MetaItem):
    """Metadata for image inputs/outputs."""
    type: str = MetaItemType.IMAGE.value
    width: int = Field(default=0, description="Image width in pixels")
    height: int = Field(default=0, description="Image height in pixels")
    resolution_mp: float = Field(
        default=0,
        description="Resolution in megapixels (width * height / 1_000_000)"
    )
    steps: int = Field(default=0, description="Number of diffusion steps")
    count: int = Field(default=1, description="Number of images")


class VideoMeta(MetaItem):
    """Metadata for video inputs/outputs."""
    type: str = MetaItemType.VIDEO.value
    width: int = Field(default=0, description="Video width in pixels")
    height: int = Field(default=0, description="Video height in pixels")
    resolution_mp: float = Field(
        default=0,
        description="Resolution in megapixels per frame"
    )
    resolution: Optional[VideoResolution] = Field(
        default=None,
        description="Standard resolution preset (480p, 720p, 1080p, 1440p, 4k)"
    )
    seconds: float = Field(default=0, description="Duration in seconds")
    fps: int = Field(default=0, description="Frames per second")


class AudioMeta(MetaItem):
    """Metadata for audio inputs/outputs."""
    type: str = MetaItemType.AUDIO.value
    seconds: float = Field(default=0, description="Duration in seconds")
    sample_rate: int = Field(default=0, description="Sample rate in Hz")


class OutputMeta(BaseModel):
    """
    Structured metadata about task inputs and outputs for pricing calculation.
    
    Apps include this in their output to report what was consumed (inputs)
    and what was produced (outputs). The backend uses this with CEL expressions
    to calculate app-level pricing.
    
    Example usage in an LLM app:
        output_meta = OutputMeta(
            inputs=[TextMeta(tokens=150)],
            outputs=[TextMeta(tokens=500)]
        )
    
    Example usage in a video generation app:
        output_meta = OutputMeta(
            outputs=[VideoMeta(
                resolution=VideoResolution.RES_1080P,
                resolution_mp=2.07,
                seconds=10.5,
                fps=30
            )]
        )
    """
    inputs: List[MetaItem] = Field(
        default_factory=list,
        description="Metadata about consumed inputs"
    )
    outputs: List[MetaItem] = Field(
        default_factory=list,
        description="Metadata about produced outputs"
    )

