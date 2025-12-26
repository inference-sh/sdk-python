"""Request and response models for inference.sh kernel operations."""

from typing import Any, Dict, Optional
from pydantic import BaseModel

from .metadata import Metadata


class APIRequest(BaseModel):
    """Generic API request wrapper with data and metadata."""
    data: Dict[str, Any]
    metadata: Metadata


class SetupRequest(BaseModel):
    """Request model for app setup operations."""
    setup: Optional[Any] = None
    metadata: Metadata


class RunRequest(BaseModel):
    """Request model for app run operations."""
    input: Any
    metadata: Metadata


class RunResponse(BaseModel):
    """Response model for app run operations."""
    output: Any


class PlaceholderAppInput(BaseModel):
    """Placeholder input model used before app is loaded."""
    class Config:
        extra = "allow"


class PlaceholderAppOutput(BaseModel):
    """Placeholder output model used before app is loaded."""
    class Config:
        extra = "allow"

