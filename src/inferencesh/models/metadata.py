"""Metadata model for inference.sh apps."""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class Metadata(BaseModel):
    """Runtime metadata passed to app setup and run methods.
    
    This class contains information about the current execution context,
    including app identification, version, variant, and worker information.
    
    Attributes:
        app_id: Unique identifier for the app
        app_version_id: Version identifier for the app
        app_variant: Name of the variant being executed
        worker_id: Identifier for the worker executing the app
    """
    app_id: Optional[str] = None
    app_version_id: Optional[str] = None
    app_variant: Optional[str] = None
    worker_id: Optional[str] = None
    
    def update(self, other: Dict[str, Any] | BaseModel) -> None:
        """Update metadata fields from another dict or BaseModel.
        
        Args:
            other: Dictionary or BaseModel containing fields to update
        """
        update_dict = other.model_dump() if isinstance(other, BaseModel) else other
        for key, value in update_dict.items():
            setattr(self, key, value)
    
    class Config:
        extra = "allow"

