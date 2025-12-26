"""Error types for the inference.sh SDK.

These mirror the Go types in the API:
- RequirementsNotMetError
- RequirementError  
- SetupAction
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SetupAction:
    """Actionable info for resolving a missing requirement.
    
    Mirrors Go struct:
        type SetupAction struct {
            Type     string   `json:"type"`               // "add_secret" | "connect" | "add_scopes"
            Provider string   `json:"provider,omitempty"` // For integration actions
            Scopes   []string `json:"scopes,omitempty"`   // Scopes to request
        }
    """
    type: str  # "add_secret" | "connect" | "add_scopes"
    provider: Optional[str] = None  # For integration actions
    scopes: Optional[List[str]] = None  # Scopes to request
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional['SetupAction']:
        if not data:
            return None
        return cls(
            type=data.get("type", ""),
            provider=data.get("provider"),
            scopes=data.get("scopes"),
        )


@dataclass
class RequirementError:
    """A single missing requirement with actionable info.
    
    Mirrors Go struct:
        type RequirementError struct {
            Type    string       `json:"type"`    // "secret" | "integration" | "scope"
            Key     string       `json:"key"`     // The requirement key that's missing
            Message string       `json:"message"` // Human-readable error message
            Action  *SetupAction `json:"action,omitempty"`
        }
    """
    type: str  # "secret" | "integration" | "scope"
    key: str  # The requirement key that's missing
    message: str  # Human-readable error message
    action: Optional[SetupAction] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequirementError':
        return cls(
            type=data.get("type", ""),
            key=data.get("key", ""),
            message=data.get("message", ""),
            action=SetupAction.from_dict(data.get("action")),
        )


class RequirementsNotMetError(Exception):
    """Error raised when app requirements (secrets, integrations, scopes) are not met.
    
    This is raised for HTTP 412 responses that contain structured requirement errors.
    
    Mirrors Go struct:
        type RequirementsNotMetError struct {
            Errors []RequirementError
        }
    
    Attributes:
        errors: List of RequirementError objects describing what's missing
        status_code: HTTP status code (412)
        
    Example:
        ```python
        try:
            task = client.run(params)
        except RequirementsNotMetError as e:
            for err in e.errors:
                print(f"Missing {err.type}: {err.key}")
                if err.action:
                    print(f"  Fix: {err.action.type}")
        ```
    """
    def __init__(self, errors: List[RequirementError], status_code: int = 412):
        self.errors = errors
        self.status_code = status_code
        message = errors[0].message if errors else "requirements not met"
        super().__init__(message)
    
    @classmethod
    def from_response(cls, data: Dict[str, Any], status_code: int = 412) -> 'RequirementsNotMetError':
        """Create from API response data."""
        errors_data = data.get("errors", [])
        errors = [RequirementError.from_dict(e) for e in errors_data]
        return cls(errors, status_code)
    
    def __repr__(self) -> str:
        return f"RequirementsNotMetError(errors={self.errors!r})"


class APIError(Exception):
    """General API error with HTTP status and response details.
    
    Attributes:
        status_code: HTTP status code
        message: Error message
        response_body: Raw response body (if available)
    """
    def __init__(self, status_code: int, message: str, response_body: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"HTTP {status_code}: {message}")
    
    def __repr__(self) -> str:
        return f"APIError(status_code={self.status_code}, message={self.message!r})"

