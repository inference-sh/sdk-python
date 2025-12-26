"""Kernel module for inference.sh app runtime.

This module provides the core functionality for loading, validating, and executing
inference.sh apps. It serves as the single source of truth for:

- App loading and setup (loader.py)
- App execution with backwards compatibility (executor.py)
- Schema generation (schema.py)
- Utility functions (utils.py)

Both the CLI (local development) and engine (production containers) import from
this module to ensure consistent behavior across all execution contexts.
"""

from .loader import setup_app, load_app_module, cleanup_build
from .executor import (
    AppContext,
    run_app,
    setup_app_instance,
    execute_app_run,
    validate_and_parse_input,
    load_and_setup_app,
)
from .schema import generate_schemas, generate_example_input, generate_example_value
from .utils import aislast

__all__ = [
    # Loader
    "setup_app",
    "load_app_module",
    "cleanup_build",
    # Executor
    "AppContext",
    "run_app",
    "setup_app_instance",
    "execute_app_run",
    "validate_and_parse_input",
    "load_and_setup_app",
    # Schema
    "generate_schemas",
    "generate_example_input",
    "generate_example_value",
    # Utils
    "aislast",
]
