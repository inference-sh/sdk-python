"""App loading and environment setup for inference.sh apps.

This module handles loading user app code into a Python module that can be
executed by the kernel. It supports two modes:

1. CLI mode: Copies source files to .infsh/build and imports as build.inference
2. Engine mode: Imports directly from src.inference in container

The loader validates that required classes (App, AppInput, AppOutput) exist.
"""

import importlib
import os
import shutil
import sys
from types import ModuleType
from typing import Optional


def setup_app(
    source_dir: str = ".",
    build_dir: Optional[str] = None,
    module_name: str = "build.inference"
) -> ModuleType:
    """Set up the app environment and return the loaded module.
    
    This is the primary entry point for CLI local development. It:
    1. Creates .infsh/build directory
    2. Copies all Python files and directories to build
    3. Imports the module as build.inference
    4. Validates required classes exist
    
    Args:
        source_dir: Directory containing the app source code (default: ".")
        build_dir: Custom build directory (default: .infsh/build)
        module_name: Module name to import as (default: "build.inference")
        
    Returns:
        The loaded module containing App, AppInput, AppOutput classes
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If required classes are missing
    """
    # Create build directory
    if build_dir is None:
        infsh_dir = ".infsh"
        build_dir = os.path.join(infsh_dir, "build")
    
    os.makedirs(build_dir, exist_ok=True)

    # Copy all Python files to build directory
    for item in os.listdir(source_dir):
        if item.startswith("."):
            continue
        
        src_path = os.path.join(source_dir, item)
        dst_path = os.path.join(build_dir, item)
        
        if item.endswith(".py"):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(item) and item != "infsh":
            # Copy directories (excluding hidden ones and .infsh)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)

    # Ensure parent of build/ is on sys.path so the module is importable
    parent_dir = os.path.abspath(os.path.dirname(build_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Import the module
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Failed to import {module_name}: {str(e)}") from e

    # Validate required classes and methods
    _validate_module(module)
    
    return module


def load_app_module(
    app_path: str,
    module_name: str = "src.inference"
) -> ModuleType:
    """Load an app module from a specified path.
    
    This is the primary entry point for engine/container mode. It:
    1. Adds app_path to sys.path
    2. Imports the module directly
    3. Validates required classes exist
    
    Args:
        app_path: Path to the app directory containing src/inference.py
        module_name: Module name to import (default: "src.inference")
        
    Returns:
        The loaded module containing App, AppInput, AppOutput classes
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If required classes are missing
    """
    # Add app_path to Python path
    if app_path and app_path not in sys.path:
        sys.path.insert(0, app_path)
        print(f"✓ added {app_path} to Python path")
    
    # Import the module
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Failed to import {module_name}: {str(e)}") from e
    
    # Validate required classes
    _validate_module(module)
    
    return module


def _validate_module(module: ModuleType) -> None:
    """Validate that a module has required classes.
    
    Args:
        module: The loaded module to validate
        
    Raises:
        AttributeError: If required classes are missing
    """
    required_attrs = ["App", "AppInput", "AppOutput"]
    missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
    
    if missing_attrs:
        raise AttributeError(
            f"Missing required classes in inference.py: {', '.join(missing_attrs)}"
        )


def cleanup_build(build_dir: str = ".infsh/build") -> None:
    """Clean up the build directory.
    
    Args:
        build_dir: Path to the build directory to clean up
    """
    try:
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
    except (OSError, IOError) as e:
        print(f"Warning: Failed to clean up build directory: {e}")

