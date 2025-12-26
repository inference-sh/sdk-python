"""App execution logic for inference.sh apps.

This module handles the actual execution of user app code, including:
- Setup phase with backwards compatibility for signature variations
- Run phase with support for both async generators and regular async returns
- Proper input validation and error handling

This is the single source of truth for app execution logic, used by both
the CLI (local development) and engine (production containers).
"""

import inspect
import json
import os
import sys
from types import ModuleType
from typing import Any, AsyncGenerator, Callable, Optional, Type

from ..models.metadata import Metadata


class AppContext:
    """Context object holding app instance and execution state.
    
    This maintains the state needed during app execution:
    - The loaded app instance
    - The AppInput/AppOutput classes for validation
    - The AppSetup class (optional) for backwards compatibility
    - Setup completion status
    - Runtime metadata
    """
    
    def __init__(self, metadata: Optional[Metadata] = None):
        self.app = None
        self.setup_complete = False
        self.AppInput: Optional[Type] = None
        self.AppOutput: Optional[Type] = None
        self.AppSetup: Optional[Type] = None  # Optional, for backwards compatibility
        self.metadata = metadata or Metadata(
            app_id=os.environ.get("APP_ID", None),
            worker_id=os.environ.get("WORKER_ID", None)
        )


async def load_and_setup_app(
    context: AppContext,
    app_path: str,
    src_path: str,
    setup_data: Optional[Any] = None,
    module_name: str = "src.inference"
) -> None:
    """Load an app module and run setup.
    
    This is the primary entry point for engine/container mode. It:
    1. Adds app_path to sys.path if provided
    2. Changes to src_path as working directory
    3. Imports the App, AppInput, AppOutput classes
    4. Optionally imports AppSetup for backwards compatibility
    5. Creates app instance and runs setup with proper signature handling
    
    Args:
        context: AppContext to store the loaded app and classes
        app_path: Path to add to sys.path (can be empty)
        src_path: Path to change to as working directory
        setup_data: Optional setup data (dict or model instance)
        module_name: Module name to import (default: "src.inference")
        
    Raises:
        ImportError: If the module cannot be imported
        ValueError: If setup data type is invalid
    """
    # Add to path if provided
    if app_path and app_path not in sys.path:
        sys.path.insert(0, app_path)
        print(f"✓ added {app_path} to Python path")
    
    # Import the module dynamically
    import importlib
    module = importlib.import_module(module_name)
    
    # Change working directory
    os.chdir(src_path)
    
    # Extract classes
    context.app = module.App()
    context.AppInput = module.AppInput
    context.AppOutput = module.AppOutput
    
    # Try to import AppSetup (optional for backwards compatibility)
    try:
        context.AppSetup = module.AppSetup
    except AttributeError:
        pass
    
    # Run setup
    await setup_app_instance(context, setup_data)
    context.setup_complete = True


async def setup_app_instance(
    context: AppContext,
    setup_data: Optional[Any] = None,
) -> None:
    """Run setup on an app instance.
    
    This handles the setup phase with backwards compatibility:
    - Older apps: setup(metadata)
    - Newer apps: setup(setup_data, metadata)
    
    Args:
        context: AppContext containing the app instance
        setup_data: Optional setup data (dict, JSON string, or AppSetup instance)
        
    Raises:
        ValueError: If setup data type is invalid
    """
    # Parse and validate setup_data if AppSetup exists (backwards compatibility)
    setup_model = None
    if context.AppSetup is not None:
        SetupCls = context.AppSetup
        if setup_data is None:
            setup_model = SetupCls()
        elif isinstance(setup_data, str):
            try:
                setup_model = SetupCls.model_validate(json.loads(setup_data))
            except Exception as e:
                raise ValueError(f"Invalid setup data: {str(e)}")
        elif isinstance(setup_data, dict):
            setup_model = SetupCls(**setup_data)
        elif isinstance(setup_data, SetupCls):
            setup_model = setup_data
        else:
            raise ValueError(f"Invalid setup type: {type(setup_data)}")
    
    # Check setup signature for backwards compatibility
    # Older apps only accept metadata, newer ones accept (setup_data, metadata)
    setup_sig = inspect.signature(context.app.setup)
    setup_params = list(setup_sig.parameters.keys())
    
    if len(setup_params) >= 2:
        await context.app.setup(setup_model, context.metadata)
    else:
        await context.app.setup(metadata=context.metadata)


async def execute_app_run(
    context: AppContext,
    input_data: Any,
) -> AsyncGenerator[Any, None]:
    """Execute the app's run method and yield outputs.
    
    This handles both async generators and regular async returns,
    yielding outputs in a consistent manner.
    
    Args:
        context: AppContext containing the app instance
        input_data: Raw input data (dict or AppInput instance)
        
    Yields:
        Output from the app's run method
        
    Raises:
        ValueError: If input type is invalid
    """
    # Validate and parse input
    input_model = validate_and_parse_input(context.AppInput, input_data)
    
    # Execute
    result = context.app.run(input_data=input_model, metadata=context.metadata)
    
    if inspect.isasyncgen(result):
        # Handle async generator case
        async for output in result:
            yield output
    elif hasattr(result, "__aiter__"):
        # Handle any async iterable
        async for output in result:
            yield output
    else:
        # Handle regular async function case
        output = await result
        yield output


def validate_and_parse_input(
    InputCls: Type,
    input_data: Any
) -> Any:
    """Validate and parse input data into the app's AppInput model.
    
    Args:
        InputCls: The AppInput class to validate against
        input_data: Raw input data (dict, JSON string, or AppInput instance)
        
    Returns:
        Validated AppInput instance
        
    Raises:
        ValueError: If input type is invalid
    """
    if isinstance(input_data, dict):
        return InputCls(**input_data)
    elif isinstance(input_data, str):
        return InputCls.model_validate(json.loads(input_data))
    elif isinstance(input_data, InputCls):
        return input_data
    else:
        raise ValueError(f"Invalid input type: {type(input_data)}")


# ============================================================================
# CLI-specific convenience functions
# ============================================================================

async def run_app(
    module: ModuleType,
    setup_data: Optional[str] = None,
    input_data: Optional[str] = None,
    metadata: Optional[Metadata] = None
) -> AsyncGenerator[Any, None]:
    """Run the app with the given input data (CLI convenience function).
    
    This is a convenience function that combines setup and run phases.
    Used primarily by CLI for local development.
    
    Args:
        module: The loaded app module
        setup_data: JSON string containing setup data (optional)
        input_data: JSON string containing input data
        metadata: Runtime metadata
        
    Yields:
        Output from the app's run method
        
    Raises:
        Exception: If input/setup validation fails
    """
    try:
        # Create context
        context = AppContext(metadata=metadata)
        
        # Set up classes from module
        context.app = module.App()
        context.AppInput = module.AppInput
        context.AppOutput = module.AppOutput
        
        # Try to get AppSetup
        if hasattr(module, 'AppSetup'):
            context.AppSetup = module.AppSetup
        
        # Run setup
        await setup_app_instance(context, setup_data)
        
        # Parse input
        if input_data is None:
            input_model = context.AppInput()
        else:
            input_model = validate_and_parse_input(context.AppInput, input_data)

        # Execute and yield results
        async for output in execute_app_run(context, input_model):
            yield output

    except Exception as e:
        # Get the original error message without nesting
        error_msg = str(e)
        if "Error running app: " in error_msg:
            error_msg = error_msg.replace("Error running app: ", "")
        raise Exception(error_msg)
