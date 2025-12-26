"""Schema generation for inference.sh apps.

This module generates JSON schemas from Pydantic models and creates
example input data based on field definitions.
"""

import json
from types import ModuleType
from typing import Any, Dict


def generate_example_value(field_schema: Dict[str, Any]) -> Any:
    """Generate an example value based on a field schema.
    
    This recursively generates example values for any JSON schema type,
    using default values, example annotations, or sensible defaults.
    
    Args:
        field_schema: JSON schema definition for a field
        
    Returns:
        An example value appropriate for the field type
    """
    anyof = field_schema.get("anyOf", [])
    type_ = field_schema.get("type", "string")
    
    if anyof:
        for item in anyof:
            if item.get("type") != "null":
                type_ = item.get("type")
                break
    
    if type_ == "null":
        return None
    
    # If the key "default" exists and its value is None, return None
    if "default" in field_schema and field_schema["default"] is None:
        return None
    
    # Handle different types
    if type_ == "string":
        return field_schema.get("example") or field_schema.get("default") or "example_string"
    elif type_ == "integer":
        return field_schema.get("example") or field_schema.get("default") or 0
    elif type_ == "number":
        return field_schema.get("example") or field_schema.get("default") or 0.0
    elif type_ == "boolean":
        return field_schema.get("example") or field_schema.get("default") or False
    elif type_ == "array":
        items = field_schema.get("items", {})
        return [generate_example_value(items)]
    elif type_ == "object":
        properties = field_schema.get("properties", {})
        return {k: generate_example_value(v) for k, v in properties.items()}
    
    return None


def generate_example_input(module: ModuleType) -> str:
    """Generate example input JSON based on AppInput schema.
    
    Args:
        module: The loaded app module containing AppInput class
        
    Returns:
        JSON string with example input data
    """
    schema = module.AppInput.model_json_schema()
    
    # Generate example values for each field
    example_data = {}
    for field_name, field_schema in schema.get("properties", {}).items():
        example_data[field_name] = generate_example_value(field_schema)
    
    return json.dumps(example_data, indent=2)


def generate_schemas(
    module: ModuleType,
    output_dir: str = "."
) -> Dict[str, Dict[str, Any]]:
    """Generate and write input/output schemas to files.
    
    Args:
        module: The loaded app module
        output_dir: Directory to write schema files (default: current dir)
        
    Returns:
        Dict containing the generated schemas
        
    Raises:
        IOError: If schema files cannot be written
    """
    import os
    
    schemas = {}
    
    # Generate input schema
    input_schema = module.AppInput.model_json_schema()
    schemas["input"] = input_schema
    
    input_path = os.path.join(output_dir, "input_schema.json")
    try:
        with open(input_path, "w+", encoding='utf-8') as f:
            json.dump(input_schema, f, indent=2, sort_keys=False)
    except IOError as e:
        raise IOError(f"Failed to write input_schema.json: {str(e)}") from e
    
    # Generate output schema
    output_schema = module.AppOutput.model_json_schema()
    schemas["output"] = output_schema
    
    output_path = os.path.join(output_dir, "output_schema.json")
    try:
        with open(output_path, "w+", encoding='utf-8') as f:
            json.dump(output_schema, f, indent=2, sort_keys=False)
    except IOError as e:
        raise IOError(f"Failed to write output_schema.json: {str(e)}") from e
    
    # AppSetup is optional for backwards compatibility
    if hasattr(module, 'AppSetup'):
        setup_schema = module.AppSetup.model_json_schema()
        schemas["setup"] = setup_schema
        
        setup_path = os.path.join(output_dir, "setup_schema.json")
        try:
            with open(setup_path, "w+", encoding='utf-8') as f:
                json.dump(setup_schema, f, indent=2, sort_keys=False)
        except IOError as e:
            raise IOError(f"Failed to write setup_schema.json: {str(e)}") from e
    
    return schemas


def get_schema(module: ModuleType, schema_type: str = "input") -> Dict[str, Any]:
    """Get a schema without writing to files.
    
    Args:
        module: The loaded app module
        schema_type: Type of schema ('input', 'output', or 'setup')
        
    Returns:
        The JSON schema as a dictionary
        
    Raises:
        ValueError: If invalid schema_type
        AttributeError: If setup schema requested but AppSetup doesn't exist
    """
    if schema_type == "input":
        return module.AppInput.model_json_schema()
    elif schema_type == "output":
        return module.AppOutput.model_json_schema()
    elif schema_type == "setup":
        if not hasattr(module, 'AppSetup'):
            raise AttributeError("AppSetup not defined in module")
        return module.AppSetup.model_json_schema()
    else:
        raise ValueError(f"Invalid schema_type: {schema_type}")

