from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict
import inspect
import ast
import textwrap
from collections import OrderedDict


class OrderedSchemaModel(BaseModel):
    """A base model that ensures the JSON schema properties and required fields are in the order of field definition."""

    @classmethod
    def model_json_schema(cls, by_alias: bool = True, **kwargs: Any) -> Dict[str, Any]:
        schema = super().model_json_schema(by_alias=by_alias, **kwargs)

        field_order = cls._get_field_order()

        if field_order:
            # Order properties
            ordered_properties = OrderedDict()
            for field_name in field_order:
                if field_name in schema['properties']:
                    ordered_properties[field_name] = schema['properties'][field_name]

            # Add any remaining properties that weren't in field_order
            for field_name, field_schema in schema['properties'].items():
                if field_name not in ordered_properties:
                    ordered_properties[field_name] = field_schema

            schema['properties'] = ordered_properties

            # Order required fields
            if 'required' in schema:
                ordered_required = [field for field in field_order if field in schema['required']]
                # Add any remaining required fields that weren't in field_order
                ordered_required.extend([field for field in schema['required'] if field not in ordered_required])
                schema['required'] = ordered_required

        return schema

    @classmethod
    def _get_field_order(cls) -> List[str]:
        """Get the order of fields as they were defined in the class."""
        source = inspect.getsource(cls)

        # Unindent the entire source code
        source = textwrap.dedent(source)

        try:
            module = ast.parse(source)
        except IndentationError:
            # If we still get an IndentationError, wrap the class in a dummy module
            source = f"class DummyModule:\n{textwrap.indent(source, '    ')}"
            module = ast.parse(source)
            # Adjust to look at the first class def inside DummyModule
            # noinspection PyUnresolvedReferences
            class_def = module.body[0].body[0]
        else:
            # Find the class definition
            class_def = next(
                node for node in module.body if isinstance(node, ast.ClassDef) and node.name == cls.__name__
            )

        # Extract field names in the order they were defined
        field_order = []
        for node in class_def.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_order.append(node.target.id)

        return field_order


class BaseAppInput(OrderedSchemaModel):
    pass


class BaseAppOutput(OrderedSchemaModel):
    pass


class BaseApp(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )

    async def setup(self):
        pass

    async def run(self, app_input: BaseAppInput) -> BaseAppOutput:
        raise NotImplementedError("run method must be implemented")

    async def unload(self):
        pass 