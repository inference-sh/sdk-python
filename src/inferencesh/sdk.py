from typing import Optional, Union, ClassVar
from pydantic import BaseModel, ConfigDict
import mimetypes
import os
import urllib.request
import urllib.parse
import tempfile

from typing import Any, Dict, List
import inspect
import ast
import textwrap
from collections import OrderedDict


# inspired by https://github.com/pydantic/pydantic/issues/7580
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


class File(BaseModel):
    """A class representing a file in the inference.sh ecosystem."""
    path: str  # Absolute path to the file or URL
    mime_type: Optional[str] = None  # MIME type of the file
    size: Optional[int] = None  # File size in bytes
    filename: Optional[str] = None  # Original filename if available
    _tmp_path: Optional[str] = None  # Internal storage for temporary file path
    
    def __init__(self, **data):
        super().__init__(**data)
        if self._is_url(self.path):
            self._download_url()
        elif not os.path.isabs(self.path):
            self.path = os.path.abspath(self.path)
        self._populate_metadata()
    
    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        parsed = urllib.parse.urlparse(path)
        return parsed.scheme in ('http', 'https')
    def _download_url(self) -> None:
        """Download the URL to a temporary file and update the path."""
        original_url = self.path
        tmp_file = None
        try:
            # Create a temporary file with a suffix based on the URL path
            suffix = os.path.splitext(urllib.parse.urlparse(original_url).path)[1]
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            self._tmp_path = tmp_file.name
            
            # Set up request with user agent
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/91.0.4472.124 Safari/537.36'
                )
            }
            req = urllib.request.Request(original_url, headers=headers)
            
            # Download the file
            print(f"Downloading URL: {original_url} to {self._tmp_path}")
            try:
                with urllib.request.urlopen(req) as response, open(self._tmp_path, 'wb') as out_file:
                    out_file.write(response.read())
                self.path = self._tmp_path
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                raise RuntimeError(f"Failed to download URL {original_url}: {str(e)}")
            except IOError as e:
                raise RuntimeError(f"Failed to write downloaded file to {self._tmp_path}: {str(e)}")
        except Exception as e:
            # Clean up temp file if something went wrong
            if tmp_file is not None and hasattr(self, '_tmp_path'):
                try:
                    os.unlink(self._tmp_path)
                except:
                    pass
            raise RuntimeError(f"Error downloading URL {original_url}: {str(e)}")

    def __del__(self):
        """Cleanup temporary file if it exists."""
        if hasattr(self, '_tmp_path') and self._tmp_path:
            try:
                os.unlink(self._tmp_path)
            except:
                pass

    def _populate_metadata(self) -> None:
        """Populate file metadata from the path if it exists."""
        if os.path.exists(self.path):
            if not self.mime_type:
                self.mime_type = self._guess_mime_type()
            if not self.size:
                self.size = self._get_file_size()
            if not self.filename:
                self.filename = self._get_filename()
    
    @classmethod
    def from_path(cls, path: Union[str, os.PathLike]) -> 'File':
        """Create a File instance from a file path."""
        return cls(path=str(path))
    
    def _guess_mime_type(self) -> Optional[str]:
        """Guess the MIME type of the file."""
        return mimetypes.guess_type(self.path)[0]
    
    def _get_file_size(self) -> int:
        """Get the size of the file in bytes."""
        return os.path.getsize(self.path)
    
    def _get_filename(self) -> str:
        """Get the base filename from the path."""
        return os.path.basename(self.path)
    
    def exists(self) -> bool:
        """Check if the file exists."""
        return os.path.exists(self.path)
    
    def refresh_metadata(self) -> None:
        """Refresh all metadata from the file."""
        self._populate_metadata()
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True 