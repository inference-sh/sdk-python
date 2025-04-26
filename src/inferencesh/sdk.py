from typing import Optional, Union, ClassVar
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
import mimetypes
import os
import urllib.request
import urllib.parse
import tempfile
from pydantic import Field
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
    uri: Optional[str] = None  # Original location (URL or file path)
    path: Optional[str] = None  # Resolved local file path
    content_type: Optional[str] = None  # MIME type of the file
    size: Optional[int] = None  # File size in bytes
    filename: Optional[str] = None  # Original filename if available
    _tmp_path: Optional[str] = PrivateAttr(default=None)  # Internal storage for temporary file path
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )

    @model_validator(mode='after')
    def check_uri_or_path(self) -> 'File':
        """Validate that either uri or path is provided."""
        if not self.uri and not self.path:
            raise ValueError("Either 'uri' or 'path' must be provided")
        return self

    def model_post_init(self, _: Any) -> None:
        if self.uri and self._is_url(self.uri):
            self._download_url()
        elif self.uri and not os.path.isabs(self.uri):
            self.path = os.path.abspath(self.uri)
        elif self.uri:
            self.path = self.uri
        self._populate_metadata()
    
    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        parsed = urllib.parse.urlparse(path)
        return parsed.scheme in ('http', 'https')

    def _download_url(self) -> None:
        """Download the URL to a temporary file and update the path."""
        original_url = self.uri
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
            if not self.content_type:
                self.content_type = self._guess_content_type()
            if not self.size:
                self.size = self._get_file_size()
            if not self.filename:
                self.filename = self._get_filename()
    
    @classmethod
    def from_path(cls, path: Union[str, os.PathLike]) -> 'File':
        """Create a File instance from a file path."""
        return cls(uri=str(path))
    
    def _guess_content_type(self) -> Optional[str]:
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


class ContextMessage(BaseModel):
    role: str = Field(
        description="The role of the message",
        enum=["user", "assistant", "system"]
    )
    text: str = Field(
        description="The text content of the message"
    )

class ContextMessageWithImage(ContextMessage):
    image: Optional[File] = Field(
        description="The image url of the message",
        default=None
    )

class LLMInput(BaseAppInput):
    system_prompt: str = Field(
        description="The system prompt to use for the model",
        default="You are a helpful assistant that can answer questions and help with tasks.",
        examples=[
            "You are a helpful assistant that can answer questions and help with tasks.",
            "You are a certified medical professional who can provide accurate health information.",
            "You are a certified financial advisor who can give sound investment guidance.",
            "You are a certified cybersecurity expert who can explain security best practices.",
            "You are a certified environmental scientist who can discuss climate and sustainability.",
        ]
    )
    context: list[ContextMessage] = Field(
        description="The context to use for the model",
        examples=[
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the weather like today?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "I apologize, but I don't have access to real-time weather information. You would need to check a weather service or app to get current weather conditions for your location."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Can you help me write a poem about spring?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "Here's a short poem about spring:\n\nGreen buds awakening,\nSoft rain gently falling down,\nNew life springs anew.\n\nWarm sun breaks through clouds,\nBirds return with joyful song,\nNature's sweet rebirth."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Explain quantum computing in simple terms"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "Quantum computing is like having a super-powerful calculator that can solve many problems at once instead of one at a time. While regular computers use bits (0s and 1s), quantum computers use quantum bits or \"qubits\" that can be both 0 and 1 at the same time - kind of like being in two places at once! This allows them to process huge amounts of information much faster than regular computers for certain types of problems."}]}
            ]
        ],
        default=[]
    )
    text: str = Field(
        description="The user prompt to use for the model",
        examples=[
            "What is the capital of France?",
            "What is the weather like today?",
            "Can you help me write a poem about spring?",
            "Explain quantum computing in simple terms"
        ],
    )

class LLMInputWithImage(LLMInput):
    image: Optional[File] = Field(
        description="The image to use for the model",
        default=None
    )