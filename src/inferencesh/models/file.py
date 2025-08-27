from typing import Optional, Union, Any
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import mimetypes
import os
import urllib.request
import urllib.parse
import tempfile
import hashlib
from pathlib import Path
from tqdm import tqdm


class File(BaseModel):
    """A class representing a file in the inference.sh ecosystem."""
    
    @classmethod
    def get_cache_dir(cls) -> Path:
        """Get the cache directory path based on environment variables or default location."""
        if cache_dir := os.environ.get("FILE_CACHE_DIR"):
            path = Path(cache_dir)
        else:
            path = Path.home() / ".cache" / "inferencesh" / "files"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _get_cache_path(self, url: str) -> Path:
        """Get the cache path for a URL using a hash-based directory structure."""
        # Parse URL components
        parsed_url = urllib.parse.urlparse(url)
        
        # Create hash from URL path and query parameters for uniqueness
        url_components = parsed_url.netloc + parsed_url.path
        if parsed_url.query:
            url_components += '?' + parsed_url.query
        url_hash = hashlib.sha256(url_components.encode()).hexdigest()[:12]
        
        # Get filename from URL or use default
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = 'download'
            
        # Create hash directory in cache
        cache_dir = self.get_cache_dir() / url_hash
        cache_dir.mkdir(exist_ok=True)
        
        return cache_dir / filename
    uri: Optional[str] = Field(default=None)  # Original location (URL or file path)
    path: Optional[str] = None  # Resolved local file path
    content_type: Optional[str] = None  # MIME type of the file
    size: Optional[int] = None  # File size in bytes
    filename: Optional[str] = None  # Original filename if available
    _tmp_path: Optional[str] = PrivateAttr(default=None)  # Internal storage for temporary file path
    
    def __init__(self, initializer=None, **data):
        if initializer is not None:
            if isinstance(initializer, str):
                data['uri'] = initializer
            elif isinstance(initializer, File):
                data = initializer.model_dump()
            else:
                raise ValueError(f'Invalid input for File: {initializer}')
        super().__init__(**data)

    @model_validator(mode='before')
    @classmethod
    def convert_str_to_file(cls, values):
        if isinstance(values, str):  # Only accept strings
            return {"uri": values}
        elif isinstance(values, dict):
            return values
        raise ValueError(f'Invalid input for File: {values}')
    
    @model_validator(mode='after')
    def validate_required_fields(self) -> 'File':
        """Validate that either uri or path is provided."""
        if not self.uri and not self.path:
            raise ValueError("Either 'uri' or 'path' must be provided")
        return self

    def model_post_init(self, _: Any) -> None:
        """Initialize file path and metadata after model creation.
        
        This method handles:
        1. Downloading URLs to local files if uri is a URL
        2. Converting relative paths to absolute paths
        3. Populating file metadata
        """
        # Handle uri if provided
        if self.uri:
            if self._is_url(self.uri):
                self._download_url()
            else:
                # Convert relative paths to absolute, leave absolute paths unchanged
                self.path = os.path.abspath(self.uri)
        
        # Handle path if provided
        if self.path:
            # Convert relative paths to absolute, leave absolute paths unchanged
            self.path = os.path.abspath(self.path)
            self._populate_metadata()
            return
            
        raise ValueError("Either 'uri' or 'path' must be provided and be valid")

    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        parsed = urllib.parse.urlparse(path)
        return parsed.scheme in ('http', 'https')

    def _download_url(self) -> None:
        """Download the URL to the cache directory and update the path."""
        original_url = self.uri
        cache_path = self._get_cache_path(original_url)
        
        # If file exists in cache, use it
        if cache_path.exists():
            print(f"Using cached file: {cache_path}")
            self.path = str(cache_path)
            return
            
        print(f"Downloading URL: {original_url} to {cache_path}")
        tmp_file = None
        try:
            # Download to temporary file first to avoid partial downloads in cache
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
            
            # Download the file with progress bar
            print(f"Downloading URL: {original_url} to {self._tmp_path}")
            try:
                with urllib.request.urlopen(req) as response:
                    # Safely retrieve content-length if available
                    total_size = 0
                    try:
                        if hasattr(response, 'headers') and response.headers is not None:
                            # urllib may expose headers as an email.message.Message
                            cl = response.headers.get('content-length')
                            total_size = int(cl) if cl is not None else 0
                        elif hasattr(response, 'getheader'):
                            cl = response.getheader('content-length')
                            total_size = int(cl) if cl is not None else 0
                    except Exception:
                        total_size = 0

                    block_size = 1024  # 1 Kibibyte
                    
                    with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                        with open(self._tmp_path, 'wb') as out_file:
                            while True:
                                non_chunking = False
                                try:
                                    buffer = response.read(block_size)
                                except TypeError:
                                    # Some mocks (or minimal implementations) expose read() without size
                                    buffer = response.read()
                                    non_chunking = True
                                if not buffer:
                                    break
                                out_file.write(buffer)
                                try:
                                    pbar.update(len(buffer))
                                except Exception:
                                    pass
                                if non_chunking:
                                    # If we read the whole body at once, exit loop
                                    break
                            
                # Move the temporary file to the cache location
                os.replace(self._tmp_path, cache_path)
                self._tmp_path = None  # Prevent deletion in __del__
                self.path = str(cache_path)
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                raise RuntimeError(f"Failed to download URL {original_url}: {str(e)}")
            except IOError as e:
                raise RuntimeError(f"Failed to write downloaded file to {self._tmp_path}: {str(e)}")
        except Exception as e:
            # Clean up temp file if something went wrong
            if tmp_file is not None and hasattr(self, '_tmp_path'):
                try:
                    os.unlink(self._tmp_path)
                except (OSError, IOError):
                    pass
            raise RuntimeError(f"Error downloading URL {original_url}: {str(e)}")

    def __del__(self):
        """Cleanup temporary file if it exists."""
        if hasattr(self, '_tmp_path') and self._tmp_path:
            try:
                os.unlink(self._tmp_path)
            except (OSError, IOError):
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
        if os.path.exists(self.path):
            self.content_type = self._guess_content_type()
            self.size = self._get_file_size()  # Always update size
            self.filename = self._get_filename()

    @classmethod
    def model_json_schema(cls, **kwargs):
        schema = super().model_json_schema(**kwargs)
        schema["$id"] = "/schemas/File"
        # Create a schema that accepts either a string or the full object
        return {
            "oneOf": [
                {"type": "string"},  # Accept string input
                schema  # Accept full object input
            ]
        } 