from typing import Optional, Union, ClassVar
from pydantic import BaseModel, ConfigDict
import mimetypes
import os
import urllib.request
import urllib.parse
import tempfile

class BaseAppInput(BaseModel):
    pass

class BaseAppOutput(BaseModel):
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
            print(f"Downloading URL: {self.path}")
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
        # Create a temporary file with a suffix based on the URL path
        suffix = os.path.splitext(urllib.parse.urlparse(original_url).path)[1]
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self._tmp_path = tmp_file.name
        
        # Download the file
        print(f"Downloading URL: {original_url} to {self._tmp_path}")
        urllib.request.urlretrieve(original_url, self._tmp_path)
        self.path = self._tmp_path

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