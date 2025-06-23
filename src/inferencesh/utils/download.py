import hashlib
import os
import urllib.parse
import shutil
from pathlib import Path
from typing import Union

from ..models.file import File
from .storage import StorageDir


def download(url: str, directory: Union[str, Path, StorageDir]) -> str:
    """Download a file to the specified directory and return its path.
    
    Args:
        url: The URL to download from
        directory: The directory to save the file to. Can be a string path, 
                  Path object, or StorageDir enum value.
        
    Returns:
        str: The path to the downloaded file
    """
    # Convert directory to Path
    dir_path = Path(directory)
    dir_path.mkdir(exist_ok=True)
    
    # Create hash directory from URL
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    hash_dir = dir_path / url_hash
    hash_dir.mkdir(exist_ok=True)
    
    # Keep original filename
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    if not filename:
        filename = 'download'
        
    output_path = hash_dir / filename
    
    # If file exists in directory and it's not a temp directory, return it
    if output_path.exists() and directory != StorageDir.TEMP:
        return str(output_path)
    
    # Download the file
    file = File(url)
    if file.path:
        shutil.copy2(file.path, output_path)
        # Prevent the File instance from deleting its temporary file
        file._tmp_path = None
        return str(output_path)
    
    raise RuntimeError(f"Failed to download {url}") 