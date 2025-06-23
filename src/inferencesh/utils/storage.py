from enum import Enum
from pathlib import Path


class StorageDir(str, Enum):
    """Standard storage directories used by the SDK."""
    DATA = "/app/data"   # Persistent storage/cache directory
    TEMP = "/app/tmp"    # Temporary storage directory
    CACHE = "/app/cache" # Cache directory

    @property
    def path(self) -> Path:
        """Get the Path object for this storage directory, ensuring it exists."""
        path = Path(self.value)
        path.mkdir(parents=True, exist_ok=True)
        return path 