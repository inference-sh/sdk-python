# inference.sh CLI

Helper package for inference.sh Python applications.

## Installation

```bash
pip install infsh
```

## File Handling

The `File` class provides a standardized way to handle files in the inference.sh ecosystem:

```python
from infsh import File

# Basic file creation
file = File(path="/path/to/file.png")

# File with explicit metadata
file = File(
    path="/path/to/file.png",
    mime_type="image/png",
    filename="custom_name.png",
    size=1024  # in bytes
)

# Create from path (automatically populates metadata)
file = File.from_path("/path/to/file.png")

# Check if file exists
exists = file.exists()

# Access file metadata
print(file.mime_type)  # automatically detected if not specified
print(file.size)       # file size in bytes
print(file.filename)   # basename of the file

# Refresh metadata (useful if file has changed)
file.refresh_metadata()
```

The `File` class automatically handles:
- MIME type detection
- File size calculation
- Filename extraction from path
- File existence checking

## Creating an App

To create an inference app, inherit from `BaseApp` and define your input/output types:

```python
from infsh import BaseApp, BaseAppInput, BaseAppOutput, File

class AppInput(BaseAppInput):
    image: str  # URL or file path to image
    mask: str   # URL or file path to mask

class AppOutput(BaseAppOutput):
    image: File

class MyApp(BaseApp):
    async def setup(self):
        # Initialize your model here
        pass

    async def run(self, app_input: AppInput) -> AppOutput:
        # Process input and return output
        result_path = "/tmp/result.png"
        return AppOutput(image=File(path=result_path))

    async def unload(self):
        # Clean up resources
        pass
```

The app lifecycle has three main methods:
- `setup()`: Called when the app starts, use it to initialize models
- `run()`: Called for each inference request
- `unload()`: Called when shutting down, use it to free resources
