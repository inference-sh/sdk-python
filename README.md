# inference.sh sdk

helper package for inference.sh python applications.

## installation

```bash
pip install infsh
```

## client usage

```python
from inferencesh import Inference, TaskStatus

# Create client
client = Inference(api_key="your-api-key")

# Simple synchronous usage
try:
    task = client.run({
        "app": "your-app",
        "input": {"key": "value"},
        "infra": "cloud",
        "variant": "default"
    })
    
    print(f"Task ID: {task.get('id')}")

    if task.get("status") == TaskStatus.COMPLETED:
        print("✓ Task completed successfully!")
        print(f"Output: {task.get('output')}")
    else:
        status = task.get("status")
        status_name = TaskStatus(status).name if status is not None else "UNKNOWN"
        print(f"✗ Task did not complete. Final status: {status_name}")

except Exception as exc:
    print(f"Error: {type(exc).__name__}: {exc}")
    raise  # Re-raise to see full traceback

# Streaming updates (recommended)
try:
    for update in client.run(
        {
            "app": "your-app",
            "input": {"key": "value"},
            "infra": "cloud",
            "variant": "default"
        },
        stream=True  # Enable streaming updates
    ):
        status = update.get("status")
        status_name = TaskStatus(status).name if status is not None else "UNKNOWN"
        print(f"Status: {status_name}")
        
        if status == TaskStatus.COMPLETED:
            print("✓ Task completed!")
            print(f"Output: {update.get('output')}")
            break
        elif status == TaskStatus.FAILED:
            print(f"✗ Task failed: {update.get('error')}")
            break
        elif status == TaskStatus.CANCELLED:
            print("✗ Task was cancelled")
            break

except Exception as exc:
    print(f"Error: {type(exc).__name__}: {exc}")
    raise  # Re-raise to see full traceback

# Async support
async def run_async():
    from inferencesh import AsyncInference
    
    client = AsyncInference(api_key="your-api-key")
    
    # Simple usage
    result = await client.run({
        "app": "your-app",
        "input": {"key": "value"},
        "infra": "cloud",
        "variant": "default"
    })
    
    # Stream updates
    async for update in await client.run(
        {
            "app": "your-app",
            "input": {"key": "value"},
            "infra": "cloud",
            "variant": "default"
        },
        stream=True
    ):
        status = update.get("status")
        status_name = TaskStatus(status).name if status is not None else "UNKNOWN"
        print(f"Status: {status_name}")
```

## file handling

the `File` class provides a standardized way to handle files in the inference.sh ecosystem:

```python
from infsh import File

# Basic file creation
file = File(path="/path/to/file.png")

# File with explicit metadata
file = File(
    path="/path/to/file.png",
    content_type="image/png",
    filename="custom_name.png",
    size=1024  # in bytes
)

# Create from path (automatically populates metadata)
file = File.from_path("/path/to/file.png")

# Check if file exists
exists = file.exists()

# Access file metadata
print(file.content_type)  # automatically detected if not specified
print(file.size)       # file size in bytes
print(file.filename)   # basename of the file

# Refresh metadata (useful if file has changed)
file.refresh_metadata()
```

the `File` class automatically handles:
- mime type detection
- file size calculation
- filename extraction from path
- file existence checking

## creating an app

to create an inference app, inherit from `BaseApp` and define your input/output types:

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

app lifecycle has three main methods:
- `setup()`: called when the app starts, use it to initialize models
- `run()`: called for each inference request
- `unload()`: called when shutting down, use it to free resources
