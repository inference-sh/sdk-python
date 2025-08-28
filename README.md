# inference.sh sdk

helper package for inference.sh python applications.

## installation

```bash
pip install infsh
```

## client usage

```python
from infsh import Inference, TaskStatus

# create client
client = Inference(api_key="your-api-key")

# simple usage - wait for result
result = client.run({
    "app": "your-app",
    "input": {"key": "value"},
    "variant": "default"
})
print(f"output: {result['output']}")

# get task info without waiting
task = client.run(params, wait=False)
print(f"task id: {task['id']}")

# stream updates (recommended)
for update in client.run(params, stream=True):
    status = update.get("status")
    print(f"status: {TaskStatus(status).name}")
    
    if status == TaskStatus.COMPLETED:
        print(f"output: {update.get('output')}")
        break
    elif status == TaskStatus.FAILED:
        print(f"error: {update.get('error')}")
        break

# async support
async def run_async():
    from infsh import AsyncInference
    
    client = AsyncInference(api_key="your-api-key")
    
    # simple usage
    result = await client.run(params)
    
    # stream updates
    async for update in await client.run(params, stream=True):
        print(f"status: {TaskStatus(update['status']).name}")
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
