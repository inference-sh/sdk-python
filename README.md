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

# Simple synchronous usage - waits for completion by default
result = client.run({
    "app": "your-app",
    "input": {"key": "value"},
    "infra": "cloud",
    "variant": "default"
})

print(f"Task ID: {result.get('id')}")
print(f"Output: {result.get('output')}")
```

### run options

```python
# Wait for completion (default behavior)
result = client.run(params)  # wait=True is default

# Return immediately without waiting
task = client.run(params, wait=False)
task_id = task["id"]  # Use this to check status later

# Stream updates as they happen
for update in client.run(params, stream=True):
    print(f"Status: {TaskStatus(update['status']).name}")
    if update.get("status") == TaskStatus.COMPLETED:
        print(f"Output: {update.get('output')}")
```

### task management

```python
# Get current task state
task = client.get_task(task_id)
print(f"Status: {TaskStatus(task['status']).name}")

# Cancel a running task
client.cancel(task_id)

# Wait for a task to complete
result = client.wait_for_completion(task_id)

# Stream updates for an existing task
with client.stream_task(task_id) as stream:
    for update in stream:
        print(f"Status: {TaskStatus(update['status']).name}")
        if update.get("status") == TaskStatus.COMPLETED:
            print(f"Result: {update.get('output')}")
            break

# Access final result after streaming
print(f"Final result: {stream.result}")
```

### task status values

```python
from inferencesh import TaskStatus

TaskStatus.RECEIVED    # 1 - Task received by server
TaskStatus.QUEUED      # 2 - Task queued for processing
TaskStatus.SCHEDULED   # 3 - Task scheduled to a worker
TaskStatus.PREPARING   # 4 - Worker preparing environment
TaskStatus.SERVING     # 5 - Model being loaded
TaskStatus.SETTING_UP  # 6 - Task setup in progress
TaskStatus.RUNNING     # 7 - Task actively running
TaskStatus.UPLOADING   # 8 - Uploading results
TaskStatus.COMPLETED   # 9 - Task completed successfully
TaskStatus.FAILED      # 10 - Task failed
TaskStatus.CANCELLED   # 11 - Task was cancelled
```

### file upload

```python
from inferencesh import UploadFileOptions

# Upload from file path
file_obj = client.upload_file("/path/to/image.png")
print(f"URI: {file_obj['uri']}")

# Upload from bytes
file_obj = client.upload_file(
    b"raw bytes data",
    UploadFileOptions(
        filename="data.bin",
        content_type="application/octet-stream"
    )
)

# Upload with options
file_obj = client.upload_file(
    "/path/to/image.png",
    UploadFileOptions(
        filename="custom_name.png",
        content_type="image/png",
        public=True  # Make publicly accessible
    )
)
```

Note: Files in task input are automatically uploaded. You only need `upload_file()` for manual uploads.

## async client

```python
from inferencesh import AsyncInference, TaskStatus

async def main():
    client = AsyncInference(api_key="your-api-key")
    
    # Simple usage - wait for completion
    result = await client.run({
        "app": "your-app",
        "input": {"key": "value"},
        "infra": "cloud",
        "variant": "default"
    })
    print(f"Output: {result.get('output')}")
    
    # Return immediately without waiting
    task = await client.run(params, wait=False)
    
    # Stream updates
    async for update in await client.run(params, stream=True):
        print(f"Status: {TaskStatus(update['status']).name}")
        if update.get("status") == TaskStatus.COMPLETED:
            print(f"Output: {update.get('output')}")
    
    # Task management
    task = await client.get_task(task_id)
    await client.cancel(task_id)
    result = await client.wait_for_completion(task_id)
    
    # Stream existing task
    async with client.stream_task(task_id) as stream:
        async for update in stream:
            print(f"Update: {update}")
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
