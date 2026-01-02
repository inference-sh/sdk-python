# inference.sh sdk

helper package for inference.sh python applications.

## installation

```bash
pip install infsh
```

## client usage

```python
from inferencesh import inference, TaskStatus

# Create client
client = inference(api_key="your-api-key")

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

### with setup parameters

Setup parameters configure the app instance (e.g., model selection). Workers with matching setup are "warm" and skip the setup phase:

```python
result = client.run({
    "app": "your-app",
    "setup": {"model": "schnell"},  # Setup parameters
    "input": {"prompt": "hello"}
})
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

## agent chat

Chat with AI agents using `client.agent()`.

### using a template agent

Use an existing agent from your workspace by its `namespace/name@shortid`:

```python
from inferencesh import inference

client = inference(api_key="your-api-key")

# Create agent from template
agent = client.agent("my-org/assistant@abc123")

# Send a message with streaming
def on_message(msg):
    content = msg.get("content", [])
    for c in content:
        if c.get("type") == "text" and c.get("text"):
            print(c["text"], end="", flush=True)

response = agent.send_message("Hello!", on_message=on_message)
print(f"\nChat ID: {agent.chat_id}")
```

### creating an ad-hoc agent

Create agents on-the-fly without saving to your workspace:

```python
from inferencesh import inference, AdHocAgentOptions
from inferencesh import tool, string

client = inference(api_key="your-api-key")

# Define a client tool
weather_tool = (
    tool("get_weather")
    .description("Get current weather")
    .params({"city": string("City name")})
    .handler(lambda args: '{"temp": 72, "conditions": "sunny"}')
    .build()
)

# Create ad-hoc agent
agent = client.agent(AdHocAgentOptions(
    core_app="infsh/claude-sonnet-4@abc123",  # LLM to use
    system_prompt="You are a helpful assistant.",
    tools=[weather_tool]
))

def on_tool_call(call):
    print(f"[Tool: {call.name}]")
    # Tools with handlers are auto-executed

response = agent.send_message(
    "What's the weather in Paris?",
    on_message=on_message,
    on_tool_call=on_tool_call
)
```

### agent methods

| Method | Description |
|--------|-------------|
| `send_message(text, ...)` | Send a message to the agent |
| `get_chat(chat_id=None)` | Get chat history |
| `stop_chat(chat_id=None)` | Stop current generation |
| `submit_tool_result(tool_id, result)` | Submit result for a client tool |
| `stream_messages(chat_id=None, ...)` | Stream message updates |
| `stream_chat(chat_id=None, ...)` | Stream chat updates |
| `reset()` | Start a new conversation |

### async agent

```python
from inferencesh import async_inference

client = async_inference(api_key="your-api-key")
agent = client.agent("my-org/assistant@abc123")

response = await agent.send_message("Hello!")
```

## async client

```python
from inferencesh import async_inference, TaskStatus

async def main():
    client = async_inference(api_key="your-api-key")
    
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
