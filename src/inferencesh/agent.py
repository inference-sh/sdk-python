"""
Headless Agent SDK

Chat with AI agents without UI dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Callable, Iterator, AsyncIterator
from dataclasses import dataclass, field

from .types import (
    ChatDTO,
    ChatMessageDTO,
    AgentTool,
    InternalToolsConfig,
    ToolType,
    ToolInvocationStatus,
)
from .client import StreamManager


@dataclass
class AgentConfig:
    """Configuration for the Agent client."""
    api_key: str
    base_url: str = "https://api.inference.sh"


@dataclass
class AdHocAgentOptions:
    """Ad-hoc agent configuration (no saved template)."""
    core_app: str
    """Core LLM app: namespace/name@shortid"""
    core_app_input: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[list[AgentTool]] = None
    internal_tools: Optional[InternalToolsConfig] = None


@dataclass
class TemplateAgentOptions:
    """Template agent configuration."""
    agent: str
    """Agent reference: namespace/name@version (e.g., "my-org/assistant@abc123")"""


AgentOptions = AdHocAgentOptions | TemplateAgentOptions


@dataclass
class ToolCallInfo:
    """Information about a pending tool call."""
    id: str
    name: str
    args: Dict[str, Any]


class Agent:
    """
    Headless agent client for chat interactions.
    
    Example:
        ```python
        config = AgentConfig(api_key="your-key")
        options = AdHocAgentOptions(core_app="infsh/claude-sonnet-4@abc123")
        
        agent = Agent(config, options)
        
        # Send a message
        response = agent.send_message("Hello!")
        
        # Stream messages
        for message in agent.stream_messages():
            print(message)
        ```
    """
    
    def __init__(self, config: AgentConfig, options: AgentOptions):
        self._api_key = config.api_key
        self._base_url = config.base_url
        self._options = options
        self._chat_id: Optional[str] = None
    
    @property
    def chat_id(self) -> Optional[str]:
        """Current chat ID."""
        return self._chat_id
    
    def send_message(
        self,
        text: str,
        files: Optional[list[bytes | str]] = None,
        on_message: Optional[Callable[[ChatMessageDTO], None]] = None,
        on_tool_call: Optional[Callable[[ToolCallInfo], None]] = None,
    ) -> ChatMessageDTO:
        """
        Send a message to the agent.
        
        Args:
            text: Message text
            files: File attachments (bytes or base64/data URI strings)
            on_message: Callback for streaming message updates
            on_tool_call: Callback when a client tool needs execution
            
        Returns:
            The assistant's response message
        """
        requests = _require_requests()
        
        # Upload files if provided
        image_uri: Optional[str] = None
        file_uris: Optional[list[str]] = None
        
        if files:
            uploaded = [self.upload_file(f) for f in files]
            images = [f for f in uploaded if f.get("content_type", "").startswith("image/")]
            others = [f for f in uploaded if not f.get("content_type", "").startswith("image/")]
            
            if images:
                image_uri = images[0]["uri"]
            if others:
                file_uris = [f["uri"] for f in others]
        
        is_adhoc = isinstance(self._options, AdHocAgentOptions)
        endpoint = "/agents/message"
        
        if is_adhoc:
            body = {
                "chat_id": self._chat_id,
                "core_app": self._options.core_app,
                "core_app_input": self._options.core_app_input,
                "name": self._options.name,
                "system_prompt": self._options.system_prompt,
                "tools": self._options.tools,
                "internal_tools": self._options.internal_tools,
                "input": {"text": text, "image": image_uri, "files": file_uris, "role": "user", "context": [], "system_prompt": "", "context_size": 0},
            }
        else:
            body = {
                "chat_id": self._chat_id,
                "agent": self._options.agent,
                "input": {"text": text, "image": image_uri, "files": file_uris, "role": "user", "context": [], "system_prompt": "", "context_size": 0},
            }
        
        response = self._request("post", endpoint, data=body)
        
        # Update chat ID
        assistant_msg = response.get("assistant_message", {})
        if not self._chat_id and assistant_msg.get("chat_id"):
            self._chat_id = assistant_msg["chat_id"]
        
        # Start streaming if callbacks provided
        if on_message or on_tool_call:
            self._start_streaming(on_message, on_tool_call)
        
        return assistant_msg
    
    def get_chat(self, chat_id: Optional[str] = None) -> Optional[ChatDTO]:
        """Get chat by ID."""
        cid = chat_id or self._chat_id
        if not cid:
            return None
        return self._request("get", f"/chats/{cid}")
    
    def stop_chat(self) -> None:
        """Stop the current chat generation."""
        if self._chat_id:
            self._request("post", f"/chats/{self._chat_id}/stop")
    
    def submit_tool_result(self, tool_invocation_id: str, result: str) -> None:
        """Submit a tool execution result."""
        if not self._chat_id:
            raise RuntimeError("No active chat")
        self._request("post", f"/chats/{self._chat_id}/tool-result", data={
            "tool_invocation_id": tool_invocation_id,
            "result": result,
        })
    
    def stream_messages(
        self,
        auto_reconnect: bool = True,
        max_reconnects: int = 5,
        reconnect_delay_ms: int = 1000,
    ) -> Iterator[ChatMessageDTO]:
        """
        Stream messages from the current chat with auto-reconnect.
        
        Args:
            auto_reconnect: Whether to automatically reconnect on connection loss
            max_reconnects: Maximum number of reconnection attempts
            reconnect_delay_ms: Delay between reconnection attempts in milliseconds
        
        Yields:
            ChatMessageDTO: Message updates
        """
        if not self._chat_id:
            raise RuntimeError("No active chat - send a message first")
        
        from queue import Queue
        import threading
        
        message_queue: Queue[ChatMessageDTO | Exception | None] = Queue()
        
        def create_event_source():
            return self._create_sse_generator(f"/chats/{self._chat_id}/messages/stream")
        
        manager = StreamManager(
            create_event_source=create_event_source,
            auto_reconnect=auto_reconnect,
            max_reconnects=max_reconnects,
            reconnect_delay_ms=reconnect_delay_ms,
            on_data=lambda msg: message_queue.put(msg),
            on_error=lambda err: message_queue.put(err),
            on_stop=lambda: message_queue.put(None),
        )
        
        # Run in background thread
        thread = threading.Thread(target=manager.connect, daemon=True)
        thread.start()
        
        try:
            while True:
                item = message_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            manager.stop()
    
    def stream_chat(
        self,
        auto_reconnect: bool = True,
        max_reconnects: int = 5,
        reconnect_delay_ms: int = 1000,
    ) -> Iterator[ChatDTO]:
        """
        Stream chat updates with auto-reconnect.
        
        Args:
            auto_reconnect: Whether to automatically reconnect on connection loss
            max_reconnects: Maximum number of reconnection attempts
            reconnect_delay_ms: Delay between reconnection attempts in milliseconds
        
        Yields:
            ChatDTO: Chat updates
        """
        if not self._chat_id:
            raise RuntimeError("No active chat - send a message first")
        
        from queue import Queue
        import threading
        
        chat_queue: Queue[ChatDTO | Exception | None] = Queue()
        
        def create_event_source():
            return self._create_sse_generator(f"/chats/{self._chat_id}/stream")
        
        manager = StreamManager(
            create_event_source=create_event_source,
            auto_reconnect=auto_reconnect,
            max_reconnects=max_reconnects,
            reconnect_delay_ms=reconnect_delay_ms,
            on_data=lambda chat: chat_queue.put(chat),
            on_error=lambda err: chat_queue.put(err),
            on_stop=lambda: chat_queue.put(None),
        )
        
        thread = threading.Thread(target=manager.connect, daemon=True)
        thread.start()
        
        try:
            while True:
                item = chat_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            manager.stop()
    
    def reset(self) -> None:
        """Reset the agent (start fresh chat)."""
        self._chat_id = None
    
    def upload_file(self, data: bytes | str) -> Dict[str, Any]:
        """
        Upload a file and return the file object.
        
        Args:
            data: File data (bytes, base64 string, or data URI)
            
        Returns:
            Dict with 'uri' and 'content_type'
        """
        import base64
        
        requests = _require_requests()
        
        # Determine content type and convert to bytes
        content_type = "application/octet-stream"
        raw_bytes: bytes
        
        if isinstance(data, bytes):
            raw_bytes = data
        elif data.startswith("data:"):
            # Data URI
            import re
            match = re.match(r"^data:([^;]+);base64,(.+)$", data)
            if not match:
                raise ValueError("Invalid data URI")
            content_type = match.group(1)
            raw_bytes = base64.b64decode(match.group(2))
        else:
            # Assume base64
            raw_bytes = base64.b64decode(data)
        
        # Create file record
        file_req = {"files": [{"uri": "", "content_type": content_type, "size": len(raw_bytes)}]}
        created = self._request("post", "/files", data=file_req)
        file_obj = created[0]
        
        upload_url = file_obj.get("upload_url")
        if not upload_url:
            raise RuntimeError("No upload URL")
        
        # Upload to signed URL
        resp = requests.put(upload_url, data=raw_bytes, headers={"Content-Type": content_type})
        if not resp.ok:
            raise RuntimeError("Upload failed")
        
        return {"uri": file_obj["uri"], "content_type": file_obj.get("content_type")}
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _create_sse_generator(self, endpoint: str):
        """Create an SSE generator for StreamManager."""
        requests = _require_requests()
        
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
        }
        
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        
        def generator():
            for line in resp.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
        
        return generator()
    
    def _start_streaming(
        self,
        on_message: Optional[Callable[[ChatMessageDTO], None]],
        on_tool_call: Optional[Callable[[ToolCallInfo], None]],
    ) -> None:
        """Start background streaming with auto-reconnect."""
        if not self._chat_id:
            return
        
        import threading
        
        def run_stream():
            for message in self.stream_messages(auto_reconnect=True):
                if on_message:
                    on_message(message)
                
                # Check for client tool invocations
                if on_tool_call and message.get("tool_invocations"):
                    for inv in message["tool_invocations"]:
                        if (inv.get("type") == ToolType.CLIENT and 
                            inv.get("status") == ToolInvocationStatus.AWAITING_INPUT):
                            on_tool_call(ToolCallInfo(
                                id=inv["id"],
                                name=inv.get("function", {}).get("name", ""),
                                args=inv.get("function", {}).get("arguments", {}),
                            ))
        
        # Run in background thread so send_message can return immediately
        thread = threading.Thread(target=run_stream, daemon=True)
        thread.start()
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an API request."""
        requests = _require_requests()
        
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        
        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            data=json.dumps(data) if data else None,
            timeout=30,
        )
        
        payload = resp.json() if resp.text else {}
        
        if not resp.ok or not payload.get("success"):
            error = payload.get("error", {})
            msg = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(msg or "Request failed")
        
        return payload.get("data")


# =============================================================================
# Async Agent
# =============================================================================

class AsyncAgent:
    """Async version of the Agent client."""
    
    def __init__(self, config: AgentConfig, options: AgentOptions):
        self._api_key = config.api_key
        self._base_url = config.base_url
        self._options = options
        self._chat_id: Optional[str] = None
    
    @property
    def chat_id(self) -> Optional[str]:
        return self._chat_id
    
    async def send_message(self, text: str) -> ChatMessageDTO:
        """Send a message to the agent."""
        is_adhoc = isinstance(self._options, AdHocAgentOptions)
        endpoint = "/agents/message"
        
        if is_adhoc:
            body = {
                "chat_id": self._chat_id,
                "core_app": self._options.core_app,
                "core_app_input": self._options.core_app_input,
                "name": self._options.name,
                "system_prompt": self._options.system_prompt,
                "tools": self._options.tools,
                "internal_tools": self._options.internal_tools,
                "input": {"text": text, "role": "user", "context": [], "system_prompt": "", "context_size": 0},
            }
        else:
            body = {
                "chat_id": self._chat_id,
                "agent": self._options.agent,
                "input": {"text": text, "role": "user", "context": [], "system_prompt": "", "context_size": 0},
            }
        
        response = await self._request("post", endpoint, data=body)
        
        assistant_msg = response.get("assistant_message", {})
        if not self._chat_id and assistant_msg.get("chat_id"):
            self._chat_id = assistant_msg["chat_id"]
        
        return assistant_msg
    
    async def get_chat(self, chat_id: Optional[str] = None) -> Optional[ChatDTO]:
        cid = chat_id or self._chat_id
        if not cid:
            return None
        return await self._request("get", f"/chats/{cid}")
    
    async def stop_chat(self) -> None:
        if self._chat_id:
            await self._request("post", f"/chats/{self._chat_id}/stop")
    
    async def submit_tool_result(self, tool_invocation_id: str, result: str) -> None:
        if not self._chat_id:
            raise RuntimeError("No active chat")
        await self._request("post", f"/chats/{self._chat_id}/tool-result", data={
            "tool_invocation_id": tool_invocation_id,
            "result": result,
        })
    
    async def stream_messages(self) -> AsyncIterator[ChatMessageDTO]:
        if not self._chat_id:
            raise RuntimeError("No active chat - send a message first")
        
        async for event in self._stream_sse(f"/chats/{self._chat_id}/messages/stream"):
            yield event
    
    def reset(self) -> None:
        self._chat_id = None
    
    async def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Any:
        aiohttp = await _require_aiohttp()
        
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method.upper(), url, headers=headers, json=data) as resp:
                payload = await resp.json() if resp.content_type == "application/json" else {}
                
                if not resp.ok or not payload.get("success"):
                    error = payload.get("error", {})
                    msg = error.get("message") if isinstance(error, dict) else str(error)
                    raise RuntimeError(msg or "Request failed")
                
                return payload.get("data")
    
    async def _stream_sse(self, endpoint: str) -> AsyncIterator[Dict[str, Any]]:
        aiohttp = await _require_aiohttp()
        
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                async for line in resp.content:
                    line_str = line.decode().strip()
                    if not line_str or line_str.startswith(":"):
                        continue
                    if line_str.startswith("data:"):
                        data_str = line_str[5:].strip()
                        if data_str:
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue


# =============================================================================
# Lazy imports
# =============================================================================

def _require_requests():
    try:
        import requests
        return requests
    except ImportError as exc:
        raise RuntimeError("Install requests: pip install requests") from exc


async def _require_aiohttp():
    try:
        import aiohttp
        return aiohttp
    except ImportError as exc:
        raise RuntimeError("Install aiohttp: pip install aiohttp") from exc

