"""
Headless Agent SDK

Chat with AI agents without UI dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Callable, Iterator, AsyncIterator
from dataclasses import dataclass

from .types import (
    ChatDTO,
    ChatMessageDTO,
    AgentRuntimeConfig,
    ToolType,
    ToolInvocationStatus,
)
from .client import StreamManager


@dataclass
class AgentConfig:
    """Configuration for the Agent client."""
    api_key: str
    base_url: str = "https://api.inference.sh"


# Agent options: either a template ref string or an ad-hoc config dict
AgentOptions = str | AgentRuntimeConfig


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
        options = AgentRuntimeConfig(core_app_ref="infsh/claude-sonnet-4@abc123")
        
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
        self._dispatched_tools: set[str] = set()  # tool invocation ids we've already processed
    
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
        # Clear dispatched tools from previous message
        self._dispatched_tools.clear()
        
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
        
        # Build request body - /agents/run accepts either "agent" (template ref) or "agent_config" (ad-hoc)
        input_data = {"text": text, "image": image_uri, "files": file_uris, "role": "user", "context": [], "system_prompt": "", "context_size": 0}
        if isinstance(self._options, str):
            body = {"chat_id": self._chat_id, "agent": self._options, "input": input_data}
        else:
            body = {"chat_id": self._chat_id, "agent_config": self._options, "input": input_data}
        
        response = self._request("post", "/agents/run", data=body)
        if not response:
            raise RuntimeError("Empty response from /agents/run")
        
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
    
    def submit_tool_result(
        self, 
        tool_invocation_id: str, 
        result_or_action: str | dict[str, Any]
    ) -> None:
        """
        Submit a tool result.
        
        Args:
            tool_invocation_id: The tool invocation ID
            result_or_action: Either a raw result string, or a dict with:
                - action: { "type": str, "payload"?: dict } (for widget actions)
                - form_data?: dict (optional form data for widgets)
                Dict values are JSON-serialized automatically.
        
        Example (raw result):
            agent.submit_tool_result("tool123", '{"success": true}')
        
        Example (widget action):
            agent.submit_tool_result("tool123", {
                "action": {"type": "confirm"},
                "form_data": {"name": "John"}
            })
        """
        # Serialize widget actions to JSON string
        if isinstance(result_or_action, str):
            result = result_or_action
        else:
            result = json.dumps(result_or_action)
        self._request("post", f"/tools/{tool_invocation_id}", data={"result": result})
    
    def stream_messages(
        self,
        auto_reconnect: bool = True,
        max_reconnects: int = 5,
        reconnect_delay_ms: int = 1000,
    ) -> Iterator[ChatMessageDTO]:
        """
        Stream messages from the current chat with auto-reconnect.
        Uses the unified stream endpoint with TypedEvents.
        
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
            # Use unified stream with TypedEvents
            return self._create_typed_sse_generator(f"/chats/{self._chat_id}/stream")
        
        def handle_event(event_tuple):
            event_type, data = event_tuple
            # Only yield chat_messages events
            if event_type == "chat_messages":
                message_queue.put(data)
        
        manager = StreamManager(
            create_event_source=create_event_source,
            auto_reconnect=auto_reconnect,
            max_reconnects=max_reconnects,
            reconnect_delay_ms=reconnect_delay_ms,
            on_data=handle_event,
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
        Uses the unified stream endpoint with TypedEvents.
        
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
            # Use unified stream with TypedEvents
            return self._create_typed_sse_generator(f"/chats/{self._chat_id}/stream")
        
        def handle_event(event_tuple):
            event_type, data = event_tuple
            # Only yield chats events
            if event_type == "chats":
                chat_queue.put(data)
        
        manager = StreamManager(
            create_event_source=create_event_source,
            auto_reconnect=auto_reconnect,
            max_reconnects=max_reconnects,
            reconnect_delay_ms=reconnect_delay_ms,
            on_data=handle_event,
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
    
    def stream_all(
        self,
        on_chat: Optional[Callable[[ChatDTO], None]] = None,
        on_message: Optional[Callable[[ChatMessageDTO], None]] = None,
        on_tool_call: Optional[Callable[["ToolCallInfo"], None]] = None,
    ) -> None:
        """
        Stream all events (Chat and ChatMessage) from the unified stream endpoint.
        Uses TypedEvents - single SSE connection for both event types.
        
        Automatically stops when the chat becomes idle (agent finished responding).
        
        Args:
            on_chat: Callback for Chat object updates (status changes)
            on_message: Callback for ChatMessage updates
            on_tool_call: Callback when a client tool needs execution
        """
        if not self._chat_id:
            raise RuntimeError("No active chat - send a message first")
        
        for event_type, data in self._create_typed_sse_generator(f"/chats/{self._chat_id}/stream"):
            if event_type == "chats":
                if on_chat:
                    on_chat(data)
                # Stop streaming when chat becomes idle (agent finished)
                if data.get("status") in ("idle", "completed"):
                    break
                    
            elif event_type == "chat_messages":
                if on_message:
                    on_message(data)
                
                # Check for client tool invocations awaiting input
                # (ID tracking handles duplicates, status field indicates message readiness)
                if on_tool_call:
                    for inv in data.get("tool_invocations") or []:
                        inv_id = inv.get("id")
                        if not inv_id or inv_id in self._dispatched_tools:
                            continue
                        if inv.get("type") == ToolType.CLIENT and inv.get("status") == ToolInvocationStatus.AWAITING_INPUT:
                            self._dispatched_tools.add(inv_id)
                            on_tool_call(ToolCallInfo(
                                id=inv_id,
                                name=inv.get("function", {}).get("name", ""),
                                args=inv.get("function", {}).get("arguments", {}),
                            ))
    
    def reset(self) -> None:
        """Reset the agent (start fresh chat)."""
        self._chat_id = None
        self._dispatched_tools.clear()
    
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
    
    def _create_typed_sse_generator(self, endpoint: str):
        """Create an SSE generator that yields (event_type, data) tuples for TypedEvents."""
        requests = _require_requests()
        
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
        }
        
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        
        def generator():
            current_event_type: Optional[str] = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    current_event_type = None  # Reset on empty line (event boundary)
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            data = json.loads(data_str)
                            yield (current_event_type or "message", data)
                        except json.JSONDecodeError:
                            continue
        
        return generator()
    
    def _start_streaming(
        self,
        on_message: Optional[Callable[[ChatMessageDTO], None]],
        on_tool_call: Optional[Callable[[ToolCallInfo], None]],
    ) -> None:
        """Start streaming and wait for completion.
        
        Uses unified stream with TypedEvents. Blocks until the chat is complete.
        Tool call callbacks run in a separate thread, allowing submit_tool_result
        to be called from within the callback.
        """
        if not self._chat_id:
            return
        
        # Run synchronously - stream_all blocks until the chat completes
        # Callbacks are invoked inline as events arrive
        self.stream_all(
            on_message=on_message,
            on_tool_call=on_tool_call,
        )
    
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
        
        # Try to parse JSON response
        payload = {}
        if resp.text:
            try:
                payload = json.loads(resp.text)
            except json.JSONDecodeError:
                # Some endpoints may return non-JSON (e.g., plain "true")
                # If response is OK, treat as success with no data
                if resp.ok:
                    return None
                raise RuntimeError(f"Invalid response: {resp.text[:200]}")
        
        if not resp.ok:
            error = payload.get("error", {})
            msg = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(msg or f"Request failed: {resp.status_code}")
        
        if not payload.get("success"):
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
        # Build request body - /agents/run accepts either "agent" (template ref) or "agent_config" (ad-hoc)
        input_data = {"text": text, "role": "user", "context": [], "system_prompt": "", "context_size": 0}
        if isinstance(self._options, str):
            body = {"chat_id": self._chat_id, "agent": self._options, "input": input_data}
        else:
            body = {"chat_id": self._chat_id, "agent_config": self._options, "input": input_data}
        
        response = await self._request("post", "/agents/run", data=body)
        
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
    
    async def submit_tool_result(
        self, 
        tool_invocation_id: str, 
        result_or_action: str | dict[str, Any]
    ) -> None:
        """
        Submit a tool result.
        
        Args:
            tool_invocation_id: The tool invocation ID
            result_or_action: Either a raw result string, or a dict with:
                - action: { "type": str, "payload"?: dict } (for widget actions)
                - form_data?: dict (optional form data for widgets)
                Dict values are JSON-serialized automatically.
        """
        # Serialize widget actions to JSON string
        if isinstance(result_or_action, str):
            result = result_or_action
        else:
            result = json.dumps(result_or_action)
        await self._request("post", f"/tools/{tool_invocation_id}", data={"result": result})
    
    async def stream_messages(self) -> AsyncIterator[ChatMessageDTO]:
        """Stream messages from the unified stream endpoint with TypedEvents."""
        if not self._chat_id:
            raise RuntimeError("No active chat - send a message first")
        
        async for event_type, data in self._stream_typed_sse(f"/chats/{self._chat_id}/stream"):
            if event_type == "chat_messages":
                yield data
    
    async def stream_chat(self) -> AsyncIterator[ChatDTO]:
        """Stream chat updates from the unified stream endpoint with TypedEvents."""
        if not self._chat_id:
            raise RuntimeError("No active chat - send a message first")
        
        async for event_type, data in self._stream_typed_sse(f"/chats/{self._chat_id}/stream"):
            if event_type == "chats":
                yield data
    
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
        """Stream SSE events (yields raw data without event type)."""
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
    
    async def _stream_typed_sse(self, endpoint: str) -> AsyncIterator[tuple[str, Dict[str, Any]]]:
        """Stream SSE events with TypedEvents (yields event_type, data tuples)."""
        aiohttp = await _require_aiohttp()
        
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                current_event_type: Optional[str] = None
                async for line in resp.content:
                    line_str = line.decode().strip()
                    if not line_str:
                        current_event_type = None  # Reset on empty line (event boundary)
                        continue
                    if line_str.startswith(":"):
                        continue
                    if line_str.startswith("event:"):
                        current_event_type = line_str[6:].strip()
                        continue
                    if line_str.startswith("data:"):
                        data_str = line_str[5:].strip()
                        if data_str:
                            try:
                                data = json.loads(data_str)
                                yield (current_event_type or "message", data)
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

