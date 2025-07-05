from typing import Optional, List, Any, Callable, Dict, Generator
from enum import Enum
from pydantic import Field, BaseModel
from queue import Queue, Empty
from threading import Thread
import time
from contextlib import contextmanager
import base64

from .base import BaseAppInput, BaseAppOutput
from .file import File

class ContextMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"


class Message(BaseAppInput):
    role: ContextMessageRole
    content: str


class ContextMessage(BaseAppInput):
    role: ContextMessageRole = Field(
        description="the role of the message. user, assistant, or system",
    )
    text: str = Field(
        description="the text content of the message"
    )
    image: Optional[File] = Field(
        description="the image file of the message",
        default=None
    )

class BaseLLMInput(BaseAppInput):
    """Base class with common LLM fields."""
    system_prompt: str = Field(
        description="the system prompt to use for the model",
        default="you are a helpful assistant that can answer questions and help with tasks.",
        examples=[
            "you are a helpful assistant that can answer questions and help with tasks.",
        ]
    )
    context: List[ContextMessage] = Field(
        description="the context to use for the model",
        default=[],
        examples=[
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]}
            ]
        ]
    )
    text: str = Field(
        description="the user prompt to use for the model",
        examples=[
            "write a haiku about artificial general intelligence"
        ]
    )
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    context_size: int = Field(default=4096)

class ImageCapabilityMixin(BaseModel):
    """Mixin for models that support image inputs."""
    image: Optional[File] = Field(
        description="the image to use for the model",
        default=None,
        contentMediaType="image/*",
    )

class ReasoningCapabilityMixin(BaseModel):
    """Mixin for models that support reasoning."""
    reasoning: bool = Field(
        description="enable step-by-step reasoning",
        default=False
    )

class ToolsCapabilityMixin(BaseModel):
    """Mixin for models that support tool/function calling."""
    tools: Optional[List[Dict[str, Any]]] = Field(
        description="tool definitions for function calling",
        default=None
    )

# Example of how to use:
class LLMInput(BaseLLMInput):
    """Default LLM input model with no special capabilities."""
    pass

# For backward compatibility
LLMInput.model_config["title"] = "LLMInput"

class LLMUsage(BaseAppOutput):
    stop_reason: str = ""
    time_to_first_token: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_time: float = 0.0


class BaseLLMOutput(BaseAppOutput):
    """Base class for LLM outputs with common fields."""
    response: str = Field(description="the generated text response")

class LLMUsageMixin(BaseModel):
    """Mixin for models that provide token usage statistics."""
    usage: Optional[LLMUsage] = Field(
        description="token usage statistics",
        default=None
    )

class ReasoningMixin(BaseModel):
    """Mixin for models that support reasoning."""
    reasoning: Optional[str] = Field(
        description="the reasoning output of the model",
        default=None
    )

class ToolCallsMixin(BaseModel):
    """Mixin for models that support tool calls."""
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        description="tool calls for function calling",
        default=None
    )

# Example of how to use:
class LLMOutput(LLMUsageMixin, BaseLLMOutput):
    """Default LLM output model with token usage tracking."""
    pass

# For backward compatibility
LLMOutput.model_config["title"] = "LLMOutput"

@contextmanager
def timing_context():
    """Context manager to track timing information for LLM generation."""
    class TimingInfo:
        def __init__(self):
            self.start_time = time.time()
            self.first_token_time = None
            self.reasoning_start_time = None
            self.total_reasoning_time = 0.0
            self.reasoning_tokens = 0
            self.in_reasoning = False
        
        def mark_first_token(self):
            if self.first_token_time is None:
                self.first_token_time = time.time()
        
        def start_reasoning(self):
            if not self.in_reasoning:
                self.reasoning_start_time = time.time()
                self.in_reasoning = True
        
        def end_reasoning(self, token_count: int = 0):
            if self.in_reasoning and self.reasoning_start_time:
                self.total_reasoning_time += time.time() - self.reasoning_start_time
                self.reasoning_tokens += token_count
                self.reasoning_start_time = None
                self.in_reasoning = False
        
        @property
        def stats(self):
            current_time = time.time()
            if self.first_token_time is None:
                return {
                    "time_to_first_token": 0.0,
                    "generation_time": 0.0,
                    "reasoning_time": self.total_reasoning_time,
                    "reasoning_tokens": self.reasoning_tokens
                }
            
            time_to_first = self.first_token_time - self.start_time
            generation_time = current_time - self.first_token_time
            
            return {
                "time_to_first_token": time_to_first,
                "generation_time": generation_time,
                "reasoning_time": self.total_reasoning_time,
                "reasoning_tokens": self.reasoning_tokens
            }
    
    timing = TimingInfo()
    try:
        yield timing
    finally:
        pass

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

def build_messages(
    input_data: LLMInput,
    transform_user_message: Optional[Callable[[str], str]] = None
) -> List[Dict[str, Any]]:
    """Build messages for LLaMA.cpp chat completion.

    If any message includes image content, builds OpenAI-style multipart format.
    Otherwise, uses plain string-only format.
    """
    def render_message(msg: ContextMessage, allow_multipart: bool) -> str | List[dict]:
        parts = []
        text = transform_user_message(msg.text) if transform_user_message and msg.role == ContextMessageRole.USER else msg.text
        if text:
            parts.append({"type": "text", "text": text})
        if msg.image:
            if msg.image.path:
                image_data_uri = image_to_base64_data_uri(msg.image.path)
                parts.append({"type": "image_url", "image_url": {"url": image_data_uri}})
            elif msg.image.uri:
                parts.append({"type": "image_url", "image_url": {"url": msg.image.uri}})
        if allow_multipart:
            return parts
        if len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        raise ValueError("Image content requires multipart support")

    messages = [{"role": "system", "content": input_data.system_prompt}] if input_data.system_prompt is not None and input_data.system_prompt != "" else []

    def merge_messages(messages: List[ContextMessage]) -> ContextMessage:
        text = "\n\n".join(msg.text for msg in messages if msg.text)
        images = [msg.image for msg in messages if msg.image]
        image = images[0] if images else None # TODO: handle multiple images
        return ContextMessage(role=messages[0].role, text=text, image=image)

    user_input_text = ""
    if hasattr(input_data, "text"):
        user_input_text = transform_user_message(input_data.text) if transform_user_message else input_data.text
        
    user_input_image = None
    multipart = any(m.image for m in input_data.context)
    if hasattr(input_data, "image"):
        user_input_image = input_data.image
        multipart = multipart or input_data.image is not None

    user_msg = ContextMessage(role=ContextMessageRole.USER, text=user_input_text, image=user_input_image)

    input_data.context.append(user_msg)

    current_role = None
    current_messages = []
    
    for msg in input_data.context:
        if msg.role == current_role or current_role is None:
            current_messages.append(msg)
            current_role = msg.role
        else:
            messages.append({
                "role": current_role,
                "content": render_message(merge_messages(current_messages), allow_multipart=multipart)
            })
            current_messages = [msg]
            current_role = msg.role
    if len(current_messages) > 0:
        messages.append({
            "role": current_role,
            "content": render_message(merge_messages(current_messages), allow_multipart=multipart)
        })

    return messages


class StreamResponse:
    """Holds a single chunk of streamed response."""
    def __init__(self):
        self.content = ""
        self.tool_calls = None  # Changed from [] to None
        self.finish_reason = None
        self.timing_stats = {
            "time_to_first_token": None,  # Changed from 0.0 to None
            "generation_time": 0.0,
            "reasoning_time": 0.0,
            "reasoning_tokens": 0,
            "tokens_per_second": 0.0
        }
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "stop_reason": ""
        }

    def update_from_chunk(self, chunk: Dict[str, Any], timing: Any) -> None:
        """Update response state from a chunk."""
        # Update usage stats if present
        if "usage" in chunk:
            usage = chunk["usage"]
            if usage is not None:
                # Update usage stats preserving existing values if not provided
                self.usage_stats.update({
                    "prompt_tokens": usage.get("prompt_tokens", self.usage_stats["prompt_tokens"]),
                    "completion_tokens": usage.get("completion_tokens", self.usage_stats["completion_tokens"]),
                    "total_tokens": usage.get("total_tokens", self.usage_stats["total_tokens"])
                })
        
        # Get the delta from the chunk
        delta = chunk.get("choices", [{}])[0]
        
        # Extract content and tool calls from either message or delta
        if "message" in delta:
            message = delta["message"]
            self.content = message.get("content", "")
            if message.get("tool_calls"):
                self._update_tool_calls(message["tool_calls"])
            self.finish_reason = delta.get("finish_reason")
            if self.finish_reason:
                self.usage_stats["stop_reason"] = self.finish_reason
        elif "delta" in delta:
            delta_content = delta["delta"]
            self.content = delta_content.get("content", "")
            if delta_content.get("tool_calls"):
                self._update_tool_calls(delta_content["tool_calls"])
            self.finish_reason = delta.get("finish_reason")
            if self.finish_reason:
                self.usage_stats["stop_reason"] = self.finish_reason
        
        # Update timing stats
        timing_stats = timing.stats
        if self.timing_stats["time_to_first_token"] is None:
            self.timing_stats["time_to_first_token"] = timing_stats["time_to_first_token"]
        
        self.timing_stats.update({
            "generation_time": timing_stats["generation_time"],
            "reasoning_time": timing_stats["reasoning_time"],
            "reasoning_tokens": timing_stats["reasoning_tokens"]
        })
        
        # Calculate tokens per second only if we have valid completion tokens and generation time
        if self.usage_stats["completion_tokens"] > 0 and timing_stats["generation_time"] > 0:
            self.timing_stats["tokens_per_second"] = (
                self.usage_stats["completion_tokens"] / timing_stats["generation_time"]
            )
        
    
    def _update_tool_calls(self, new_tool_calls: List[Dict[str, Any]]) -> None:
        """Update tool calls, handling both full and partial updates."""
        if self.tool_calls is None:
            self.tool_calls = []
            
        for tool_delta in new_tool_calls:
            tool_id = tool_delta.get("id")
            if not tool_id:
                continue
                
            # Find or create tool call
            current_tool = next((t for t in self.tool_calls if t["id"] == tool_id), None)
            if not current_tool:
                current_tool = {
                    "id": tool_id,
                    "type": tool_delta.get("type", "function"),
                    "function": {"name": "", "arguments": ""}
                }
                self.tool_calls.append(current_tool)
            
            # Update tool call
            if "function" in tool_delta:
                func_delta = tool_delta["function"]
                if "name" in func_delta:
                    current_tool["function"]["name"] = func_delta["name"]
                if "arguments" in func_delta:
                    current_tool["function"]["arguments"] += func_delta["arguments"]
    
    def has_updates(self) -> bool:
        """Check if this response has any content, tool call, or usage updates."""
        has_content = bool(self.content)
        has_tool_calls = bool(self.tool_calls)
        has_usage = self.usage_stats["prompt_tokens"] > 0 or self.usage_stats["completion_tokens"] > 0
        has_finish = bool(self.finish_reason)
        
        return has_content or has_tool_calls or has_usage or has_finish
    
    def to_output(self, buffer: str, transformer: Any) -> tuple[BaseLLMOutput, str]:
        """Convert current state to LLMOutput."""        
        # Create usage object if we have stats
        usage = None
        if any(self.usage_stats.values()):
            usage = LLMUsage(
                stop_reason=self.usage_stats["stop_reason"],
                time_to_first_token=self.timing_stats["time_to_first_token"] or 0.0,
                tokens_per_second=self.timing_stats["tokens_per_second"],
                prompt_tokens=self.usage_stats["prompt_tokens"],
                completion_tokens=self.usage_stats["completion_tokens"],
                total_tokens=self.usage_stats["total_tokens"],
                reasoning_time=self.timing_stats["reasoning_time"],
                reasoning_tokens=self.timing_stats["reasoning_tokens"]
            )
        
        buffer, output, _ = transformer(self.content, buffer, usage)
        
        # Add tool calls if present and supported
        if self.tool_calls and hasattr(output, 'tool_calls'):
            output.tool_calls = self.tool_calls
            
        return output, buffer

class ResponseState:
    """Holds the state of response transformation."""
    def __init__(self):
        self.buffer = ""
        self.response = ""
        self.reasoning = None
        self.function_calls = None  # For future function calling support
        self.tool_calls = None      # List to accumulate tool calls
        self.current_tool_call = None  # Track current tool call being built
        self.usage = None  # Add usage field
        self.state_changes = {
            "reasoning_started": False,
            "reasoning_ended": False,
            "function_call_started": False,
            "function_call_ended": False,
            "tool_call_started": False,
            "tool_call_ended": False
        }

class ResponseTransformer:
    """Base class for transforming model responses."""
    def __init__(self, output_cls: type[BaseLLMOutput] = LLMOutput):
        self.state = ResponseState()
        self.output_cls = output_cls
        self.timing = None  # Will be set by stream_generate
    
    def clean_text(self, text: str) -> str:
        """Clean common tokens from the text and apply model-specific cleaning.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with common and model-specific tokens removed
        """
        if text is None:
            return ""
        
        # Common token cleaning across most models
        cleaned = (text.replace("<|im_end|>", "")
                      .replace("<|im_start|>", "")
                      .replace("<start_of_turn>", "")
                      .replace("<end_of_turn>", "")
                      .replace("<eos>", ""))
        return self.additional_cleaning(cleaned)
    
    def additional_cleaning(self, text: str) -> str:
        """Apply model-specific token cleaning.
        
        Args:
            text: Text that has had common tokens removed
            
        Returns:
            Text with model-specific tokens removed
        """
        return text
    
    def handle_reasoning(self, text: str) -> None:
        """Handle reasoning/thinking detection and extraction.
        
        Args:
            text: Cleaned text to process for reasoning
        """
        # Default implementation for <think> style reasoning
        if "<think>" in text and not self.state.state_changes["reasoning_started"]:
            self.state.state_changes["reasoning_started"] = True
            if self.timing:
                self.timing.start_reasoning()
        
        if "</think>" in text and not self.state.state_changes["reasoning_ended"]:
            self.state.state_changes["reasoning_ended"] = True
            if self.timing:
                # Estimate token count from character count (rough approximation)
                token_count = len(self.state.buffer.split("<think>")[1].split("</think>")[0]) // 4
                self.timing.end_reasoning(token_count)
        
        if "<think>" in self.state.buffer:
            parts = self.state.buffer.split("</think>", 1)
            if len(parts) > 1:
                self.state.reasoning = parts[0].split("<think>", 1)[1].strip()
                self.state.response = parts[1].strip()
            else:
                self.state.reasoning = self.state.buffer.split("<think>", 1)[1].strip()
                self.state.response = ""
        else:
            self.state.response = self.state.buffer
    
    def handle_function_calls(self, text: str) -> None:
        """Handle function call detection and extraction.
        
        Args:
            text: Cleaned text to process for function calls
        """
        # Default no-op implementation
        # Models can override this to implement function call handling
        pass
    
    def handle_tool_calls(self, text: str) -> None:
        """Handle tool call detection and extraction.
        
        Args:
            text: Cleaned text to process for tool calls
        """
        # Default no-op implementation
        # Models can override this to implement tool call handling
        pass
    
    def transform_chunk(self, chunk: str) -> None:
        """Transform a single chunk of model output.
        
        This method orchestrates the transformation process by:
        1. Cleaning the text
        2. Updating the buffer
        3. Processing various capabilities (reasoning, function calls, etc)
        
        Args:
            chunk: Raw text chunk from the model
        """
        cleaned = self.clean_text(chunk)
        self.state.buffer += cleaned
        
        # Process different capabilities
        self.handle_reasoning(cleaned)
        self.handle_function_calls(cleaned)
        self.handle_tool_calls(cleaned)
    
    def build_output(self) -> tuple[str, LLMOutput, dict]:
        """Build the final output tuple.
        
        Returns:
            Tuple of (buffer, LLMOutput, state_changes)
        """
        # Build base output with required fields
        output_data = {
            "response": self.state.response.strip(),
        }
        
        # Add optional fields if they exist
        if self.state.usage is not None:
            output_data["usage"] = self.state.usage
        if self.state.reasoning:
            output_data["reasoning"] = self.state.reasoning.strip()
        if self.state.function_calls:
            output_data["function_calls"] = self.state.function_calls
        if self.state.tool_calls:
            output_data["tool_calls"] = self.state.tool_calls
            
        output = self.output_cls(**output_data)
            
        return (
            self.state.buffer,
            output,
            self.state.state_changes
        )
    
    def __call__(self, piece: str, buffer: str, usage: Optional[LLMUsage] = None) -> tuple[str, LLMOutput, dict]:
        """Transform a piece of text and return the result.
        
        Args:
            piece: New piece of text to transform
            buffer: Existing buffer content
            usage: Optional usage statistics
            
        Returns:
            Tuple of (new_buffer, output, state_changes)
        """
        self.state.buffer = buffer
        if usage is not None:
            self.state.usage = usage
        self.transform_chunk(piece)
        return self.build_output()


def stream_generate(
    model: Any,
    messages: List[Dict[str, Any]],
    transformer: ResponseTransformer = ResponseTransformer(),
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop: Optional[List[str]] = None,
    verbose: bool = False,
    output_cls: type[BaseLLMOutput] = LLMOutput,
) -> Generator[BaseLLMOutput, None, None]:
    """Stream generate from LLaMA.cpp model with timing and usage tracking."""
        
    # Create queues for communication between threads
    response_queue = Queue()
    error_queue = Queue()
    keep_alive_queue = Queue()
    
    # Set the output class for the transformer
    transformer.output_cls = output_cls
    
    def _generate_worker():
        """Worker thread to run the model generation."""
        try:
            # Build completion kwargs
            completion_kwargs = {
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop
            }
            if tools is not None:
                completion_kwargs["tools"] = tools
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice
            
            # Signal that we're starting
            keep_alive_queue.put(("init", time.time()))
            
            completion = model.create_chat_completion(**completion_kwargs)
            
            for chunk in completion:
                response_queue.put(("chunk", chunk))
                # Update keep-alive timestamp
                keep_alive_queue.put(("alive", time.time()))
                
            # Signal completion
            response_queue.put(("done", None))
            
        except Exception as e:
            # Preserve the full exception with traceback
            import sys
            error_queue.put((e, sys.exc_info()[2]))
            response_queue.put(("error", str(e)))
    
    with timing_context() as timing:
        transformer.timing = timing
        
        # Start generation thread
        generation_thread = Thread(target=_generate_worker, daemon=True)
        generation_thread.start()
        
        # Initialize response state
        response = StreamResponse()
        buffer = ""
        
        # Keep-alive tracking
        last_activity = time.time()
        init_timeout = 30.0  # 30 seconds for initial response
        chunk_timeout = 10.0  # 10 seconds between chunks
        
        try:
            # Wait for initial setup
            try:
                msg_type, timestamp = keep_alive_queue.get(timeout=init_timeout)
                if msg_type != "init":
                    raise RuntimeError("Unexpected initialization message")
                last_activity = timestamp
            except Empty:
                raise RuntimeError(f"Model failed to initialize within {init_timeout} seconds")
            
            while True:
                # Check for errors - now with proper exception chaining
                if not error_queue.empty():
                    exc, tb = error_queue.get()
                    if isinstance(exc, Exception):
                        raise exc.with_traceback(tb)
                    else:
                        raise RuntimeError(f"Unknown error in worker thread: {exc}")
                
                # Check keep-alive
                try:
                    while not keep_alive_queue.empty():
                        _, timestamp = keep_alive_queue.get_nowait()
                        last_activity = timestamp
                except Empty:
                    # Ignore empty queue - this is expected
                    pass
                
                # Check for timeout
                if time.time() - last_activity > chunk_timeout:
                    raise RuntimeError(f"No response from model for {chunk_timeout} seconds")
                
                # Get next chunk
                try:
                    msg_type, data = response_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                if msg_type == "error":
                    # If we get an error message but no exception in error_queue,
                    # create a new error
                    raise RuntimeError(f"Generation error: {data}")
                elif msg_type == "done":
                    break
                
                chunk = data
                
                if verbose:
                    print(chunk)
                
                # Mark first token time
                if not timing.first_token_time:
                    timing.mark_first_token()
                
                # Update response state from chunk
                response.update_from_chunk(chunk, timing)
                
                # Yield output if we have updates
                if response.has_updates():
                    output, buffer = response.to_output(buffer, transformer)
                    yield output
                
                # Break if we're done
                if response.finish_reason:
                    break
            
            # Wait for generation thread to finish
            if generation_thread.is_alive():
                generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds
                if generation_thread.is_alive():
                    # Thread didn't finish - this shouldn't happen normally
                    raise RuntimeError("Generation thread failed to finish")
                    
        except Exception as e:
            # Check if there's a thread error we should chain with
            if not error_queue.empty():
                thread_exc, thread_tb = error_queue.get()
                if isinstance(thread_exc, Exception):
                    raise e from thread_exc
            # If no thread error, raise the original exception
            raise 