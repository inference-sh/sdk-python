from typing import Optional, List, Any, Callable, Dict, Generator
from enum import Enum
from pydantic import Field
from queue import Queue
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
        description="The role of the message",
    )
    text: str = Field(
        description="The text content of the message"
    )
    image: Optional[File] = Field(
        description="The image url of the message",
        default=None
    )

class LLMInput(BaseAppInput):
    system_prompt: str = Field(
        description="The system prompt to use for the model",
        default="You are a helpful assistant that can answer questions and help with tasks.",
        examples=[
            "You are a helpful assistant that can answer questions and help with tasks.",
            "You are a certified medical professional who can provide accurate health information.",
            "You are a certified financial advisor who can give sound investment guidance.",
            "You are a certified cybersecurity expert who can explain security best practices.",
            "You are a certified environmental scientist who can discuss climate and sustainability.",
        ]
    )
    context: List[ContextMessage] = Field(
        description="The context to use for the model",
        examples=[
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the weather like today?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "I apologize, but I don't have access to real-time weather information. You would need to check a weather service or app to get current weather conditions for your location."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Can you help me write a poem about spring?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "Here's a short poem about spring:\n\nGreen buds awakening,\nSoft rain gently falling down,\nNew life springs anew.\n\nWarm sun breaks through clouds,\nBirds return with joyful song,\nNature's sweet rebirth."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Explain quantum computing in simple terms"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "Quantum computing is like having a super-powerful calculator that can solve many problems at once instead of one at a time. While regular computers use bits (0s and 1s), quantum computers use quantum bits or \"qubits\" that can be both 0 and 1 at the same time - kind of like being in two places at once! This allows them to process huge amounts of information much faster than regular computers for certain types of problems."}]}
            ]
        ],
        default=[]
    )
    text: str = Field(
        description="The user prompt to use for the model",
        examples=[
            "What is the capital of France?",
            "What is the weather like today?",
            "Can you help me write a poem about spring?",
            "Explain quantum computing in simple terms"
        ],
    )
    image: Optional[File] = Field(
        description="The image to use for the model",
        default=None
    )
    # Optional parameters
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    max_tokens: int = Field(default=4096)
    context_size: int = Field(default=4096)
    
    # Model specific flags
    reasoning: bool = Field(default=False)
    
    tools: List[Dict[str, Any]] = Field(default=[])

class LLMUsage(BaseAppOutput):
    stop_reason: str = ""
    time_to_first_token: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_time: float = 0.0


class LLMOutput(BaseAppOutput):
    response: str
    reasoning: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[LLMUsage] = None


@contextmanager
def timing_context():
    """Context manager to track timing information for LLM generation."""
    class TimingInfo:
        def __init__(self):
            self.start_time = time.time()
            self.first_token_time = 0
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
            end_time = time.time()
            if self.first_token_time is None:
                self.first_token_time = end_time
            
            time_to_first = self.first_token_time - self.start_time
            generation_time = end_time - self.first_token_time
            
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

    multipart = any(m.image for m in input_data.context) or input_data.image is not None
    messages = [{"role": "system", "content": input_data.system_prompt}] if input_data.system_prompt is not None and input_data.system_prompt != "" else []

    for msg in input_data.context:
        messages.append({
            "role": msg.role,
            "content": render_message(msg, allow_multipart=multipart)
        })

    user_msg = ContextMessage(role=ContextMessageRole.USER, text=input_data.text, image=input_data.image)
    messages.append({
        "role": "user",
        "content": render_message(user_msg, allow_multipart=multipart)
    })

    return messages


class StreamResponse:
    """Holds a single chunk of streamed response."""
    def __init__(self):
        self.content = ""
        self.tool_calls = []
        self.finish_reason = None
        self.timing_stats = {
            "time_to_first_token": 0.0,
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
        if "usage" in chunk and chunk["usage"] is not None:
            self.usage_stats.update(chunk["usage"])
        
        # Get the delta from the chunk
        delta = chunk.get("choices", [{}])[0]
        
        # Extract content and tool calls from either message or delta
        if "message" in delta:
            message = delta["message"]
            self.content = message.get("content", "")
            if message.get("tool_calls"):
                self._update_tool_calls(message["tool_calls"])
            self.finish_reason = delta.get("finish_reason")
        elif "delta" in delta:
            delta_content = delta["delta"]
            self.content = delta_content.get("content", "")
            if delta_content.get("tool_calls"):
                self._update_tool_calls(delta_content["tool_calls"])
            self.finish_reason = delta.get("finish_reason")
        
        # Update timing stats
        self.timing_stats = timing.stats
    
    def _update_tool_calls(self, new_tool_calls: List[Dict[str, Any]]) -> None:
        """Update tool calls, handling both full and partial updates."""
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
        """Check if this response has any content or tool call updates."""
        return bool(self.content) or bool(self.tool_calls)
    
    def to_output(self, buffer: str, transformer: Any) -> LLMOutput:
        """Convert current state to LLMOutput."""
        buffer, output, _ = transformer(self.content, buffer)
        
        # Add tool calls if present
        if self.tool_calls:
            output.tool_calls = self.tool_calls
            
        # Add usage stats if this is final
        if self.finish_reason:
            output.usage = LLMUsage(
                stop_reason=self.usage_stats["stop_reason"],
                time_to_first_token=self.timing_stats["time_to_first_token"],
                tokens_per_second=self.timing_stats["tokens_per_second"],
                prompt_tokens=self.usage_stats["prompt_tokens"],
                completion_tokens=self.usage_stats["completion_tokens"],
                total_tokens=self.usage_stats["total_tokens"],
                reasoning_time=self.timing_stats["reasoning_time"],
                reasoning_tokens=self.timing_stats["reasoning_tokens"]
            )
            
        return output, buffer

class ResponseState:
    """Holds the state of response transformation."""
    def __init__(self):
        self.buffer = ""
        self.response = ""
        self.reasoning = None
        self.function_calls = None  # For future function calling support
        self.tool_calls = []      # List to accumulate tool calls
        self.current_tool_call = None  # Track current tool call being built
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
    def __init__(self, output_cls: type[LLMOutput] = LLMOutput):
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
        return (
            self.state.buffer,
            self.output_cls(
                response=self.state.response.strip(),
                reasoning=self.state.reasoning.strip() if self.state.reasoning else None,
                function_calls=self.state.function_calls,
                tool_calls=self.state.tool_calls
            ),
            self.state.state_changes
        )
    
    def __call__(self, piece: str, buffer: str) -> tuple[str, LLMOutput, dict]:
        """Transform a piece of text and return the result.
        
        Args:
            piece: New piece of text to transform
            buffer: Existing buffer content
            
        Returns:
            Tuple of (new_buffer, output, state_changes)
        """
        self.state.buffer = buffer
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
    max_tokens: int = 4096,
    stop: Optional[List[str]] = None,
    verbose: bool = False,
) -> Generator[LLMOutput, None, None]:
    """Stream generate from LLaMA.cpp model with timing and usage tracking."""
    with timing_context() as timing:
        transformer.timing = timing
        
        # Build completion kwargs
        completion_kwargs = {
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop
        }
        if tools is not None:
            completion_kwargs["tools"] = tools
        if tool_choice is not None:
            completion_kwargs["tool_choice"] = tool_choice
        
        # Initialize response state
        response = StreamResponse()
        buffer = ""
        
        try:
            completion = model.create_chat_completion(**completion_kwargs)
            
            for chunk in completion:
                if verbose:
                    print(chunk)
                # Mark first token time as soon as we get any response
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
                    
        except Exception as e:
            # Ensure any error is properly propagated
            raise e 