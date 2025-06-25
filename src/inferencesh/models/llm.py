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
    enable_thinking: bool = Field(default=False)

class LLMUsage(BaseAppOutput):
    stop_reason: str = ""
    time_to_first_token: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMOutput(BaseAppOutput):
    response: str
    thinking_content: Optional[str] = None
    usage: Optional[LLMUsage] = None


@contextmanager
def timing_context():
    """Context manager to track timing information for LLM generation."""
    class TimingInfo:
        def __init__(self):
            self.start_time = time.time()
            self.first_token_time = None
        
        def mark_first_token(self):
            if self.first_token_time is None:
                self.first_token_time = time.time()
        
        @property
        def stats(self):
            end_time = time.time()
            if self.first_token_time is None:
                self.first_token_time = end_time
            
            time_to_first = self.first_token_time - self.start_time
            generation_time = end_time - self.first_token_time
            
            return {
                "time_to_first_token": time_to_first,
                "generation_time": generation_time
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
    
    Args:
        input_data: The input data
        transform_user_message: Optional function to transform user message text before building messages
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": input_data.system_prompt}],
        }
    ]

    # Add context messages
    for msg in input_data.context:
        message_content = []
        text = msg.text
        if transform_user_message and msg.role == ContextMessageRole.USER:
            text = transform_user_message(text)
        if text:
            message_content.append({"type": "text", "text": text})
        if hasattr(msg, 'image') and msg.image:
            if msg.image.path:
                image_data_uri = image_to_base64_data_uri(msg.image.path)
                message_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
            elif msg.image.uri:
                message_content.append({"type": "image_url", "image_url": {"url": msg.image.uri}})
        messages.append({
            "role": msg.role,
            "content": message_content
        })

    # Add user message
    user_content = []
    text = input_data.text
    if transform_user_message:
        text = transform_user_message(text)
    if text:
        user_content.append({"type": "text", "text": text})
    if hasattr(input_data, 'image') and input_data.image:
        if input_data.image.path:
            image_data_uri = image_to_base64_data_uri(input_data.image.path)
            user_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
        elif input_data.image.uri:
            user_content.append({"type": "image_url", "image_url": {"url": input_data.image.uri}})
    messages.append({"role": "user", "content": user_content})

    return messages


def stream_generate(
    model: Any,
    messages: List[Dict[str, Any]],
    output_cls: type[LLMOutput],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    stop: Optional[List[str]] = None,
    handle_thinking: bool = False,
    transform_response: Optional[Callable[[str, str], tuple[str, LLMOutput]]] = None,
) -> Generator[LLMOutput, None, None]:
    """Stream generate from LLaMA.cpp model with timing and usage tracking.
    
    Args:
        model: The LLaMA.cpp model instance
        messages: List of messages to send to the model
        output_cls: Output class type to use for responses
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        max_tokens: Maximum tokens to generate
        stop: Optional list of stop sequences
        handle_thinking: Whether to handle thinking tags
        transform_response: Optional function to transform responses, takes (piece, buffer) and returns (new_buffer, output)
    """
    response_queue: Queue[Optional[tuple[str, dict]]] = Queue()
    thread_exception = None
    usage_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "stop_reason": ""
    }

    with timing_context() as timing:
        def generation_thread():
            nonlocal thread_exception, usage_stats
            try:
                completion = model.create_chat_completion(
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop
                )
                
                for chunk in completion:
                    # Get usage from root level if present
                    if "usage" in chunk and chunk["usage"] is not None:
                        usage_stats.update(chunk["usage"])
                    
                    # Get content from choices
                    delta = chunk.get("choices", [{}])[0]
                    content = None
                    finish_reason = None
                    
                    if "message" in delta:
                        content = delta["message"].get("content", "")
                        finish_reason = delta.get("finish_reason")
                    elif "delta" in delta:
                        content = delta["delta"].get("content", "")
                        finish_reason = delta.get("finish_reason")
                    
                    if content:
                        if not timing.first_token_time:
                            timing.mark_first_token()
                        response_queue.put((content, {}))
                        
                    if finish_reason:
                        usage_stats["stop_reason"] = finish_reason
                
            except Exception as e:
                thread_exception = e
            finally:
                timing_stats = timing.stats
                generation_time = timing_stats["generation_time"]
                tokens_per_second = (usage_stats["completion_tokens"] / generation_time) if generation_time > 0 else 0
                response_queue.put((None, {
                    "time_to_first_token": timing_stats["time_to_first_token"],
                    "tokens_per_second": tokens_per_second
                }))

        thread = Thread(target=generation_thread, daemon=True)
        thread.start()

        buffer = ""
        thinking_content = "" if handle_thinking else None
        in_thinking = handle_thinking
        try:
            while True:
                try:
                    result = response_queue.get(timeout=30.0)
                    if thread_exception:
                        raise thread_exception
                    
                    piece, timing_stats = result
                    if piece is None:
                        # Final yield with complete usage stats
                        usage = LLMUsage(
                            stop_reason=usage_stats["stop_reason"],
                            time_to_first_token=timing_stats["time_to_first_token"],
                            tokens_per_second=timing_stats["tokens_per_second"],
                            prompt_tokens=usage_stats["prompt_tokens"],
                            completion_tokens=usage_stats["completion_tokens"],
                            total_tokens=usage_stats["total_tokens"]
                        )
                        
                        if transform_response:
                            buffer, output = transform_response(piece or "", buffer)
                            output.usage = usage
                            yield output
                        else:
                            # Handle thinking vs response content if enabled
                            if handle_thinking and "</think>" in piece:
                                parts = piece.split("</think>")
                                if in_thinking:
                                    thinking_content += parts[0].replace("<think>", "")
                                    buffer = parts[1] if len(parts) > 1 else ""
                                    in_thinking = False
                                else:
                                    buffer += piece
                            else:
                                if in_thinking:
                                    thinking_content += piece.replace("<think>", "")
                                else:
                                    buffer += piece
                                    
                            yield output_cls(
                                response=buffer.strip(),
                                thinking_content=thinking_content.strip() if thinking_content else None,
                                usage=usage
                            )
                        break

                    if transform_response:
                        buffer, output = transform_response(piece, buffer)
                        yield output
                    else:
                        # Handle thinking vs response content if enabled
                        if handle_thinking and "</think>" in piece:
                            parts = piece.split("</think>")
                            if in_thinking:
                                thinking_content += parts[0].replace("<think>", "")
                                buffer = parts[1] if len(parts) > 1 else ""
                                in_thinking = False
                            else:
                                buffer += piece
                        else:
                            if in_thinking:
                                thinking_content += piece.replace("<think>", "")
                            else:
                                buffer += piece

                        yield output_cls(
                            response=buffer.strip(),
                            thinking_content=thinking_content.strip() if thinking_content else None
                        )
                    
                except Exception as e:
                    if thread_exception and isinstance(e, thread_exception.__class__):
                        raise thread_exception
                    break
        finally:
            if thread and thread.is_alive():
                thread.join(timeout=2.0) 