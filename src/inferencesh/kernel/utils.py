"""Utility functions for inference.sh kernel operations."""

from typing import Any, AsyncIterator, Tuple


async def aislast(ait: AsyncIterator[Any]) -> AsyncIterator[Tuple[bool, Any]]:
    """Async iterator wrapper that yields (is_last, item) tuples.
    
    This is useful for knowing when you're processing the last item
    in an async iterator, commonly used to distinguish between
    progress updates and final results.
    
    Args:
        ait: The async iterator to wrap
        
    Yields:
        Tuple of (is_last: bool, item: Any) where is_last is True
        only for the final item
        
    Example:
        ```python
        async for is_last, output in aislast(app.run(input)):
            if is_last:
                # This is the final output
                save_result(output)
            else:
                # This is a progress update
                report_progress(output)
        ```
    """
    it = ait.__aiter__()
    try:
        e = await it.__anext__()
        while True:
            try:
                nxt = await it.__anext__()
                yield False, e
                e = nxt
            except StopAsyncIteration:
                yield True, e
                break
    except StopAsyncIteration:
        return

