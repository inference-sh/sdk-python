"""
Integration tests for inferencesh SDK

These tests hit the real API and require INFERENCE_API_KEY to be set.
Run with: make test-int API_KEY=your-key

Or export INFERENCE_API_KEY and run: make test-int
"""

import os
import pytest
from inferencesh import inference, Inference
from inferencesh.types import TaskStatus

# Get API configuration from environment
API_KEY = os.environ.get("INFERENCE_API_KEY")
BASE_URL = os.environ.get("INFERENCE_BASE_URL", "https://api.inference.sh")

# Use a pinned app version that's known to work
TEST_APP = "infsh/text-templating@53bk0yzk"

# Skip all tests if no API key is set
pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="INFERENCE_API_KEY not set - skipping integration tests"
)


@pytest.fixture
def client():
    """Create a client instance for tests."""
    return Inference(api_key=API_KEY, base_url=BASE_URL)


class TestBasicRun:
    """Tests for basic run functionality."""

    def test_run_and_wait(self, client):
        """Should run a simple task and wait for completion."""
        result = client.run({
            "app": TEST_APP,
            "input": {"template": "Hello {1}!", "strings": ["Pytest"]}
        })

        assert result is not None
        assert result["id"] is not None
        assert result["status"] == TaskStatus.COMPLETED
        assert result.get("output") is not None

    def test_run_fire_and_forget(self, client):
        """Should submit a task without waiting for completion."""
        result = client.run(
            {
                "app": TEST_APP,
                "input": {"template": "{1}", "strings": ["Fire and forget"]}
            },
            wait=False
        )

        assert result is not None
        assert result["id"] is not None
        # Status should NOT be completed yet (task was just submitted)
        assert result["status"] != TaskStatus.COMPLETED
        assert result["status"] != TaskStatus.FAILED


class TestRunWithStream:
    """Tests for run with streaming updates."""

    def test_run_with_stream(self, client):
        """Should receive status updates during task execution."""
        updates = []

        stream = client.run(
            {
                "app": TEST_APP,
                "input": {"template": "Testing {1}", "strings": ["SDK"]}
            },
            stream=True
        )

        for update in stream:
            updates.append(update)

        # Should have received some updates
        assert len(updates) > 0
        # Last update should be completed
        assert updates[-1]["status"] == TaskStatus.COMPLETED


class TestFactoryFunction:
    """Tests for the inference() factory function."""

    def test_factory_function_works(self):
        """Should work with lowercase inference() factory."""
        factory_client = inference(api_key=API_KEY, base_url=BASE_URL)

        result = factory_client.run(
            {
                "app": TEST_APP,
                "input": {"template": "{1}", "strings": ["Factory test"]}
            },
            wait=False
        )

        assert result is not None
        assert result["id"] is not None


class TestErrorHandling:
    """Tests for error handling."""

    def test_error_on_nonexistent_app(self, client):
        """Should raise an error for non-existent app."""
        with pytest.raises(Exception):
            client.run(
                {
                    "app": "non-existent/app-that-does-not-exist@xyz123",
                    "input": {}
                },
                wait=False
            )


class TestAsyncClient:
    """Tests for async client functionality."""

    @pytest.mark.asyncio
    async def test_async_run_and_wait(self):
        """Should run a simple task asynchronously and wait for completion."""
        from inferencesh import AsyncInference

        client = AsyncInference(api_key=API_KEY, base_url=BASE_URL)

        result = await client.run({
            "app": TEST_APP,
            "input": {"template": "Hello {1}!", "strings": ["AsyncPytest"]}
        })

        assert result is not None
        assert result["id"] is not None
        assert result["status"] == TaskStatus.COMPLETED
        assert result.get("output") is not None

    @pytest.mark.asyncio
    async def test_async_fire_and_forget(self):
        """Should submit a task asynchronously without waiting."""
        from inferencesh import AsyncInference

        client = AsyncInference(api_key=API_KEY, base_url=BASE_URL)

        result = await client.run(
            {
                "app": TEST_APP,
                "input": {"template": "{1}", "strings": ["Async fire and forget"]}
            },
            wait=False
        )

        assert result is not None
        assert result["id"] is not None
        assert result["status"] != TaskStatus.COMPLETED


# This always runs to ensure pytest doesn't complain about no tests
class TestIntegrationSetup:
    """Basic setup verification."""

    def test_api_key_check(self):
        """Verify test setup is working."""
        if not API_KEY:
            print("⚠️  Skipping integration tests - INFERENCE_API_KEY not set")
        assert True
