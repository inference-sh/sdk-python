import os
import pytest
import tempfile
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import urllib.parse

def test_file_creation():
    # Create a temporary file
    with open("test.txt", "w") as f:
        f.write("test")
    
    file = File(path="test.txt")
    assert file.exists()
    assert file.size > 0
    assert file.content_type is not None
    assert file.filename == "test.txt"
    
    os.remove("test.txt")

def test_base_app():
    class TestInput(BaseAppInput):
        text: str

    class TestOutput(BaseAppOutput):
        result: str

    # Use BaseApp directly, don't subclass with implementation
    app = BaseApp()
    import asyncio
    with pytest.raises(NotImplementedError):
        asyncio.run(app.run(TestInput(text="test")))

def test_file_from_local_path():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"test content")
        path = f.name
    
    try:
        # Test creating File from path
        file = File(uri=path)
        assert file.exists()
        assert file.size == len("test content")
        assert file.content_type == "text/plain"
        assert file.filename == os.path.basename(path)
        assert file.path == os.path.abspath(path)
        assert file._tmp_path is None  # Should not create temp file for local paths
    finally:
        os.unlink(path)

def test_file_from_relative_path():
    # Create a file in current directory
    with open("test_relative.txt", "w") as f:
        f.write("relative test")
    
    try:
        file = File(uri="test_relative.txt")
        assert file.exists()
        assert os.path.isabs(file.path)
        assert file.filename == "test_relative.txt"
    finally:
        os.unlink("test_relative.txt")

def test_file_validation():
    # Test empty initialization
    with pytest.raises(ValueError, match="Either 'uri' or 'path' must be provided"):
        File()
    
    # Test invalid input type
    with pytest.raises(ValueError, match="Invalid input for File"):
        File(123)
    
    # Test string input (should work)
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"test content")
        path = f.name
        
    try:
        file = File(path)
        assert isinstance(file, File)
        assert file.uri == path
        assert file.exists()
    finally:
        os.unlink(path)

def test_file_from_url(monkeypatch):
    # Mock URL download
    def mock_urlopen(request):
        class MockResponse:
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
            
            def read(self):
                return b"mocked content"
        
        return MockResponse()
    
    monkeypatch.setattr(urllib.request, 'urlopen', mock_urlopen)
    
    url = "https://example.com/test.pdf"
    file = File(uri=url)
    
    try:
        assert file._is_url(url)
        assert file.exists()
        assert file._tmp_path is not None
        assert file._tmp_path.endswith('.pdf')  # Just check the extension
        assert file.content_type == "application/pdf"
    finally:
        # Cleanup should happen in __del__ but let's be explicit for testing
        if file._tmp_path and os.path.exists(file._tmp_path):
            os.unlink(file._tmp_path)

def test_file_metadata_refresh():
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        initial_content = b'{"test": "data"}'
        f.write(initial_content)
        path = f.name
    
    try:
        file = File(uri=path)
        initial_size = file.size
        
        # Modify file with significantly more data
        with open(path, 'ab') as f:  # Open in append binary mode
            additional_data = b'\n{"more": "data"}\n' * 10  # Add multiple lines of data
            f.write(additional_data)
        
        # Refresh metadata
        file.refresh_metadata()
        assert file.size > initial_size, f"New size {file.size} should be larger than initial size {initial_size}"
    finally:
        os.unlink(path)

def test_file_cleanup(monkeypatch):
    # Mock URL download - same mock as test_file_from_url
    def mock_urlopen(request):
        class MockResponse:
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
            
            def read(self):
                return b"mocked content"
        
        return MockResponse()
    
    monkeypatch.setattr(urllib.request, 'urlopen', mock_urlopen)
    
    url = "https://example.com/test.txt"
    file = File(uri=url)
    
    if file._tmp_path:
        tmp_path = file._tmp_path
        assert os.path.exists(tmp_path)
        del file
        assert not os.path.exists(tmp_path) 
        
def test_file_schema():
    file = File(uri="https://example.com/test.txt")
    print(file.model_json_schema())
    assert file.model_json_schema() is not None
    assert file.model_json_schema()["$id"] == "/schemas/File"
    assert file.model_json_schema()["oneOf"] is not None
    assert file.model_json_schema()["oneOf"][0] is not None
    assert file.model_json_schema()["oneOf"][0]["type"] == "object"
    assert file.model_json_schema()["oneOf"][0]["properties"] is not None
    assert file.model_json_schema()["oneOf"][0]["properties"]["uri"] is not None
    assert file.model_json_schema()["oneOf"][0]["properties"]["path"] is not None
    assert file.model_json_schema()["oneOf"][0]["properties"]["content_type"] is not None
    assert file.model_json_schema()["oneOf"][0]["properties"]["size"] is not None
    assert file.model_json_schema()["oneOf"][0]["properties"]["filename"] is not None