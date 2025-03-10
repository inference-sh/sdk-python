import os
import pytest
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

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

    class TestApp(BaseApp):
        async def run(self, app_input: TestInput) -> TestOutput:
            return TestOutput(result=f"Processed: {app_input.text}")

    app = TestApp()
    with pytest.raises(NotImplementedError):
        app.run(TestInput(text="test")) 