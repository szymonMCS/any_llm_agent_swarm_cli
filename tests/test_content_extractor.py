"""Tests for content extractor module."""

import pytest
import tempfile
from pathlib import Path

from agentswarm.processing.content_extractor import ContentExtractor, ExtractedContent


class TestExtractedContent:
    """Tests for ExtractedContent dataclass."""

    def test_creation(self):
        """Test ExtractedContent creation."""
        content = ExtractedContent(
            text="Hello world",
            path=Path("/test/file.txt"),
            encoding="utf-8",
        )
        assert content.text == "Hello world"
        assert content.path == Path("/test/file.txt")
        assert content.encoding == "utf-8"
        assert content.metadata == {}

    def test_with_metadata(self):
        """Test ExtractedContent with metadata."""
        content = ExtractedContent(
            text="Hello",
            path=Path("/test/file.txt"),
            encoding="utf-8",
            metadata={"lines": 5},
        )
        assert content.metadata == {"lines": 5}


class TestContentExtractor:
    """Tests for ContentExtractor."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_extract_text_file(self, temp_dir):
        """Test extracting text from file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello world", encoding="utf-8")

        extractor = ContentExtractor()
        result = extractor.extract(test_file)

        assert result.text == "Hello world"
        assert result.encoding == "utf-8"
        assert result.path == test_file

    def test_extract_python_file(self, temp_dir):
        """Test extracting Python file content."""
        test_file = temp_dir / "test.py"
        test_file.write_text("def hello():\n    print('world')", encoding="utf-8")

        extractor = ContentExtractor()
        result = extractor.extract(test_file)

        assert "def hello():" in result.text
        assert result.metadata.get("language") == "python"

    def test_extract_json_file(self, temp_dir):
        """Test extracting JSON file content."""
        test_file = temp_dir / "test.json"
        test_file.write_text('{"key": "value"}', encoding="utf-8")

        extractor = ContentExtractor()
        result = extractor.extract(test_file)

        assert '"key": "value"' in result.text

    def test_extract_empty_file(self, temp_dir):
        """Test extracting empty file."""
        test_file = temp_dir / "empty.txt"
        test_file.write_text("")

        extractor = ContentExtractor()
        result = extractor.extract(test_file)

        assert result.text == ""

    def test_extract_binary_file(self, temp_dir):
        """Test extracting binary file."""
        test_file = temp_dir / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        extractor = ContentExtractor()

        with pytest.raises(ValueError):
            extractor.extract(test_file)

    def test_extract_nonexistent_file(self):
        """Test extracting non-existent file."""
        extractor = ContentExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/file.txt")

    def test_get_file_language(self):
        """Test getting language from file extension."""
        extractor = ContentExtractor()

        assert extractor._get_file_language("test.py") == "python"
        assert extractor._get_file_language("test.js") == "javascript"
        assert extractor._get_file_language("test.ts") == "typescript"
        assert extractor._get_file_language("test.java") == "java"
        assert extractor._get_file_language("test.go") == "go"
        assert extractor._get_file_language("test.rs") == "rust"
        assert extractor._get_file_language("test.cpp") == "cpp"
        assert extractor._get_file_language("test.c") == "c"
        assert extractor._get_file_language("test.txt") is None

    def test_is_text_file(self, temp_dir):
        """Test text file detection."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("Hello")

        extractor = ContentExtractor()

        assert extractor._is_text_file(text_file) is True

    def test_is_binary_file(self, temp_dir):
        """Test binary file detection."""
        binary_file = temp_dir / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02")

        extractor = ContentExtractor()

        assert extractor._is_text_file(binary_file) is False
