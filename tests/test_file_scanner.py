"""Tests for file scanner module."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentswarm.processing.file_scanner import FileScanner, FileInfo


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_creation(self):
        """Test FileInfo creation."""
        info = FileInfo(
            path=Path("/test/file.txt"),
            size=100,
            modified=1234567890.0,
            encoding="utf-8",
        )
        assert info.path == Path("/test/file.txt")
        assert info.size == 100
        assert info.modified == 1234567890.0
        assert info.encoding == "utf-8"

    def test_optional_fields(self):
        """Test FileInfo with optional fields."""
        info = FileInfo(
            path=Path("/test/file.txt"),
            size=100,
            modified=1234567890.0,
        )
        assert info.encoding is None


class TestFileScanner:
    """Tests for FileScanner."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_scan_empty_directory(self, temp_dir):
        """Test scanning empty directory."""
        scanner = FileScanner()
        files = scanner.scan(temp_dir)
        assert files == []

    def test_scan_single_file(self, temp_dir):
        """Test scanning directory with single file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello world")

        scanner = FileScanner()
        files = scanner.scan(temp_dir)

        assert len(files) == 1
        assert files[0].path == test_file
        assert files[0].size == 11

    def test_scan_with_pattern(self, temp_dir):
        """Test scanning with file pattern."""
        (temp_dir / "file1.txt").write_text("content")
        (temp_dir / "file2.py").write_text("content")
        (temp_dir / "file3.txt").write_text("content")

        scanner = FileScanner()
        files = scanner.scan(temp_dir, pattern="*.txt")

        assert len(files) == 2
        assert all(f.path.suffix == ".txt" for f in files)

    def test_scan_recursive(self, temp_dir):
        """Test recursive scanning."""
        (temp_dir / "root.txt").write_text("content")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "sub.txt").write_text("content")

        scanner = FileScanner()
        files = scanner.scan(temp_dir, recursive=True)

        assert len(files) == 2

    def test_scan_non_recursive(self, temp_dir):
        """Test non-recursive scanning."""
        (temp_dir / "root.txt").write_text("content")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "sub.txt").write_text("content")

        scanner = FileScanner()
        files = scanner.scan(temp_dir, recursive=False)

        assert len(files) == 1

    def test_scan_with_excludes(self, temp_dir):
        """Test scanning with exclude patterns."""
        (temp_dir / "include.txt").write_text("content")
        (temp_dir / "exclude.tmp").write_text("content")

        scanner = FileScanner()
        files = scanner.scan(temp_dir, excludes=["*.tmp"])

        assert len(files) == 1
        assert files[0].path.name == "include.txt"

    def test_detect_encoding_utf8(self, temp_dir):
        """Test UTF-8 encoding detection."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello world", encoding="utf-8")

        scanner = FileScanner()
        encoding = scanner._detect_encoding(test_file)

        assert encoding == "utf-8"

    def test_detect_encoding_latin1(self, temp_dir):
        """Test Latin-1 encoding detection."""
        test_file = temp_dir / "test.txt"
        test_file.write_bytes(b"Hello world \xff")

        scanner = FileScanner()
        encoding = scanner._detect_encoding(test_file)

        assert encoding is not None

    def test_is_binary_file(self, temp_dir):
        """Test binary file detection."""
        binary_file = temp_dir / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        scanner = FileScanner()
        is_binary = scanner._is_binary(binary_file)

        assert is_binary is True

    def test_is_text_file(self, temp_dir):
        """Test text file detection."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("Hello world")

        scanner = FileScanner()
        is_binary = scanner._is_binary(text_file)

        assert is_binary is False

    def test_scan_skips_binary(self, temp_dir):
        """Test that binary files are skipped by default."""
        (temp_dir / "text.txt").write_text("content")
        (temp_dir / "binary.bin").write_bytes(b"\x00\x01\x02")

        scanner = FileScanner()
        files = scanner.scan(temp_dir)

        assert len(files) == 1
        assert files[0].path.name == "text.txt"

    def test_scan_includes_binary(self, temp_dir):
        """Test including binary files."""
        (temp_dir / "text.txt").write_text("content")
        (temp_dir / "binary.bin").write_bytes(b"\x00\x01\x02")

        scanner = FileScanner()
        files = scanner.scan(temp_dir, include_binary=True)

        assert len(files) == 2

    def test_scan_nonexistent_directory(self):
        """Test scanning non-existent directory."""
        scanner = FileScanner()

        with pytest.raises(FileNotFoundError):
            scanner.scan("/nonexistent/path")

    def test_scan_file_instead_of_directory(self, temp_dir):
        """Test scanning a file instead of directory."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        scanner = FileScanner()

        with pytest.raises(NotADirectoryError):
            scanner.scan(test_file)

    def test_get_stats(self, temp_dir):
        """Test getting scan statistics."""
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2 longer")

        scanner = FileScanner()
        files = scanner.scan(temp_dir)
        stats = scanner.get_stats(files)

        assert stats["total_files"] == 2
        assert stats["total_size"] == 21  # 8 + 15 bytes
        assert stats["avg_size"] == 10.5
