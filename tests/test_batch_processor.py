"""Tests for batch processor module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from agentswarm.processing.batch_processor import BatchProcessor, BatchResult
from agentswarm.processing.file_scanner import FileInfo


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_creation(self):
        """Test BatchResult creation."""
        result = BatchResult(
            file_path=Path("/test/file.txt"),
            success=True,
            result="processed",
            error=None,
        )
        assert result.file_path == Path("/test/file.txt")
        assert result.success is True
        assert result.result == "processed"
        assert result.error is None

    def test_failed_result(self):
        """Test failed batch result."""
        result = BatchResult(
            file_path=Path("/test/file.txt"),
            success=False,
            result=None,
            error="Processing failed",
        )
        assert result.success is False
        assert result.error == "Processing failed"


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.fixture
    def processor(self):
        """Create batch processor."""
        return BatchProcessor(max_workers=2)

    @pytest.mark.asyncio
    async def test_process_single_file(self, processor):
        """Test processing single file."""
        async def mock_processor(file_info):
            return f"Processed {file_info.path.name}"

        file_info = FileInfo(
            path=Path("/test/file.txt"),
            size=100,
            modified=1234567890.0,
        )

        results = []
        async for result in processor.process([file_info], mock_processor):
            results.append(result)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].result == "Processed file.txt"

    @pytest.mark.asyncio
    async def test_process_multiple_files(self, processor):
        """Test processing multiple files."""
        async def mock_processor(file_info):
            return f"Processed {file_info.path.name}"

        files = [
            FileInfo(path=Path("/test/file1.txt"), size=100, modified=1.0),
            FileInfo(path=Path("/test/file2.txt"), size=200, modified=2.0),
            FileInfo(path=Path("/test/file3.txt"), size=300, modified=3.0),
        ]

        results = []
        async for result in processor.process(files, mock_processor):
            results.append(result)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_process_with_error(self, processor):
        """Test processing with error."""
        async def mock_processor(file_info):
            if file_info.path.name == "error.txt":
                raise ValueError("Processing error")
            return "Success"

        files = [
            FileInfo(path=Path("/test/ok.txt"), size=100, modified=1.0),
            FileInfo(path=Path("/test/error.txt"), size=100, modified=2.0),
        ]

        results = []
        async for result in processor.process(files, mock_processor):
            results.append(result)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert "Processing error" in results[1].error

    @pytest.mark.asyncio
    async def test_process_with_continue_on_error(self, processor):
        """Test processing continues on error."""
        async def mock_processor(file_info):
            raise ValueError("Error")

        files = [
            FileInfo(path=Path("/test/file1.txt"), size=100, modified=1.0),
            FileInfo(path=Path("/test/file2.txt"), size=100, modified=2.0),
        ]

        results = []
        async for result in processor.process(files, mock_processor, continue_on_error=True):
            results.append(result)

        assert len(results) == 2
        assert all(not r.success for r in results)

    @pytest.mark.asyncio
    async def test_process_without_continue_on_error(self, processor):
        """Test processing stops on error without continue_on_error."""
        async def mock_processor(file_info):
            raise ValueError("Error")

        files = [
            FileInfo(path=Path("/test/file1.txt"), size=100, modified=1.0),
            FileInfo(path=Path("/test/file2.txt"), size=100, modified=2.0),
        ]

        results = []
        with pytest.raises(ValueError):
            async for result in processor.process(files, mock_processor, continue_on_error=False):
                results.append(result)

    @pytest.mark.asyncio
    async def test_process_with_progress_callback(self, processor):
        """Test processing with progress callback."""
        async def mock_processor(file_info):
            return "Success"

        progress_updates = []
        def progress_callback(current, total):
            progress_updates.append((current, total))

        files = [
            FileInfo(path=Path("/test/file1.txt"), size=100, modified=1.0),
            FileInfo(path=Path("/test/file2.txt"), size=100, modified=2.0),
        ]

        results = []
        async for result in processor.process(files, mock_processor, progress_callback=progress_callback):
            results.append(result)

        assert len(progress_updates) == 2
        assert progress_updates[0] == (1, 2)
        assert progress_updates[1] == (2, 2)

    @pytest.mark.asyncio
    async def test_process_empty_list(self, processor):
        """Test processing empty file list."""
        async def mock_processor(file_info):
            return "Success"

        results = []
        async for result in processor.process([], mock_processor):
            results.append(result)

        assert results == []

    @pytest.mark.asyncio
    async def test_process_with_timeout(self, processor):
        """Test processing with timeout."""
        async def slow_processor(file_info):
            await asyncio.sleep(10)  # Will timeout
            return "Success"

        file_info = FileInfo(path=Path("/test/file.txt"), size=100, modified=1.0)

        results = []
        async for result in processor.process([file_info], slow_processor, timeout=0.01):
            results.append(result)

        assert len(results) == 1
        assert results[0].success is False
        assert "timeout" in results[0].error.lower()

    def test_default_max_workers(self):
        """Test default max workers."""
        processor = BatchProcessor()
        assert processor.max_workers > 0

    def test_custom_max_workers(self):
        """Test custom max workers."""
        processor = BatchProcessor(max_workers=5)
        assert processor.max_workers == 5
