"""Tests for orchestrator module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.processing.orchestrator import FileProcessor, ProcessingPipeline
from agentswarm.processing.file_scanner import FileInfo
from agentswarm.processing.content_extractor import ExtractedContent


class TestFileProcessor:
    """Tests for FileProcessor."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def processor(self):
        """Create file processor."""
        return FileProcessor(max_workers=2)

    @pytest.mark.asyncio
    async def test_process_single_file(self, processor, temp_dir):
        """Test processing single file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello world")

        async def mock_processor(content):
            return {"processed": True}

        results = []
        async for result in processor.process_file(test_file, mock_processor):
            results.append(result)

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_process_directory(self, processor, temp_dir):
        """Test processing directory."""
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")

        async def mock_processor(content):
            return {"length": len(content.text)}

        results = []
        async for result in processor.process_directory(temp_dir, mock_processor):
            results.append(result)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_process_with_pattern(self, processor, temp_dir):
        """Test processing with file pattern."""
        (temp_dir / "file.txt").write_text("Content")
        (temp_dir / "script.py").write_text("print('hello')")

        async def mock_processor(content):
            return {"processed": True}

        results = []
        async for result in processor.process_directory(
            temp_dir, mock_processor, pattern="*.txt"
        ):
            results.append(result)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_process_with_checkpoint(self, processor, temp_dir):
        """Test processing with checkpoint."""
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")

        async def mock_processor(content):
            return {"processed": True}

        with patch.object(processor.checkpoint_manager, 'save_checkpoint') as mock_save:
            results = []
            async for result in processor.process_directory(
                temp_dir, mock_processor, checkpoint_interval=1
            ):
                results.append(result)

            # Should save checkpoint at least once
            assert mock_save.called


class TestProcessingPipeline:
    """Tests for ProcessingPipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def pipeline(self):
        """Create processing pipeline."""
        return ProcessingPipeline()

    def test_add_stage(self, pipeline):
        """Test adding stage to pipeline."""
        stage = MagicMock()

        pipeline.add_stage("test_stage", stage)

        assert "test_stage" in pipeline.stages

    @pytest.mark.asyncio
    async def test_execute_pipeline(self, pipeline, temp_dir):
        """Test executing pipeline."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello world")

        async def stage1(content):
            content.text = content.text.upper()
            return content

        async def stage2(content):
            content.metadata["processed"] = True
            return content

        pipeline.add_stage("uppercase", stage1)
        pipeline.add_stage("mark", stage2)

        content = ExtractedContent(
            text="Hello world",
            path=test_file,
            encoding="utf-8",
        )

        result = await pipeline.execute(content)

        assert result.text == "HELLO WORLD"
        assert result.metadata.get("processed") is True

    @pytest.mark.asyncio
    async def test_execute_empty_pipeline(self, pipeline, temp_dir):
        """Test executing empty pipeline."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello")

        content = ExtractedContent(
            text="Hello",
            path=test_file,
            encoding="utf-8",
        )

        result = await pipeline.execute(content)

        # Should return content unchanged
        assert result.text == "Hello"
