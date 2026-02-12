"""Tests for checkpoint manager module."""

import json
import pytest
import tempfile
from pathlib import Path

from agentswarm.processing.checkpoint_manager import CheckpointManager, Checkpoint


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_creation(self):
        """Test Checkpoint creation."""
        checkpoint = Checkpoint(
            job_id="job-123",
            processed_files=["file1.txt", "file2.txt"],
            total_files=10,
            timestamp=1234567890.0,
        )
        assert checkpoint.job_id == "job-123"
        assert checkpoint.processed_files == ["file1.txt", "file2.txt"]
        assert checkpoint.total_files == 10
        assert checkpoint.timestamp == 1234567890.0

    def test_to_dict(self):
        """Test converting checkpoint to dictionary."""
        checkpoint = Checkpoint(
            job_id="job-123",
            processed_files=["file1.txt"],
            total_files=5,
            timestamp=1234567890.0,
        )

        data = checkpoint.to_dict()

        assert data["job_id"] == "job-123"
        assert data["processed_files"] == ["file1.txt"]
        assert data["total_files"] == 5

    def test_from_dict(self):
        """Test creating checkpoint from dictionary."""
        data = {
            "job_id": "job-123",
            "processed_files": ["file1.txt"],
            "total_files": 5,
            "timestamp": 1234567890.0,
        }

        checkpoint = Checkpoint.from_dict(data)

        assert checkpoint.job_id == "job-123"
        assert checkpoint.processed_files == ["file1.txt"]


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager."""
        return CheckpointManager(checkpoint_dir=temp_dir)

    def test_save_checkpoint(self, manager, temp_dir):
        """Test saving checkpoint."""
        checkpoint = Checkpoint(
            job_id="job-123",
            processed_files=["file1.txt", "file2.txt"],
            total_files=10,
        )

        manager.save_checkpoint(checkpoint)

        checkpoint_file = temp_dir / "job-123.json"
        assert checkpoint_file.exists()

        data = json.loads(checkpoint_file.read_text())
        assert data["job_id"] == "job-123"

    def test_load_checkpoint(self, manager):
        """Test loading checkpoint."""
        checkpoint = Checkpoint(
            job_id="job-123",
            processed_files=["file1.txt"],
            total_files=5,
        )
        manager.save_checkpoint(checkpoint)

        loaded = manager.load_checkpoint("job-123")

        assert loaded is not None
        assert loaded.job_id == "job-123"
        assert loaded.processed_files == ["file1.txt"]

    def test_load_nonexistent_checkpoint(self, manager):
        """Test loading non-existent checkpoint."""
        loaded = manager.load_checkpoint("nonexistent")

        assert loaded is None

    def test_list_checkpoints(self, manager):
        """Test listing checkpoints."""
        checkpoint1 = Checkpoint(job_id="job-1", processed_files=[], total_files=10)
        checkpoint2 = Checkpoint(job_id="job-2", processed_files=[], total_files=20)

        manager.save_checkpoint(checkpoint1)
        manager.save_checkpoint(checkpoint2)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 2
        assert "job-1" in checkpoints
        assert "job-2" in checkpoints

    def test_delete_checkpoint(self, manager, temp_dir):
        """Test deleting checkpoint."""
        checkpoint = Checkpoint(job_id="job-123", processed_files=[], total_files=10)
        manager.save_checkpoint(checkpoint)

        result = manager.delete_checkpoint("job-123")

        assert result is True
        assert not (temp_dir / "job-123.json").exists()

    def test_delete_nonexistent_checkpoint(self, manager):
        """Test deleting non-existent checkpoint."""
        result = manager.delete_checkpoint("nonexistent")

        assert result is False

    def test_get_progress(self, manager):
        """Test getting progress from checkpoint."""
        checkpoint = Checkpoint(
            job_id="job-123",
            processed_files=["file1.txt", "file2.txt"],
            total_files=10,
        )
        manager.save_checkpoint(checkpoint)

        progress = manager.get_progress("job-123")

        assert progress == 20.0  # 2/10 * 100

    def test_get_progress_no_checkpoint(self, manager):
        """Test getting progress without checkpoint."""
        progress = manager.get_progress("nonexistent")

        assert progress == 0.0

    def test_is_completed(self, manager):
        """Test checking if job is completed."""
        checkpoint = Checkpoint(
            job_id="job-123",
            processed_files=["file1.txt", "file2.txt"],
            total_files=2,
        )
        manager.save_checkpoint(checkpoint)

        assert manager.is_completed("job-123") is True

    def test_is_not_completed(self, manager):
        """Test checking if job is not completed."""
        checkpoint = Checkpoint(
            job_id="job-123",
            processed_files=["file1.txt"],
            total_files=2,
        )
        manager.save_checkpoint(checkpoint)

        assert manager.is_completed("job-123") is False

    def test_resume_job(self, manager):
        """Test resuming job from checkpoint."""
        checkpoint = Checkpoint(
            job_id="job-123",
            processed_files=["file1.txt"],
            total_files=10,
        )
        manager.save_checkpoint(checkpoint)

        remaining = manager.get_remaining_files("job-123", ["file1.txt", "file2.txt", "file3.txt"])

        assert remaining == ["file2.txt", "file3.txt"]

    def test_clear_all_checkpoints(self, manager, temp_dir):
        """Test clearing all checkpoints."""
        checkpoint1 = Checkpoint(job_id="job-1", processed_files=[], total_files=10)
        checkpoint2 = Checkpoint(job_id="job-2", processed_files=[], total_files=20)

        manager.save_checkpoint(checkpoint1)
        manager.save_checkpoint(checkpoint2)

        manager.clear_all()

        assert manager.list_checkpoints() == []
