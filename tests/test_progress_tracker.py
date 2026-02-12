"""Tests for progress tracker module."""

import pytest
import time
from unittest.mock import MagicMock

from agentswarm.processing.progress_tracker import (
    ProgressTracker,
    ProgressStats,
    ConsoleProgressReporter,
)


class TestProgressStats:
    """Tests for ProgressStats dataclass."""

    def test_creation(self):
        """Test ProgressStats creation."""
        stats = ProgressStats(
            total_files=100,
            processed_files=50,
            failed_files=5,
            start_time=1234567890.0,
        )
        assert stats.total_files == 100
        assert stats.processed_files == 50
        assert stats.failed_files == 5
        assert stats.start_time == 1234567890.0

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        stats = ProgressStats(
            total_files=100,
            processed_files=50,
            failed_files=0,
        )
        assert stats.progress_percentage == 50.0

    def test_progress_percentage_zero_total(self):
        """Test progress percentage with zero total."""
        stats = ProgressStats(
            total_files=0,
            processed_files=0,
        )
        assert stats.progress_percentage == 0.0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        start = time.time()
        stats = ProgressStats(
            total_files=100,
            processed_files=50,
            start_time=start,
        )

        elapsed = stats.elapsed_time
        assert elapsed >= 0

    def test_estimated_time_remaining(self):
        """Test estimated time remaining calculation."""
        start = time.time() - 10  # Started 10 seconds ago
        stats = ProgressStats(
            total_files=100,
            processed_files=50,
            start_time=start,
        )

        eta = stats.estimated_time_remaining
        # At 50% after 10 seconds, ETA should be around 10 seconds
        assert eta is not None
        assert eta > 0

    def test_estimated_time_remaining_zero_progress(self):
        """Test ETA with zero progress."""
        stats = ProgressStats(
            total_files=100,
            processed_files=0,
        )

        assert stats.estimated_time_remaining is None

    def test_processing_rate(self):
        """Test processing rate calculation."""
        start = time.time() - 10  # 10 seconds ago
        stats = ProgressStats(
            total_files=100,
            processed_files=50,
            start_time=start,
        )

        rate = stats.processing_rate
        # 50 files in 10 seconds = 5 files/second
        assert rate is not None
        assert rate > 0


class TestProgressTracker:
    """Tests for ProgressTracker."""

    @pytest.fixture
    def tracker(self):
        """Create progress tracker."""
        return ProgressTracker(total_files=100)

    def test_initial_state(self, tracker):
        """Test initial tracker state."""
        assert tracker.total_files == 100
        assert tracker.processed_files == 0
        assert tracker.failed_files == 0

    def test_update_progress(self, tracker):
        """Test updating progress."""
        tracker.update_processed(10)

        assert tracker.processed_files == 10

    def test_update_failed(self, tracker):
        """Test updating failed count."""
        tracker.update_failed(5)

        assert tracker.failed_files == 5

    def test_get_stats(self, tracker):
        """Test getting progress stats."""
        tracker.update_processed(50)
        tracker.update_failed(5)

        stats = tracker.get_stats()

        assert stats.total_files == 100
        assert stats.processed_files == 50
        assert stats.failed_files == 5

    def test_is_complete(self, tracker):
        """Test checking if processing is complete."""
        tracker.update_processed(100)

        assert tracker.is_complete() is True

    def test_is_not_complete(self, tracker):
        """Test checking if processing is not complete."""
        tracker.update_processed(50)

        assert tracker.is_complete() is False

    def test_reset(self, tracker):
        """Test resetting tracker."""
        tracker.update_processed(50)
        tracker.update_failed(5)

        tracker.reset()

        assert tracker.processed_files == 0
        assert tracker.failed_files == 0

    def test_callback_on_update(self):
        """Test callback on progress update."""
        callback_called = False
        callback_stats = None

        def callback(stats):
            nonlocal callback_called, callback_stats
            callback_called = True
            callback_stats = stats

        tracker = ProgressTracker(total_files=100, callback=callback)
        tracker.update_processed(10)

        assert callback_called is True
        assert callback_stats is not None
        assert callback_stats.processed_files == 10


class TestConsoleProgressReporter:
    """Tests for ConsoleProgressReporter."""

    def test_report_progress(self, capsys):
        """Test reporting progress to console."""
        reporter = ConsoleProgressReporter()
        stats = ProgressStats(
            total_files=100,
            processed_files=50,
            failed_files=5,
        )

        reporter.report(stats)

        captured = capsys.readouterr()
        assert "50/100" in captured.out or "50%" in captured.out

    def test_report_completion(self, capsys):
        """Test reporting completion."""
        reporter = ConsoleProgressReporter()
        stats = ProgressStats(
            total_files=100,
            processed_files=100,
            failed_files=0,
        )

        reporter.report(stats)

        captured = capsys.readouterr()
        assert "100" in captured.out or "Complete" in captured.out
