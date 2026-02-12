"""Tests for utils logging module."""

import logging
import pytest
from unittest.mock import patch, MagicMock

from agentswarm.utils.logging import setup_logging, get_logger


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_default_setup(self):
        """Test default logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_get_logger.return_value = mock_root

            setup_logging()

            mock_get_logger.assert_called_with()
            mock_root.setLevel.assert_called_with(logging.INFO)

    def test_debug_level(self):
        """Test setup with DEBUG level."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_get_logger.return_value = mock_root

            setup_logging(log_level="DEBUG")

            mock_root.setLevel.assert_called_with(logging.DEBUG)

    def test_warning_level(self):
        """Test setup with WARNING level."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_get_logger.return_value = mock_root

            setup_logging(log_level="WARNING")

            mock_root.setLevel.assert_called_with(logging.WARNING)

    def test_invalid_level(self):
        """Test setup with invalid log level."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_get_logger.return_value = mock_root

            setup_logging(log_level="INVALID")

            # Should default to INFO
            mock_root.setLevel.assert_called_with(logging.INFO)

    def test_handler_added(self):
        """Test that handler is added."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_root.handlers = []
            mock_get_logger.return_value = mock_root

            setup_logging()

            assert mock_root.addHandler.called

    def test_existing_handlers_not_duplicated(self):
        """Test that existing handlers are not duplicated."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_root.handlers = [MagicMock()]  # Existing handler
            mock_get_logger.return_value = mock_root

            setup_logging()

            # Should not add another handler
            assert not mock_root.addHandler.called


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_name(self):
        """Test getting logger with name."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            logger = get_logger("test_logger")

            mock_get_logger.assert_called_with("test_logger")
            assert logger == mock_logger

    def test_get_logger_default_name(self):
        """Test getting logger with default name."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            logger = get_logger()

            mock_get_logger.assert_called_with("agentswarm")
            assert logger == mock_logger

    def test_logger_has_methods(self):
        """Test that logger has standard logging methods."""
        logger = get_logger("test")

        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')
