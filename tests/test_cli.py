"""Tests for CLI module."""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from io import StringIO

from agentswarm.cli.main import main


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_version_flag(self, capsys):
        """Test --version flag."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, 'argv', ['agentswarm', '--version']):
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "agentswarm" in captured.out.lower()

    def test_help_flag(self, capsys):
        """Test --help flag."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, 'argv', ['agentswarm', '--help']):
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "AgentSwarm" in captured.out

    def test_init_command(self, temp_dir, capsys):
        """Test init command."""
        os.chdir(temp_dir)

        with patch.object(sys, 'argv', ['agentswarm', 'init', 'test-project']):
            main()

        project_path = temp_dir / "test-project"
        assert project_path.exists()
        assert (project_path / "agents").exists()
        assert (project_path / "configs").exists()
        assert (project_path / "data").exists()
        assert (project_path / "output").exists()
        assert (project_path / "README.md").exists()
        assert (project_path / ".env").exists()

    def test_init_duplicate(self, temp_dir, capsys):
        """Test init with duplicate project name."""
        os.chdir(temp_dir)

        # Create project first
        with patch.object(sys, 'argv', ['agentswarm', 'init', 'test-project']):
            main()

        # Try to create again
        with pytest.raises(SystemExit):
            with patch.object(sys, 'argv', ['agentswarm', 'init', 'test-project']):
                main()

    def test_config_list_command(self, capsys):
        """Test config list command."""
        with patch.object(sys, 'argv', ['agentswarm', 'config', 'list']):
            with patch.dict(os.environ, {}, clear=True):
                main()

        captured = capsys.readouterr()
        assert "OpenAI" in captured.out
        assert "Anthropic" in captured.out

    def test_config_set_command(self, temp_dir, capsys):
        """Test config set command."""
        os.chdir(temp_dir)

        with patch.object(sys, 'argv', ['agentswarm', 'config', 'set', 'openai', '-k', 'test-key']):
            main()

        env_file = temp_dir / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "OPENAI_API_KEY=test-key" in content

    def test_config_set_unknown_provider(self, capsys):
        """Test config set with unknown provider."""
        with pytest.raises(SystemExit):
            with patch.object(sys, 'argv', ['agentswarm', 'config', 'set', 'unknown']):
                main()

    def test_providers_command(self, capsys):
        """Test providers command."""
        with patch.object(sys, 'argv', ['agentswarm', 'providers']):
            main()

        captured = capsys.readouterr()
        assert "OpenAI" in captured.out
        assert "Anthropic" in captured.out
        assert "Google" in captured.out
        assert "GPT-4" in captured.out
        assert "Claude" in captured.out

    def test_run_command(self, temp_dir, capsys):
        """Test run command."""
        os.chdir(temp_dir)

        with patch.object(sys, 'argv', ['agentswarm', 'run', 'Test prompt']):
            main()

        captured = capsys.readouterr()
        assert "Running" in captured.out
        assert "Test prompt" in captured.out

    def test_run_with_options(self, temp_dir, capsys):
        """Test run command with options."""
        os.chdir(temp_dir)

        with patch.object(sys, 'argv', [
            'agentswarm', 'run', 'Test',
            '--provider', 'anthropic',
            '--workers', '10'
        ]):
            main()

        captured = capsys.readouterr()
        assert "anthropic" in captured.out

    @patch('agentswarm.cli.main.create_provider')
    @patch('agentswarm.cli.main.asyncio.run')
    def test_config_test_command(self, mock_run, mock_create, capsys):
        """Test config test command."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=MagicMock(
            text="OK",
            model="gpt-4",
        ))
        mock_create.return_value = mock_provider

        with patch.object(sys, 'argv', ['agentswarm', 'config', 'test', 'openai']):
            main()

        mock_create.assert_called_once_with('openai')
        mock_run.assert_called_once()

    def test_no_command(self, capsys):
        """Test running without command."""
        with pytest.raises(SystemExit):
            with patch.object(sys, 'argv', ['agentswarm']):
                main()

        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
