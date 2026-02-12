"""Tests for main package initialization."""

import pytest
from unittest.mock import patch


class TestPackageInit:
    """Tests for package initialization."""

    def test_version(self):
        """Test version is defined."""
        from agentswarm import __version__
        assert __version__ == "0.1.0"

    def test_author(self):
        """Test author is defined."""
        from agentswarm import __author__
        assert __author__ == "AgentSwarm Team"

    def test_license(self):
        """Test license is defined."""
        from agentswarm import __license__
        assert __license__ == "MIT"

    def test_imports(self):
        """Test that main classes can be imported."""
        from agentswarm import (
            Swarm,
            Agent,
            Message,
            Task,
        )
        assert Swarm is not None
        assert Agent is not None
        assert Message is not None
        assert Task is not None

    def test_provider_imports(self):
        """Test that provider functions can be imported."""
        from agentswarm import (
            create_provider,
            create_provider_from_env,
            get_available_providers,
            ProviderType,
            ProviderConfig,
        )
        assert create_provider is not None
        assert create_provider_from_env is not None
        assert get_available_providers is not None
        assert ProviderType is not None
        assert ProviderConfig is not None

    def test_all_exports(self):
        """Test that __all__ is properly defined."""
        import agentswarm

        assert hasattr(agentswarm, '__all__')
        assert 'Swarm' in agentswarm.__all__
        assert 'Agent' in agentswarm.__all__
        assert 'Message' in agentswarm.__all__
        assert 'Task' in agentswarm.__all__
        assert 'create_provider' in agentswarm.__all__
