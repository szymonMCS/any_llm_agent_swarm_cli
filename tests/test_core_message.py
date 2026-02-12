"""Tests for core message module."""

import pytest
from datetime import datetime

from agentswarm.core.message import Message


class TestMessage:
    """Tests for Message class."""

    def test_basic_creation(self):
        """Test basic message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.sender is None
        assert msg.recipient is None
        assert msg.metadata == {}

    def test_full_creation(self):
        """Test message creation with all parameters."""
        msg = Message(
            role="assistant",
            content="Hi there",
            sender="agent1",
            recipient="agent2",
            metadata={"key": "value"},
        )
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
        assert msg.sender == "agent1"
        assert msg.recipient == "agent2"
        assert msg.metadata == {"key": "value"}

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        msg = Message(role="user", content="Hello")
        assert msg.timestamp is not None
        assert isinstance(msg.timestamp, datetime)

    def test_timestamp_custom(self):
        """Test setting custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        msg = Message(role="user", content="Hello", timestamp=custom_time)
        assert msg.timestamp == custom_time

    def test_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(role="user", content="Hello", sender="agent1")
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert data["sender"] == "agent1"
        assert "timestamp" in data

    def test_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "role": "assistant",
            "content": "Hi",
            "sender": "agent1",
            "recipient": "agent2",
            "metadata": {"key": "value"},
        }

        msg = Message.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "Hi"
        assert msg.sender == "agent1"
        assert msg.recipient == "agent2"
        assert msg.metadata == {"key": "value"}

    def test_str_representation(self):
        """Test string representation."""
        msg = Message(role="user", content="Hello")
        assert str(msg) == "[user]: Hello"

    def test_repr(self):
        """Test repr representation."""
        msg = Message(role="user", content="Hello")
        assert "Message" in repr(msg)
        assert "user" in repr(msg)

    def test_system_message_factory(self):
        """Test system message factory method."""
        msg = Message.system("You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"

    def test_user_message_factory(self):
        """Test user message factory method."""
        msg = Message.user("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message_factory(self):
        """Test assistant message factory method."""
        msg = Message.assistant("Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
