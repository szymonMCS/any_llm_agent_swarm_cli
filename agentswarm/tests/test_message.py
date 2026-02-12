"""Tests for the Message class."""

import pytest

from agentswarm import Message


class TestMessage:
    """Test cases for the Message class."""
    
    def test_message_creation(self) -> None:
        """Test basic message creation."""
        message = Message(
            sender="agent1",
            recipient="agent2",
            content="Hello!",
            message_type="chat"
        )
        
        assert message.sender == "agent1"
        assert message.recipient == "agent2"
        assert message.content == "Hello!"
        assert message.message_type == "chat"
    
    def test_broadcast_message(self) -> None:
        """Test broadcast message creation."""
        message = Message(
            sender="agent1",
            content="Broadcast!",
            recipient=None
        )
        
        assert message.is_broadcast() is True
    
    def test_direct_message(self) -> None:
        """Test direct message (not broadcast)."""
        message = Message(
            sender="agent1",
            recipient="agent2",
            content="Hello!"
        )
        
        assert message.is_broadcast() is False
    
    def test_message_reply(self) -> None:
        """Test creating a reply message."""
        original = Message(
            sender="agent1",
            recipient="agent2",
            content="Hello!"
        )
        
        reply = original.reply("Hi there!")
        
        assert reply.sender == "agent2"
        assert reply.recipient == "agent1"
        assert reply.content == "Hi there!"
    
    def test_message_to_dict(self) -> None:
        """Test message serialization to dict."""
        message = Message(
            sender="agent1",
            recipient="agent2",
            content="Hello!"
        )
        data = message.to_dict()
        
        assert data["sender"] == "agent1"
        assert data["recipient"] == "agent2"
        assert data["content"] == "Hello!"
    
    def test_message_repr(self) -> None:
        """Test message string representation."""
        message = Message(
            sender="agent1",
            recipient="agent2",
            content="Hello!"
        )
        repr_str = repr(message)
        
        assert "agent1" in repr_str
        assert "agent2" in repr_str
