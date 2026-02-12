"""Message class for AgentSwarm.

This module provides the Message class for inter-agent communication.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Message for inter-agent communication.
    
    Messages are the primary means of communication between agents
    in a swarm. They support various message types and can carry
    arbitrary payload data.
    
    Attributes:
        id: Unique identifier for the message.
        sender: ID or name of the sending agent.
        recipient: ID or name of the receiving agent (None for broadcast).
        content: The message content.
        message_type: Type of the message.
        timestamp: When the message was created.
        metadata: Additional metadata for the message.
    
    Example:
        >>> message = Message(
        ...     sender="agent1",
        ...     recipient="agent2",
        ...     content="Hello!",
        ...     message_type="chat"
        ... )
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str = Field(..., description="ID or name of the sending agent")
    recipient: Optional[str] = Field(default=None, description="ID or name of the receiving agent (None for broadcast)")
    content: Any = Field(..., description="Message content")
    message_type: str = Field(default="text", description="Type of message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message.
        
        Returns:
            True if the message is a broadcast, False otherwise.
        """
        return self.recipient is None
    
    def reply(self, content: Any, message_type: Optional[str] = None) -> "Message":
        """Create a reply message.
        
        Args:
            content: Content of the reply.
            message_type: Type of the reply message.
        
        Returns:
            A new Message instance configured as a reply.
        """
        return Message(
            sender=self.recipient or "unknown",
            recipient=self.sender,
            content=content,
            message_type=message_type or self.message_type,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.
        
        Returns:
            Dictionary representation of the message.
        """
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        """String representation of the message.
        
        Returns:
            String representation.
        """
        return f"Message(from='{self.sender}', to='{self.recipient}', type='{self.message_type}')"
