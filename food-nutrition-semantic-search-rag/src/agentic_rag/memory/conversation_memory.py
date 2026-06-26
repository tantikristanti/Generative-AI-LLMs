"""
Conversation memory module for agentic RAG.

Stores chat history, manages context length, and provides utilities
for retrieving past interactions.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """A single message in the conversation."""
    role: str          # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

class ConversationMemory:
    """
    Manages conversation history for an agent.
    Supports:
    - Adding messages (user, assistant, tool, system)
    - Retrieving recent messages
    - Truncating to a maximum number of messages or token count (approximate)
    - Clearing history
    - Formatting history as a prompt context
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: Optional[int] = None,
        include_timestamps: bool = False,
    ):
        """
        Args:
            max_messages: Maximum number of messages to keep (oldest are dropped).
            max_tokens: Approximate token limit (if set, messages are truncated
                        from the beginning to stay within limit). Token counting
                        is approximate (4 chars ~ 1 token).
            include_timestamps: Whether to include timestamps in formatted output.
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.include_timestamps = include_timestamps
        self._messages: List[Message] = []

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a new message to the history."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)

        # Apply limits
        self._truncate()

    def add_user_message(self, content: str) -> None:
        """Convenience method for user messages."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Convenience method for assistant messages."""
        self.add_message("assistant", content)

    def add_system_message(self, content: str) -> None:
        """Convenience method for system messages."""
        self.add_message("system", content)

    def add_tool_message(self, content: str, tool_name: str = "tool") -> None:
        """Convenience method for tool responses."""
        self.add_message("tool", content, metadata={"tool": tool_name})

    def get_messages(self, last_n: Optional[int] = None) -> List[Message]:
        """Return the last N messages (or all if None)."""
        if last_n is None:
            return self._messages.copy()
        return self._messages[-last_n:]

    def get_last_user_message(self) -> Optional[str]:
        """Return the content of the most recent user message."""
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_conversation_context(
        self,
        last_n: Optional[int] = None,
        include_roles: bool = True,
    ) -> str:
        """
        Format the conversation history as a text block for prompt inclusion.

        Args:
            last_n: Only include the last N messages.
            include_roles: Prepend role labels (e.g., "User: ...").

        Returns:
            A string representing the conversation.
        """
        messages = self.get_messages(last_n)
        parts = []
        for msg in messages:
            prefix = f"[{msg.role}] " if include_roles else ""
            if self.include_timestamps:
                ts = msg.timestamp.strftime("%H:%M:%S")
                prefix = f"[{ts}] {prefix}"
            parts.append(f"{prefix}{msg.content}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def _truncate(self) -> None:
        """Enforce max_messages and max_tokens limits."""
        # Limit by count
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

        # Limit by approximate token count (if set)
        if self.max_tokens is not None:
            # Approximate tokens: 1 token ≈ 4 characters (roughly)
            # We'll truncate from the oldest until we're under the limit.
            # Keep at least the most recent message.
            while len(self._messages) > 1:
                total_chars = sum(len(m.content) for m in self._messages)
                if total_chars // 4 <= self.max_tokens:
                    break
                # Remove oldest (index 0)
                self._messages.pop(0)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self._messages)})"