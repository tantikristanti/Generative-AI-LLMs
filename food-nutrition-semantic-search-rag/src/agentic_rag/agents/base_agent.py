"""Base agent class with common functionality."""
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class Tool:
    """Definition of a tool that an agent can use."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

@dataclass
class AgentMessage:
    """A message in the agent conversation."""
    role: str  # "user", "assistant", "tool", "developer"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

@dataclass
class AgentResponse:
    """Response from an agent."""
    success: bool
    content: str
    messages: List[AgentMessage] = field(default_factory=list)
    tool_calls_made: List[Dict] = field(default_factory=list)
    iterations: int = 0
    error: Optional[str] = None

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._messages: List[AgentMessage] = []
        
        # Build tool registry
        self._tool_registry = {t.name: t for t in self.tools}
    
    def add_message(self, message: AgentMessage):
        """Add a message to the conversation history."""
        self._messages.append(message)
    
    def get_messages(self) -> List[Dict]:
        """Get messages in a format suitable for LLM API."""
        result = []
        for msg in self._messages:
            entry = {"role": msg.role}
            if msg.content:
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
            result.append(entry)
        return result
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool by name with given arguments."""
        if tool_name not in self._tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self._tool_registry[tool_name]
        try:
            result = tool.function(**arguments)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e)}
    
    def _format_tool_result(self, result: Any) -> str:
        """Format tool result for inclusion in messages."""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, default=str, indent=2)
        except:
            return str(result)
    
    @abstractmethod
    def process(self, input_text: str, **kwargs) -> AgentResponse:
        """Process input and return a response."""
        pass
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self._messages = []