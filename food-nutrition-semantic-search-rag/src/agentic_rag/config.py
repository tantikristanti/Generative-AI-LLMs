"""Configuration for the agentic RAG system."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    
    # Search settings
    max_search_attempts: int = 3
    min_results_threshold: int = 2
    default_top_k: int = 5
    
    # Agent loop settings
    max_iterations: int = 5
    max_tool_calls_per_iteration: int = 3
    
    # Model settings
    model_name: str = "llama3.2"  
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Query refinement
    enable_typo_detection: bool = True
    enable_query_expansion: bool = True
    max_query_variations: int = 3
    
    # Logging
    verbose: bool = True
    
    @classmethod
    def default(cls) -> "AgentConfig":
        return cls()