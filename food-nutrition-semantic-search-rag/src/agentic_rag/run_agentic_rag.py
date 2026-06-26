# The agentic RAG entry point

import sys
import logging
from pathlib import Path

# Add the src directory to PYTHONPATH if needed
sys.path.insert(0, str(Path(__file__).parent / "src"))

""" # Cross‑package imports
from rag.food_image_retriever import ImageAwareFoodRetriever
from rag.ollama_client import OllamaClient
from rag.config import RAGConfig

# Cross module import underr the same package (agentic_rag)
from .tools.search_tools import SearchTools
from .tools.response_tools import ResponseTools
from .agents import OrchestratorAgent
from .config import AgentConfig """

from rag import ImageAwareFoodRetriever, OllamaClient, RAGConfig
from agentic_rag import (
    SearchTools,
    ResponseTools,
    OrchestratorAgent,
    AgentConfig,
    SearchResult,
    FormattedResponse,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    rag_config = RAGConfig.from_yaml("config/rag_config.yaml")

    # 1. Database retriever (uses RAGConfig internally; or passing other configs if needed)
    retriever = ImageAwareFoodRetriever(config=rag_config)  # uses default config (localhost, etc.)
    
    # 2. LLM client (Ollama)
    llm = OllamaClient(
        base_url=rag_config.ollama_base_url,
        text_model=rag_config.text_model,
        vision_model=rag_config.vision_model,
        image_model=rag_config.image_model,  
        temperature=rag_config.temperature,
        max_tokens=rag_config.max_tokens
    )
    
    # 3. SearchTools with the refined pipeline
    search_tools = SearchTools(
        retriever=retriever,
        llm_client=llm,
        reranker_model_name="/Volumes/TantiK/Hugging_Face/cross-encoder/ms-marco-MiniLM-L6-v2/", # OR the default cache directory (e.g., ~/.cache/huggingface/hub).
        spell_dictionary_path="dictionary/SymSpell/frequency_dictionary_en_82_765.txt",
        top_k=20,          # candidates before reranking
        final_top_k=5,     # final results
    )
    
    # 4. ResponseTools for generating answers and follow-ups
    response_tools = ResponseTools(llm_client=llm, max_sources=5)
    
    # 5. Config (optional)
    config = AgentConfig.default()
    
    # 6. Orchestrator Agent
    orchestrator = OrchestratorAgent(
        search_tools=search_tools,
        response_tools=response_tools,
        config=config,
    )
    
    # 7. Run a query
    #query = "What is the protien content of chicken breast?"
    query="What are the nutritional values of the mixed salad with fish?"
    response = orchestrator.process(query)
    
    # The response.content is a JSON string; parse it for pretty printing
    import json
    result = json.loads(response.content)
    
    print("\n" + "="*50)
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Total sources: {result['total_sources']}")
    print(f"Follow-up questions: {result['follow_ups']}")
    print("\nRefined query:")
    if result.get('refined_query'):
        rq = result['refined_query']
        print(f"  Original: {rq['original']}")
        print(f"  Corrected: {rq['corrected']}")
        print(f"  Rewritten: {rq['rewritten']}")
        print(f"  Variations: {', '.join(rq['variations'])}")
    print("="*50 + "\n")
    
    # Optionally print sources
    for i, src in enumerate(result['sources'][:3], 1):
        print(f"Source {i}: {src['content'][:200]}... (score: {src.get('score', 'N/A')})")

if __name__ == "__main__":
    main()