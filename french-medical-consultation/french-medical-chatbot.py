"""
Medical Chatbot for Patient-Doctor Consultations (French)
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Data handling
from datasets import load_dataset
import pandas as pd

# Elasticsearch
from elasticsearch import Elasticsearch, helpers

# Gradio UI
import gradio as gr

# LLM integration 
from openai import OpenAI
import ollama  # Alternative local LLM


# 1. Data Loading Function
# ==================================================
def load_medical_data(num_rows: int = 500, cache_dir: str = "./medical_data_cache"):
    """
    Load medical dataset from Hugging Face, limit to first num_rows.
    
    Args:
        num_rows: Number of rows to load (default 500)
        cache_dir: Directory to cache the dataset
    
    Returns:
        List of dictionaries containing medical cases
    """
    logging.info(f"Loading dataset from Hugging Face (limit: {num_rows} rows)...")
    
    try:
        # Load the French medical dataset
        dataset = load_dataset(
            "rntc/combined-medical-french",
            split="train",
            streaming=False,  
            cache_dir=cache_dir
        )
        
        # Convert to list and limit rows
        data = list(dataset.select(range(min(num_rows, len(dataset)))))
        
        # Convert to pandas for easier handling
        df = pd.DataFrame(data)
        
        # Basic pre-processing
        if "text" in df.columns:
            # Remove rows with empty text
            df = df.dropna(subset=["text"])
            df = df[df["text"].str.strip() != ""]
        
        # Limit to specified number after cleaning
        df = df.head(num_rows)
        
        # Convert to list of dicts
        documents = df.to_dict("records")
        
        logging.info(f"Successfully loaded {len(documents)} medical cases")
        return documents
        
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        # Fallback: create sample data if dataset cannot be loaded
        return create_sample_data(num_rows)
    
def load_medical_data_limited(num_rows: int = 50, cache_dir: str = "./medical_data_cache"):
    """
    Load medical dataset from Hugging Face, limit to first num_rows.
    
    Args:
        num_rows: Number of rows to load (default 50)
        cache_dir: Directory to cache the dataset
    
    Returns:
        List of dictionaries containing medical cases
    """
    logging.info(f"Loading dataset from Hugging Face (limit: {num_rows} rows)...")
    
    # Load the French medical dataset
    dataset = load_dataset(
        "rntc/combined-medical-french",
        split="train",
        streaming=False,  # Load fully to limit rows
        cache_dir=cache_dir
    )
    
    # Convert to list and limit rows
    data = list(dataset.select(range(min(num_rows, len(dataset)))))
    
    # Convert to pandas for easier handling
    df = pd.DataFrame(data)
    
    # Basic preprocessing
    if "text" in df.columns:
        # Remove rows with empty text
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip() != ""]
    
    # Limit to specified number after cleaning
    df = df.head(num_rows)
    
    # Convert to list of dicts
    documents = df.to_dict("records")
    
    logging.info(f"Successfully loaded {len(documents)} medical cases")
    return documents
        
def create_sample_data(num_rows: int = 500) -> List[Dict]:
    """
    Create sample medical data for testing when dataset is unavailable.
    """
    sample_cases = [
        {
            "text": "Cas n°1: Patient de 45 ans présente des douleurs thoraciques typiques. Diagnostic: angine de poitrine stable.",
            "id": "sample_1",
            "source": "sample"
        },
        {
            "text": "Femme de 32 ans avec fièvre et toux depuis 5 jours. Diagnostic probable: bronchite aiguë.",
            "id": "sample_2",
            "source": "sample"
        },
        {
            "text": "Homme de 60 ans, hypertendu, se plaint de céphalées occipitales. Pression artérielle à 180/100 mmHg.",
            "id": "sample_3",
            "source": "sample"
        }
    ]
    
    # Repeat samples to reach desired count
    documents = []
    for i in range(num_rows):
        doc = sample_cases[i % len(sample_cases)].copy()
        doc["id"] = f"sample_{i}"
        documents.append(doc)
    
    return documents


# 2. Elasticsearch Setup and Search Function
# ==================================================
def setup_elasticsearch_index(documents: List[Dict], index_name: str = "medical_cases"):
    """
    Index medical documents into Elasticsearch for efficient searching.
    
    Args:
        documents: List of medical case dictionaries
        index_name: Name of the Elasticsearch index
    """
    logging.info(f"Setting up Elasticsearch index '{index_name}'...")
    
    # Connect to Elasticsearch
    es_client = Elasticsearch("http://localhost:9200")
    
    # Check if index exists and delete if needed (for fresh setup)
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        logging.info(f"Deleted existing index '{index_name}'")
    
    # Define index mapping for better search
    mapping = {
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "french",  # French language analyzer
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "id": {"type": "keyword"},
                "source": {"type": "keyword"}
            }
        }
    }
    
    # Create index with mapping
    es_client.indices.create(index=index_name, body=mapping)
    logging.info(f"Created index '{index_name}' with French analyzer")
    
    # Prepare documents for bulk indexing
    actions = [
        {
            "_index": index_name,
            "_id": doc.get("id", f"doc_{i}"),
            "_source": {k: v for k, v in doc.items() if v is not None}
        }
        for i, doc in enumerate(documents)
    ]
    
    # Bulk index documents
    success, failed = helpers.bulk(es_client, actions, stats_only=True)
    logging.info(f"Indexed {success} documents, {failed} failed")
    
    # Refresh index to make documents searchable
    es_client.indices.refresh(index=index_name)
    
    return es_client


def elastic_search(query: str, es_client: Elasticsearch, index_name: str = "medical_cases", 
                   size: int = 5) -> List[Dict]:
    """
    Search for relevant medical cases using Elasticsearch.
    
    Args:
        query: User's question in French
        es_client: Elasticsearch client instance
        index_name: Name of the index to search
        size: Number of results to return
    
    Returns:
        List of relevant medical documents
    """
    search_query = {
        "size": size, # top_k
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^3"],  # Boost text field
                        "type": "best_fields",
                        "fuzziness": "AUTO"  # Handle typos
                    }
                }
            }
        },
        # Boost more recent or relevant documents
        "sort": [
            {"_score": {"order": "desc"}}
        ]
    }
    
    try:
        response = es_client.search(index=index_name, body=search_query)
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'score': hit['_score'],
                'text': hit['_source'].get('text', ''),
                'id': hit['_source'].get('id', '')
            })
        
        logging.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results
        
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return []


# 3. Prompt Generation Function (French)
# ==================================================
def build_medical_prompt(question: str, search_results: List[Dict]) -> str:
    """
    Build a French prompt for the medical chatbot.
    
    Args:
        question: User's medical question in French
        search_results: Relevant medical cases from Elasticsearch
    
    Returns:
        Formatted prompt for the LLM
    """
    # System instructions in French
    system_instructions = {
        "role": "système",
        "content": """Vous êtes un assistant médical virtuel spécialisé dans les consultations patient-médecin. 
            Votre rôle est de fournir des informations médicales précises, basées sur des données fiables.
            Important: Vous n'êtes pas un substitut à un médecin réel. En cas d'urgence, recommandez une consultation médicale immédiate.
            Utilisez uniquement les informations fournies dans le CONTEXTE pour répondre.
            Si vous ne trouvez pas la réponse dans le contexte, indiquez-le clairement et recommandez de consulter un professionnel de santé.
            Répondez toujours en français, de manière claire, empathique et professionnelle.
        """
    }
    
    # Build context from search results
    context = ""
    for i, doc in enumerate(search_results, 1):
        context += f"Cas médical {i}:\n{doc.get('text', '')}\n\n"
    
    # User prompt template
    user_prompt_template = """# QUESTION MÉDICALE
    {question}

    # CONTEXTE MÉDICAL (Utilisez uniquement ces informations)
    {context}

    # INSTRUCTIONS
    Répondez à la question médicale ci-dessus en utilisant UNIQUEMENT le contexte fourni.
    Structurez votre réponse comme suit:
    1. Résumé de la situation (si applicable)
    2. Informations médicales pertinentes
    3. Recommandations basées sur le contexte
    4. Avertissement sur les limites de cette consultation virtuelle

    RÉPONSE:"""
    
    user_prompt = user_prompt_template.format(
        question=question,
        context=context if context else "Aucun contexte médical spécifique trouvé. Recommandez une consultation médicale en personne."
    )
    
    # Format as conversation (if needed)
    conversation = [
        system_instructions,
        {"role": "utilisateur", "content": user_prompt}
    ]
    
    # Convert to string format for simpler LLMs
    prompt = f"{system_instructions['content']}\n\n{user_prompt}"
    
    return prompt


# 4. LLM Integration (Multiple options)
# ============================================================================
def get_llm_response(prompt: str, model: str = "deepseek-v3.2:cloud", provider: str = "ollama") -> str:
    """
    Get response from LLM using specified provider.
    
    Args:
        prompt: The formatted prompt
        model: Model name
        provider: "ollama", "openai", or "mock"
    
    Returns:
        LLM response text
    """
    try:
        if provider == "ollama":
            # Local Ollama (free, open-source)
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
            
        elif provider == "openai":
            # OpenAI API (requires API key)
            client = OpenAI(
                base_url='http://localhost:11434/v1/',  # For local, or use OpenAI's endpoint
                api_key='ollama'
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
        else:  # mock provider for testing
            return generate_mock_response(prompt)
            
    except Exception as e:
        logging.error(f"LLM error: {str(e)}")
        return "Désolé, je rencontre une difficulté technique. Veuillez réessayer ou consulter un médecin directement."


def generate_mock_response(prompt: str) -> str:
    """
    Generate a mock response for testing without LLM.
    """
    responses = [
        "D'après les informations médicales disponibles, il est recommandé de consulter un médecin généraliste pour une évaluation complète de vos symptômes.",
        "Les cas similaires dans notre base de données suggèrent qu'une consultation médicale serait appropriée. Pouvez-vous décrire plus précisément vos symptômes ?",
        "En me basant sur les données médicales, je vous recommande de surveiller vos symptômes et de consulter si la situation s'aggrave dans les 48 heures.",
        "Je comprends votre préoccupation. D'après le contexte médical, il serait prudent de prendre rendez-vous avec votre médecin traitant pour un examen approfondi."
    ]
    import random
    return random.choice(responses)


# 5. RAG Pipeline
# ============================================================================
def rag_pipeline(question: str, es_client: Elasticsearch, index_name: str = "medical_cases",
                 model: str = "deepseek-v3.2:cloud", provider: str = "ollama") -> str:
    """
    Complete RAG pipeline: search -> prompt -> LLM -> answer.
    
    Args:
        question: User's medical question in French
        es_client: Elasticsearch client
        index_name: Name of the index
        model: LLM model name
        provider: LLM provider
    
    Returns:
        Generated answer
    """
    # Step 1: Search for relevant medical cases
    search_results = elastic_search(question, es_client, index_name)
    
    if not search_results:
        return "Je n'ai pas trouvé d'informations médicales pertinentes. Je vous recommande de consulter un médecin directement pour obtenir des conseils personnalisés."
    
    # Step 2: Build prompt with context
    prompt = build_medical_prompt(question, search_results)
    
    # Step 3: Get LLM response
    answer = get_llm_response(prompt, model, provider)
    
    # Step 4: Add disclaimer
    disclaimer = "\n\n---\n⚠️ **Avertissement médical**: Cette consultation virtuelle est fournie à titre informatif uniquement. En cas d'urgence médicale, contactez immédiatement les services d'urgence (15 en France)."
    
    return answer + disclaimer


# 6. Gradio User Interface
# ============================================================================
import gradio as gr
from elasticsearch import Elasticsearch

def create_gradio_interface(es_client, index_name: str = "medical_cases"):
    
    def respond(message, history):
        if not message or not message.strip():
            return "Veuillez poser une question médicale spécifique."
        
        return rag_pipeline(message, es_client, index_name)
    
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .chatbot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(title="Assistant Médical - Consultation Virtuelle") as demo:
        
        gr.HTML("""
        <div class="chatbot-header">
            <h1>🏥 Assistant Médical Virtuel</h1>
            <p>Consultation patient-médecin basée sur des données médicales françaises</p>
            <p><small>⚠️ À titre informatif uniquement</small></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                gr.ChatInterface(
                    fn=respond,
                    title="💬 Consultation avec votre assistant médical",
                    description="""
                    Posez vos questions médicales en français.
                    """,
                    examples=[
                        "Quels sont les symptômes d'une angine ?",
                        "Que faire en cas de fièvre ?"
                    ]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### ℹ️ Informations importantes...")
    
    return demo, custom_css


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main function to run the complete medical chatbot system."""
    
    import logging

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting Medical Chatbot System...")
    
    # Step 1: Load medical data
    print("\n📥 Étape 1: Chargement des données médicales...")
    #medical_cases = load_medical_data(num_rows=500)
    medical_cases = load_medical_data_limited(num_rows=50)
    print(f"✅ {len(medical_cases)} cas médicaux chargés")
    
    # Step 2: Setup Elasticsearch
    print("\n🔍 Étape 2: Configuration d'Elasticsearch...")
    try:
        es_client = setup_elasticsearch_index(medical_cases)
        print("✅ Elasticsearch configuré et données indexées")
    except Exception as e:
        print(f"⚠️ Erreur Elasticsearch: {e}")
        print("   Utilisation d'un client simulé...")
        es_client = None
    
    # Step 3: Launch Gradio interface
    print("\n🌐 Étape 3: Lancement de l'interface utilisateur...")
    
    # ✅ FIX: unpack both return values
    demo, custom_css = create_gradio_interface(es_client)
    
    print("\n" + "="*60)
    print("🏥 ASSISTANT MÉDICAL VIRTUEL PRÊT")
    print("="*60)
    print("🌐 Interface Gradio disponible sur: http://localhost:7860")
    print("ℹ️ Appuyez sur Ctrl+C pour arrêter l'application")
    print("="*60 + "\n")
    
    # Css (Gradio 6.x requirement)
    demo.launch(
    css=custom_css,
    theme="soft",   
    share=False,
    debug=False,
    server_name="0.0.0.0",
    server_port=7860
)

if __name__ == "__main__":
    # The entry point
    main()