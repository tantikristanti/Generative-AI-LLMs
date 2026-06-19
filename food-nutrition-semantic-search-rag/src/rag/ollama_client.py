# src/rag/ollama_client.py
"""Ollama LLM client for RAG system."""

import logging
import json
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import requests
import base64
from io import BytesIO
from ollama import generate
import io
from pathlib import Path

from rag import BaseLLMClient

logger = logging.getLogger(__name__)

class OllamaClient(BaseLLMClient):
    """
    Client for Ollama LLM with multimodal support.
    
    Supports:
    - Text generation with Ollama models (llama3, mistral, etc.)
    - Multimodal generation with LLaVA or similar vision models
    - Image generation (if supported by the model)
    """

    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 text_model: str = "llama3.2",
                 vision_model: str = "llava",
                 image_model: str = "x/flux2-klein",  
                 temperature: float = 0.7,
                 max_tokens: int = 2048):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL.
            text_model: Default text generation model.
            vision_model: Default vision model for multimodal.
            image_model: Default model for image generation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self.base_url = base_url
        self.text_model = text_model
        self.vision_model = vision_model
        self.image_model = image_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_ollama(self, model: str, prompt: str, 
                     system_prompt: Optional[str] = None,
                     images: Optional[List[Image.Image]] = None,
                     stream: bool = False) -> Dict:
        """Make a request to Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": stream,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if images:
            payload["images"] = self._encode_images(images)
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        if stream:
            return response  # Return raw response for streaming
        
        return response.json()

    def _encode_images(self, images: List[Image.Image]) -> List[str]:
        """Encode PIL images as base64 for Ollama."""
        encoded = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            encoded.append(img_base64)
        return encoded

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 **kwargs) -> str:
        """Generate text response."""
        model = kwargs.get('model', self.text_model)
        response = self._call_ollama(model, prompt, system_prompt)
        return response.get('response', '')

    def generate_multimodal(self, prompt: str, images: List[Image.Image],
                           system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response from text + images."""
        model = kwargs.get('model', self.vision_model)
        response = self._call_ollama(model, prompt, system_prompt, images)
        return response.get('response', '')

    def generate_image(self, prompt: str, out_file: str, **kwargs):
        """
        Generate an image from a text prompt using an Ollama image model.

        Returns:
            str: Output image path if successful, None otherwise.
        """

        model = kwargs.get("model", self.image_model)

        try:
            # Call Ollama through the same API path as other methods
            response = self._call_ollama(
                model=model,
                prompt=prompt,
                stream=False
            )

            # Image models return an "image" field
            image_b64 = response.get('image')

            if not image_b64:
                text_response = response.get('response', '')

                raise RuntimeError(
                    f"Model '{model}' did not generate an image.\n"
                    f"Response: {text_response}"
                )

            # Decode Base64 image
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception as e:
                raise RuntimeError(
                    f"Invalid Base64 image returned by '{model}': {e}"
                )

            # Convert bytes to PIL image
            try:
                image = Image.open(BytesIO(image_bytes))
                image.load()
            except Exception as e:
                raise RuntimeError(
                    f"Cannot decode generated image: {e}"
                )

            # Save image
            output_path = Path(out_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            image.save(output_path)

            logger.info(
                "Image generated: %s (%dx%d)",
                output_path,
                image.width,
                image.height
            )

            return str(output_path)

        except requests.exceptions.RequestException as e:
            logger.error("Ollama API error: %s", e)
            return None

        except Exception as e:
            logger.error("Image generation failed: %s", e)
            return None
        
    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None,
                        **kwargs):
        """Stream token-by-token response."""
        model = kwargs.get('model', self.text_model)
        response = self._call_ollama(model, prompt, system_prompt, stream=True)
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    yield data.get('response', '')
                except json.JSONDecodeError:
                    continue