"""
LLM Integration Templates and Examples for Community Naming

This module provides templates and examples for integrating different LLM services
for automated community naming. These are skeleton implementations that can be
extended with actual API keys and service configurations.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion from prompt."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client for GPT models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var)
            model: Model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model

        if not self.api_key:
            logger.warning(
                "No OpenAI API key provided. LLM naming will fall back to simple rules.")

    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI API."""
        if not self.api_key:
            return "Fallback_Community_Name"

        try:
            # TODO: Uncomment and configure when ready to use
            # import openai
            # openai.api_key = self.api_key
            #
            # response = openai.ChatCompletion.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "You are a research assistant that names academic communities."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=50,
            #     temperature=0.3,
            #     **kwargs
            # )
            #
            # return response.choices[0].message.content.strip()

            # Placeholder for now
            logger.info("OpenAI API call would be made here")
            return "OpenAI_Generated_Name"

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "Error_Community_Name"


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """Initialize Anthropic client."""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model

        if not self.api_key:
            logger.warning(
                "No Anthropic API key provided. LLM naming will fall back to simple rules.")

    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using Anthropic API."""
        if not self.api_key:
            return "Fallback_Community_Name"

        try:
            # TODO: Uncomment and configure when ready to use
            # import anthropic
            #
            # client = anthropic.Anthropic(api_key=self.api_key)
            #
            # response = client.messages.create(
            #     model=self.model,
            #     max_tokens=50,
            #     temperature=0.3,
            #     messages=[
            #         {"role": "user", "content": prompt}
            #     ],
            #     **kwargs
            # )
            #
            # return response.content[0].text.strip()

            # Placeholder for now
            logger.info("Anthropic API call would be made here")
            return "Claude_Generated_Name"

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return "Error_Community_Name"


class OllamaClient(LLMClient):
    """Local Ollama client for running models locally."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """Initialize Ollama client."""
        self.model = model
        self.base_url = base_url

    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using local Ollama."""
        try:
            # TODO: Uncomment and configure when ready to use
            # import requests
            #
            # response = requests.post(
            #     f"{self.base_url}/api/generate",
            #     json={
            #         "model": self.model,
            #         "prompt": prompt,
            #         "stream": False,
            #         **kwargs
            #     }
            # )
            #
            # if response.status_code == 200:
            #     return response.json()["response"].strip()
            # else:
            #     raise Exception(f"Ollama API returned status {response.status_code}")

            # Placeholder for now
            logger.info("Ollama API call would be made here")
            return "Ollama_Generated_Name"

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return "Error_Community_Name"


class LLMNamingService:
    """Service for generating community names using various LLM providers."""

    def __init__(self, client: LLMClient):
        """Initialize with specific LLM client."""
        self.client = client

    def create_naming_prompt(self, keywords: List[str], community_info: Dict[str, Any]) -> str:
        """Create an optimized prompt for community naming."""
        keywords_str = ", ".join(keywords[:10])
        size = community_info.get('size', 'unknown')

        # Enhanced prompt with better instructions
        prompt = f"""You are an expert research librarian helping to categorize academic papers.

Based on the following information about a research community, generate a concise, professional name:

Community Details:
- Size: {size} research papers
- Key terms: {keywords_str}
- Domain: Academic research

Requirements:
1. Name should be 2-4 words maximum
2. Use academic/scientific terminology
3. Be specific and descriptive
4. Avoid generic terms like "research", "study", "analysis"
5. Focus on the core topic or methodology

Examples of good names:
- "Machine Learning Optimization"
- "Quantum Computing Applications" 
- "Climate Change Modeling"
- "Natural Language Processing"

Generate only the community name, no explanation:"""

        return prompt.strip()

    def name_community(self, keywords: List[str], community_info: Dict[str, Any]) -> str:
        """Generate a name for a single community."""
        prompt = self.create_naming_prompt(keywords, community_info)

        try:
            raw_name = self.client.generate_completion(prompt)
            # Clean up the response
            clean_name = self._clean_community_name(raw_name)
            return clean_name
        except Exception as e:
            logger.error(f"Error generating community name: {e}")
            # Fallback to simple naming
            if keywords:
                return "_".join(keywords[:3]).title()
            else:
                return f"Community_{community_info.get('id', 'Unknown')}"

    def _clean_community_name(self, raw_name: str) -> str:
        """Clean and validate the generated name."""
        # Remove quotes, extra whitespace, etc.
        clean_name = raw_name.strip().strip('"').strip("'")

        # Ensure it's not too long
        if len(clean_name) > 50:
            words = clean_name.split()
            clean_name = " ".join(words[:4])

        # Title case
        clean_name = clean_name.title()

        # Replace spaces with underscores for consistency
        clean_name = clean_name.replace(" ", "_")

        return clean_name


# Factory function for easy client creation
def create_llm_client(provider: str = "openai", **kwargs) -> LLMClient:
    """Factory function to create LLM clients.

    Args:
        provider: LLM provider ("openai", "anthropic", "ollama")
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLM client
    """
    if provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider.lower() == "ollama":
        return OllamaClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# Example usage configuration
def create_naming_service_from_config(config: Dict[str, Any]) -> LLMNamingService:
    """Create naming service from configuration.

    Example config:
    {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": "your-key-here"
    }
    """
    provider = config.get("provider", "openai")
    client = create_llm_client(provider, **config)
    return LLMNamingService(client)


# Configuration templates
DEFAULT_CONFIGS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": None  # Set via environment variable
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "api_key": None  # Set via environment variable
    },
    "ollama": {
        "provider": "ollama",
        "model": "llama2",
        "base_url": "http://localhost:11434"
    }
}
