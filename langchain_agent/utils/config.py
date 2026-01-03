"""Configuration for the agent system."""

from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings for the agent system."""
    
    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # Output settings
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
    CHARTS_DIR: str = os.getenv("CHARTS_DIR", "output/charts")
    GRAPHS_DIR: str = os.getenv("GRAPHS_DIR", "output/graphs")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "output/reports")
    
    # Agent settings
    TEMPERATURE: float = 0.7
    MAX_ITERATIONS: int = 50
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all output directories exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHARTS_DIR, exist_ok=True)
        os.makedirs(cls.GRAPHS_DIR, exist_ok=True)
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

