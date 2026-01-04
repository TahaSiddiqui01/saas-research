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
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    # Generic LLM provider selection: 'ollama' or 'openai'
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # OpenAI settings (when provider is 'openai')
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    MAX_ITERATIONS: int = 50
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", "15"))
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all output directories exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHARTS_DIR, exist_ok=True)
        os.makedirs(cls.GRAPHS_DIR, exist_ok=True)
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

    @classmethod
    def validate(cls):
        """Perform quick validation of configuration and create output dirs.

        Should be called at program startup.
        """
        cls.ensure_directories()
        # Additional validation could be added here in future

    # LLM instance cache
    _LLM_INSTANCE = None

    @classmethod
    def get_chat_llm(cls):
        """Return a chat-capable LLM instance based on current configuration.

        This centralizes LLM creation so agents and tools call a single factory.
        The instance is cached on the class so it's only created once.
        """
        if cls._LLM_INSTANCE is not None:
            return cls._LLM_INSTANCE

        provider = cls.LLM_PROVIDER.lower()
        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama

                llm = ChatOllama(model=cls.OLLAMA_MODEL, base_url=cls.OLLAMA_BASE_URL, temperature=cls.TEMPERATURE)
                cls._LLM_INSTANCE = llm
                return llm
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Ollama LLM: {e}")
        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(model_name=cls.OPENAI_MODEL, temperature=cls.TEMPERATURE, openai_api_key=cls.OPENAI_API_KEY)
                cls._LLM_INSTANCE = llm
                return llm
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI LLM: {e}")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {cls.LLM_PROVIDER}")

