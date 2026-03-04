# src/reflexion_agent/config/settings.py
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional
import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from reflexion_agent.utils.exceptions import ConfigurationError


class LLMRoleConfig(BaseModel):
    """Configuration for a specific LLM role."""
    provider: Literal["openai", "anthropic"]
    model: str
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    max_tokens: int = Field(gt=0, default=1024)
    
    @model_validator(mode='after')
    def validate_model_for_provider(self) -> 'LLMRoleConfig':
        """Validate that the model name matches the provider."""
        provider_models = {
            "openai": ["gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", 
                         "claude-opus-4-6", "claude-sonnet-4-6"]
        }
        
        # Check if model is known (optional, can be relaxed)
        known_models = provider_models.get(self.provider, [])
        if known_models and not any(m in self.model.lower() for m in known_models):
            # Just a warning, not an error - models change frequently
            import logging
            logging.warning(f"Unknown model '{self.model}' for provider '{self.provider}'")
        
        return self


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""
    name: str = "ReflexionAgent"
    max_iterations: int = Field(default=5, ge=1, le=20)
    stopping_score: float = Field(default=0.85, ge=0.0, le=1.0)
    min_improvement_delta: float = Field(default=0.05, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=120, gt=0)
    
    @model_validator(mode='after')
    def validate_delta_not_greater_than_stopping(self) -> 'AgentConfig':
        """Ensure min_improvement_delta is not greater than stopping_score."""
        if self.min_improvement_delta > self.stopping_score:
            raise ValueError(
                f"min_improvement_delta ({self.min_improvement_delta}) cannot be "
                f"greater than stopping_score ({self.stopping_score})"
            )
        return self


class MemoryConfig(BaseModel):
    """Memory configuration for vector stores."""
    
    class BackendConfig(BaseModel):
        """Configuration for a specific memory backend."""
        backend: Literal["chroma", "in_memory", "qdrant"] = "chroma"
        persist_directory: Optional[str] = None
        collection_name: str
        similarity_top_k: int = Field(default=5, ge=1, le=20)
        
        @model_validator(mode='after')
        def validate_persist_directory(self) -> 'BackendConfig':
            """Validate and create persist directory if needed."""
            if self.backend == "chroma" and self.persist_directory:
                # Convert to absolute path
                path = Path(self.persist_directory)
                if not path.is_absolute():
                    # If relative, make it relative to project root
                    path = Path.cwd() / self.persist_directory
                
                # Create directory if it doesn't exist
                path.mkdir(parents=True, exist_ok=True)
                self.persist_directory = str(path)
            
            return self
    
    episodic: BackendConfig
    reflection: BackendConfig
    short_term: dict[str, Any] = Field(default_factory=lambda: {"max_tokens": 8192})


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "pretty"] = "pretty"
    enable_structlog: bool = True


class ObservabilityConfig(BaseModel):
    """Observability (tracing/metrics) configuration."""
    enable_tracing: bool = False
    trace_exporter: Literal["console", "otlp", "none"] = "console"
    otlp_endpoint: Optional[str] = None
    metrics_enabled: bool = False


class ToolsConfig(BaseModel):
    """Tools configuration."""
    enabled: list[str] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_tools(self) -> 'ToolsConfig':
        """Validate that enabled tools are known."""
        known_tools = {"calculator", "web_search", "python_repl", "file_operations"}
        for tool in self.enabled:
            if tool not in known_tools:
                import logging
                logging.warning(f"Unknown tool '{tool}' enabled in config")
        return self


class LLMConfig(BaseModel):
    """LLM configuration with multiple roles."""
    default_provider: Literal["openai", "anthropic"] = "anthropic"
    actor: LLMRoleConfig
    evaluator: LLMRoleConfig
    reflector: LLMRoleConfig
    
    @model_validator(mode='after')
    def validate_default_provider_has_key(self) -> 'LLMConfig':
        """Ensure default provider has API key set (validation happens at runtime)."""
        return self


class Settings(BaseModel):
    """Main application settings loaded from YAML."""
    agent: AgentConfig
    llm: LLMConfig
    memory: MemoryConfig
    logging: LoggingConfig
    observability: ObservabilityConfig
    tools: ToolsConfig


class EnvSettings(BaseSettings):
    """Reads env vars and .env file. Secrets only — never YAML."""
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    env: Literal["development", "staging", "production"] = "development"
    config_path: str = "configs/default.yaml"
    project_root: Optional[str] = None  # For absolute path resolution
    
    def get_config_path(self) -> Path:
        """Get absolute path to config file."""
        if self.project_root:
            base = Path(self.project_root)
        else:
            # Try to find project root (where pyproject.toml is)
            current = Path.cwd()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    base = current
                    break
                current = current.parent
            else:
                base = Path.cwd()
        
        config_path = Path(self.config_path)
        if not config_path.is_absolute():
            config_path = base / config_path
        
        return config_path
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider."""
        if provider == "openai":
            return self.openai_api_key
        elif provider == "anthropic":
            return self.anthropic_api_key
        else:
            raise ConfigurationError(f"Unknown provider: {provider}")


@lru_cache(maxsize=1)
def get_settings() -> tuple[Settings, EnvSettings]:
    """Load and cache settings. Call get_settings() anywhere."""
    env = EnvSettings()
    
    config_path = env.get_config_path()
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        
        if not raw:
            raise ConfigurationError(f"Empty config file: {config_path}")
        
        settings = Settings(**raw)
        return settings, env
    
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config from {config_path}: {e}")


# Helper function to get config for specific environment
def get_config_for_env(env_name: str) -> tuple[Settings, EnvSettings]:
    """Load configuration for a specific environment."""
    env = EnvSettings(env=env_name)
    config_path = env.get_config_path()
    
    # Try environment-specific config first
    env_config_path = config_path.parent / f"{env_name}.yaml"
    if env_config_path.exists():
        env.config_path = str(env_config_path)
        return get_settings()
    
    # Fall back to default
    return get_settings()