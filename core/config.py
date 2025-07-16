"""Configuration management for the Autogen project."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class Config:
    """Configuration loader for environment variables and settings."""

    def __init__(self, env_file: str | None = None) -> None:
        """Initialize configuration loader.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in project root.
        """
        self._custom_env_file = env_file is not None
        self._original_env_values = {}

        if env_file is None:
            # Look for .env in project root
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"

        # For custom env files (mainly for testing), we need to ensure clean state
        if self._custom_env_file:
            env_vars_to_manage = [
                "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DEFAULT_PROVIDER", "OPENAI_MODEL",
                "DEEPSEEK_MODEL", "OPENAI_API_BASE", "DEEPSEEK_API_BASE", "AZURE_API_VERSION",
                "AZURE_DEPLOYMENT_NAME", "LOG_DIR", "LOG_LEVEL", "API_HOST", "API_PORT",
                "TEMPLATES_DIR", "MAX_ROUNDS"
            ]

            # Store original values and temporarily clear them
            for var in env_vars_to_manage:
                if var in os.environ:
                    self._original_env_values[var] = os.environ[var]
                    del os.environ[var]

        # Load the env file
        load_dotenv(env_file, override=True)
        self._validate_config()

    def restore_env_vars(self) -> None:
        """Restore original environment variables (for testing cleanup)."""
        if self._custom_env_file and hasattr(self, '_original_env_values'):
            # Clear current test values
            env_vars_to_manage = [
                "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DEFAULT_PROVIDER", "OPENAI_MODEL",
                "DEEPSEEK_MODEL", "OPENAI_API_BASE", "DEEPSEEK_API_BASE", "AZURE_API_VERSION",
                "AZURE_DEPLOYMENT_NAME", "LOG_DIR", "LOG_LEVEL", "API_HOST", "API_PORT",
                "TEMPLATES_DIR", "MAX_ROUNDS"
            ]

            for var in env_vars_to_manage:
                if var in os.environ:
                    del os.environ[var]

            # Restore original values
            for var, value in self._original_env_values.items():
                os.environ[var] = value

    def _validate_config(self) -> None:
        """Validate required configuration values."""
        required_keys = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]

        if missing_keys:
            raise ValueError(f"Missing required environment variables: {missing_keys}")

    @property
    def openai_api_key(self) -> str:
        """OpenAI API key."""
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def deepseek_api_key(self) -> str:
        """DeepSeek API key."""
        return os.getenv("DEEPSEEK_API_KEY", "")

    @property
    def default_provider(self) -> str:
        """Default LLM provider."""
        return os.getenv("DEFAULT_PROVIDER", "openai")

    @property
    def openai_model(self) -> str:
        """OpenAI model name."""
        return os.getenv("OPENAI_MODEL", "gpt-4")

    @property
    def deepseek_model(self) -> str:
        """DeepSeek model name."""
        return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    @property
    def openai_api_base(self) -> str:
        """OpenAI API base URL."""
        return os.getenv("OPENAI_API_BASE", "https://john-m80uo29y-eastus2.cognitiveservices.azure.com/")

    @property
    def azure_api_version(self) -> str:
        """Azure OpenAI API version."""
        return os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

    @property
    def is_azure_openai(self) -> bool:
        """Check if the OpenAI endpoint is Azure OpenAI."""
        api_base = self.openai_api_base.lower()
        return ("cognitiveservices.azure.com" in api_base or
                "openai.azure.com" in api_base)

    @property
    def azure_deployment_name(self) -> str:
        """Azure OpenAI deployment name (used instead of model name)."""
        return os.getenv("AZURE_DEPLOYMENT_NAME", self.openai_model)

    @property
    def deepseek_api_base(self) -> str:
        """DeepSeek API base URL."""
        return os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

    @property
    def log_dir(self) -> str:
        """Logging directory."""
        return os.getenv("LOG_DIR", "logs")

    @property
    def log_level(self) -> str:
        """Logging level."""
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def api_host(self) -> str:
        """API host."""
        return os.getenv("API_HOST", "localhost")

    @property
    def api_port(self) -> int:
        """API port."""
        return int(os.getenv("API_PORT", "8000"))

    @property
    def templates_dir(self) -> str:
        """Templates directory."""
        return os.getenv("TEMPLATES_DIR", "templates")

    @property
    def max_rounds(self) -> int:
        """Maximum conversation rounds."""
        return int(os.getenv("MAX_ROUNDS", "10"))

    def get_llm_config(self, provider: str | None = None, force_fallback: bool = False) -> dict[str, Any]:
        """Get LLM configuration for specified provider.
        
        Args:
            provider: LLM provider ('openai' or 'deepseek'). 
                     If None, uses default provider.
            force_fallback: If True, force OpenAI fallback instead of Azure.
        
        Returns:
            Dictionary containing LLM configuration.
        """
        if provider is None:
            provider = self.default_provider

        if provider.lower() == "openai":
            config_item = {
                "api_key": self.openai_api_key,
            }

            # Add Azure-specific configuration if available and not forcing fallback
            if self.is_azure_openai and not force_fallback:
                config_item.update({
                    "model": self.azure_deployment_name,
                    "azure_endpoint": self.openai_api_base,
                    "api_version": self.azure_api_version,
                    "azure_deployment": self.azure_deployment_name
                })
            else:
                # Regular OpenAI configuration (fallback)
                config_item.update({
                    "model": self.openai_model,
                    "base_url": self.openai_api_base if not self.is_azure_openai else "https://api.openai.com/v1"
                })

            return {
                "config_list": [config_item],
                "temperature": 0.7,
            }
        elif provider.lower() == "deepseek":
            return {
                "config_list": [
                    {
                        "model": self.deepseek_model,
                        "api_key": self.deepseek_api_key,
                        "base_url": self.deepseek_api_base,
                    }
                ],
                "temperature": 0.7,
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_all_config_info(self) -> dict[str, Any]:
        """Get all configuration information for debugging and validation.
        
        Returns:
            Dictionary containing all configuration details.
        """
        return {
            "providers": {
                "default_provider": self.default_provider,
                "openai": {
                    "api_key_set": bool(self.openai_api_key),
                    "model": self.openai_model,
                    "api_base": self.openai_api_base,
                    "is_azure": self.is_azure_openai,
                    "azure_api_version": self.azure_api_version if self.is_azure_openai else None,
                    "azure_deployment": self.azure_deployment_name if self.is_azure_openai else None,
                },
                "deepseek": {
                    "api_key_set": bool(self.deepseek_api_key),
                    "model": self.deepseek_model,
                    "api_base": self.deepseek_api_base,
                }
            },
            "api": {
                "host": self.api_host,
                "port": self.api_port,
            },
            "logging": {
                "log_dir": self.log_dir,
                "log_level": self.log_level,
            },
            "templates_dir": self.templates_dir,
            "max_rounds": self.max_rounds,
        }


# Global configuration instance
config = Config()
