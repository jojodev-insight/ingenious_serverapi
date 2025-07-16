"""Test configuration module."""

import pytest
import os
from pathlib import Path
from core.config import Config


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_config_initialization(self, tmp_path):
        """Test configuration initialization with custom env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_openai_key
DEEPSEEK_API_KEY=test_deepseek_key
DEFAULT_PROVIDER=openai
LOG_LEVEL=DEBUG
""")
        
        config = Config(str(env_file))
        assert config.openai_api_key == "test_openai_key"
        assert config.deepseek_api_key == "test_deepseek_key"
        assert config.default_provider == "openai"
        assert config.log_level == "DEBUG"
    
    def test_missing_required_keys(self, tmp_path):
        """Test validation with missing required keys."""
        # Clear environment variables to ensure clean state
        old_openai_key = os.environ.pop("OPENAI_API_KEY", None)
        old_deepseek_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        
        try:
            env_file = tmp_path / ".env"
            env_file.write_text("SOME_OTHER_KEY=value")
            
            with pytest.raises(ValueError, match="Missing required environment variables"):
                Config(str(env_file))
        finally:
            # Restore environment variables
            if old_openai_key:
                os.environ["OPENAI_API_KEY"] = old_openai_key
            if old_deepseek_key:
                os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
    
    def test_llm_config_openai(self, tmp_path):
        """Test OpenAI LLM configuration."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_openai_key
DEEPSEEK_API_KEY=test_deepseek_key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_API_BASE=https://api.openai.com/v1
""")
        
        config = Config(str(env_file))
        llm_config = config.get_llm_config("openai")
        
        assert "config_list" in llm_config
        assert llm_config["config_list"][0]["model"] == "gpt-3.5-turbo"
        assert llm_config["config_list"][0]["api_key"] == "test_openai_key"
        assert llm_config["config_list"][0]["base_url"] == "https://api.openai.com/v1"
    
    def test_llm_config_azure_openai(self, tmp_path):
        """Test Azure OpenAI LLM configuration."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_azure_key
DEEPSEEK_API_KEY=test_deepseek_key
OPENAI_API_BASE=https://test-resource.openai.azure.com/
AZURE_DEPLOYMENT_NAME=gpt-4-deployment
AZURE_API_VERSION=2024-02-15-preview
""")
        
        config = Config(str(env_file))
        
        # Test Azure detection
        assert config.is_azure_openai is True
        assert config.azure_deployment_name == "gpt-4-deployment"
        assert config.azure_api_version == "2024-02-15-preview"
        
        # Test LLM config for Azure
        llm_config = config.get_llm_config("openai")
        
        assert "config_list" in llm_config
        config_item = llm_config["config_list"][0]
        assert config_item["model"] == "gpt-4-deployment"  # Should use deployment name
        assert config_item["api_key"] == "test_azure_key"
        assert config_item["azure_endpoint"] == "https://test-resource.openai.azure.com/"
        assert config_item["api_version"] == "2024-02-15-preview"
        assert config_item["azure_deployment"] == "gpt-4-deployment"
        assert "base_url" not in config_item  # Should not have base_url for Azure
    
    def test_llm_config_deepseek(self, tmp_path):
        """Test DeepSeek LLM configuration."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_openai_key
DEEPSEEK_API_KEY=test_deepseek_key
DEEPSEEK_MODEL=deepseek-coder
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
""")
        
        config = Config(str(env_file))
        llm_config = config.get_llm_config("deepseek")
        
        assert "config_list" in llm_config
        assert llm_config["config_list"][0]["model"] == "deepseek-coder"
        assert llm_config["config_list"][0]["api_key"] == "test_deepseek_key"
        assert llm_config["config_list"][0]["base_url"] == "https://api.deepseek.com/v1"
    
    def test_invalid_provider(self, tmp_path):
        """Test invalid provider raises error."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_openai_key
DEEPSEEK_API_KEY=test_deepseek_key
""")
        
        config = Config(str(env_file))
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            config.get_llm_config("invalid_provider")
    
    def test_azure_detection(self, tmp_path):
        """Test Azure OpenAI endpoint detection."""
        # Test Azure endpoint detection
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_key
DEEPSEEK_API_KEY=test_deepseek_key
OPENAI_API_BASE=https://my-resource.openai.azure.com/
""")
        
        config = Config(str(env_file))
        assert config.is_azure_openai is True
        
        # Test regular OpenAI endpoint
        env_file.write_text("""
OPENAI_API_KEY=test_key
DEEPSEEK_API_KEY=test_deepseek_key
OPENAI_API_BASE=https://api.openai.com/v1
""")
        
        config = Config(str(env_file))
        assert config.is_azure_openai is False
    
    def test_default_values(self, tmp_path):
        """Test default configuration values."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_openai_key
DEEPSEEK_API_KEY=test_deepseek_key
""")
        
        config = Config(str(env_file))
        
        # Test default values
        assert config.default_provider == "openai"
        assert config.openai_model == "gpt-4"
        assert config.deepseek_model == "deepseek-chat"
        assert config.deepseek_api_base == "https://api.deepseek.com/v1"
        assert config.log_level == "INFO"
        assert config.api_host == "localhost"
        assert config.api_port == 8000
        assert config.templates_dir == "templates"
        assert config.max_rounds == 10
        assert config.azure_api_version == "2024-02-15-preview"
    
    def test_custom_values(self, tmp_path):
        """Test custom configuration values."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_openai_key
DEEPSEEK_API_KEY=test_deepseek_key
DEFAULT_PROVIDER=deepseek
OPENAI_MODEL=gpt-3.5-turbo
DEEPSEEK_MODEL=deepseek-coder
LOG_LEVEL=DEBUG
API_HOST=0.0.0.0
API_PORT=9000
TEMPLATES_DIR=custom_templates
MAX_ROUNDS=20
AZURE_API_VERSION=2023-12-01-preview
""")
        
        config = Config(str(env_file))
        
        assert config.default_provider == "deepseek"
        assert config.openai_model == "gpt-3.5-turbo"
        assert config.deepseek_model == "deepseek-coder"
        assert config.log_level == "DEBUG"
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 9000
        assert config.templates_dir == "custom_templates"
        assert config.max_rounds == 20
        assert config.azure_api_version == "2023-12-01-preview"
    
    def test_get_all_config_info(self, tmp_path):
        """Test getting all configuration information."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_openai_key
DEEPSEEK_API_KEY=test_deepseek_key
OPENAI_API_BASE=https://test-resource.openai.azure.com/
AZURE_DEPLOYMENT_NAME=gpt-4-test
DEFAULT_PROVIDER=openai
""")
        
        config = Config(str(env_file))
        config_info = config.get_all_config_info()
        
        assert "providers" in config_info
        assert "api" in config_info
        assert "logging" in config_info
        
        # Check provider info
        providers = config_info["providers"]
        assert providers["default_provider"] == "openai"
        
        # Check OpenAI/Azure config
        openai_config = providers["openai"]
        assert openai_config["api_key_set"] is True
        assert openai_config["is_azure"] is True
        assert openai_config["azure_deployment"] == "gpt-4-test"
        assert openai_config["azure_api_version"] is not None
        
        # Check DeepSeek config
        deepseek_config = providers["deepseek"]
        assert deepseek_config["api_key_set"] is True
        
    def test_azure_deployment_name_fallback(self, tmp_path):
        """Test Azure deployment name fallback to model name."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test_key
DEEPSEEK_API_KEY=test_deepseek_key
OPENAI_API_BASE=https://test-resource.openai.azure.com/
OPENAI_MODEL=gpt-4-turbo
""")
        
        config = Config(str(env_file))
        
        # Should fallback to model name when deployment name not specified
        assert config.azure_deployment_name == "gpt-4-turbo"
