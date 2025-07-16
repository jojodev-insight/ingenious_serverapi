"""Tests for enhanced agent functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents import ModelConfig, ProviderConfig, BaseAgent, AgentFactory
from agents.sample_agents import DataAnalystAgent, ContentWriterAgent, CodeReviewerAgent, CustomAgent


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_model_config_initialization(self):
        """Test ModelConfig initialization."""
        config = ModelConfig(
            model_name="gpt-4",
            max_tokens=2000,
            temperature=0.8,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )
        
        assert config.model_name == "gpt-4"
        assert config.max_tokens == 2000
        assert config.temperature == 0.8
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2
    
    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig("test-model")
        
        assert config.model_name == "test-model"
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
    
    def test_model_config_to_dict(self):
        """Test ModelConfig conversion to dictionary."""
        config = ModelConfig("gpt-4", max_tokens=1500, temperature=0.5)
        config_dict = config.to_dict()
        
        expected = {
            "model": "gpt-4",
            "max_tokens": 1500,
            "temperature": 0.5,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        assert config_dict == expected


class TestProviderConfig:
    """Test ProviderConfig class."""
    
    def test_get_model_config_openai(self):
        """Test getting OpenAI model configuration (Azure OpenAI priority)."""
        config = ProviderConfig.get_model_config("openai", "gpt-4")
        
        assert isinstance(config, ModelConfig)
        assert config.model_name == "gpt-4"  # Provider config returns requested model name
        assert config.max_tokens == 2000
        assert config.temperature == 0.7
    
    def test_get_model_config_openai_alternative(self):
        """Test getting OpenAI alternative model configuration."""
        config = ProviderConfig.get_model_config("openai", "gpt-4o")
        
        assert isinstance(config, ModelConfig)
        assert config.model_name == "gpt-4o"
        assert config.max_tokens == 4000  # gpt-4o has higher token limit
        assert config.temperature == 0.7
    
    def test_get_model_config_invalid_provider(self):
        """Test error for invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            ProviderConfig.get_model_config("invalid", "some-model")
    
    def test_get_model_config_invalid_model(self):
        """Test behavior for unknown model with OpenAI provider (should create default)."""
        # OpenAI provider handles unknown models by creating defaults (for Azure deployment names)
        config = ProviderConfig.get_model_config("openai", "invalid-model")
        assert isinstance(config, ModelConfig)
        assert config.model_name == "invalid-model"
        # Should get defaults based on gpt-4 config
        assert config.max_tokens == 2000
    
    def test_list_models_all(self):
        """Test listing all models (Azure OpenAI priority)."""
        models = ProviderConfig.list_models()
        
        assert "openai" in models
        assert "anthropic" in models
        assert "gpt-4" in models["openai"]
        assert "gpt-4-turbo" in models["openai"]  # Azure deployment model
        # Note: DeepSeek can still be available but OpenAI/Azure is prioritized
    
    def test_list_models_specific_provider(self):
        """Test listing models for specific provider (Azure OpenAI priority)."""
        models = ProviderConfig.list_models("openai")
        
        assert "openai" in models
        assert "gpt-4" in models["openai"]
        assert "gpt-4-turbo" in models["openai"]  # Azure deployment model


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def prepare_task(self, task_data):
        return f"Mock task: {task_data}"


class TestEnhancedBaseAgent:
    """Test enhanced BaseAgent functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.default_provider = "openai"
        self.mock_config.openai_api_key = "test-key"
        self.mock_config.openai_model = "gpt-4"
        self.mock_config.templates_dir = "templates"
        # Azure OpenAI configuration for priority
        self.mock_config.is_azure_openai = True
        self.mock_config.azure_deployment_name = "gpt-4.1-nano"  # Actual Azure deployment name
        self.mock_config.azure_api_version = "2024-02-15-preview"
        self.mock_config.openai_api_base = "https://test-resource.openai.azure.com/"
    
    @patch('agents.base_agent.config')
    @patch('agents.base_agent.Path')
    @patch('builtins.open')
    def test_agent_with_custom_model_name(self, mock_open, mock_path, mock_config):
        """Test agent initialization with custom model name (Azure OpenAI priority)."""
        mock_config.default_provider = "openai"
        mock_config.openai_api_key = "test-key"
        mock_config.templates_dir = "templates"
        mock_config.is_azure_openai = True
        mock_config.azure_deployment_name = "gpt-4.1-nano"  # Actual Azure deployment (prioritized)
        
        mock_path.return_value.exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = "Test template"
        
        with patch('agents.base_agent.openai.OpenAI'):
            agent = MockAgent(
                name="TestAgent",
                template_name="test.txt",
                provider="openai",
                model_name="gpt-4.1-nano"  # Use actual Azure deployment name
            )
        
        # With Azure OpenAI, the deployment name takes priority
        assert agent.model_config.model_name == "gpt-4.1-nano"
        assert agent.provider == "openai"
    
    @patch('agents.base_agent.config')
    @patch('agents.base_agent.Path')
    @patch('builtins.open')
    def test_agent_with_custom_model_config(self, mock_open, mock_path, mock_config):
        """Test agent initialization with custom model configuration (Azure OpenAI priority)."""
        mock_config.default_provider = "openai"
        mock_config.openai_api_key = "test-key"
        mock_config.templates_dir = "templates"
        mock_config.is_azure_openai = True
        mock_config.azure_deployment_name = "custom-model-deployment"
        
        mock_path.return_value.exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = "Test template"
        
        custom_config = ModelConfig("custom-model", max_tokens=3000, temperature=0.2)
        
        with patch('agents.base_agent.openai.OpenAI'):
            agent = MockAgent(
                name="TestAgent",
                template_name="test.txt",
                model_config=custom_config
            )
        
        # With Azure OpenAI, deployment name will be used but config values preserved
        assert agent.model_config.max_tokens == 3000
        assert agent.model_config.temperature == 0.2
    
    @patch('agents.base_agent.config')
    @patch('agents.base_agent.Path')
    @patch('builtins.open')
    def test_update_model_config(self, mock_open, mock_path, mock_config):
        """Test updating model configuration."""
        mock_config.default_provider = "openai"
        mock_config.openai_api_key = "test-key"
        mock_config.openai_model = "gpt-4"
        mock_config.templates_dir = "templates"
        
        mock_path.return_value.exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = "Test template"
        
        with patch('agents.base_agent.openai.OpenAI'):
            agent = MockAgent(
                name="TestAgent",
                template_name="test.txt"
            )
        
        # Update configuration
        agent.update_model_config(temperature=0.5, max_tokens=1500)
        
        assert agent.model_config.temperature == 0.5
        assert agent.model_config.max_tokens == 1500
    
    @patch('agents.base_agent.config')
    @patch('agents.base_agent.Path')
    @patch('builtins.open')
    def test_get_model_info(self, mock_open, mock_path, mock_config):
        """Test getting model information (Azure OpenAI priority)."""
        mock_config.default_provider = "openai"
        mock_config.openai_api_key = "test-key"
        mock_config.openai_model = "gpt-4"
        mock_config.templates_dir = "templates"
        mock_config.is_azure_openai = True
        mock_config.azure_deployment_name = "gpt-4.1-nano"  # Actual Azure deployment (prioritized)
        
        mock_path.return_value.exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = "Test template"
        
        with patch('agents.base_agent.openai.OpenAI'):
            agent = MockAgent(
                name="TestAgent",
                template_name="test.txt"
            )
        
        info = agent.get_model_info()
        
        assert info["agent_name"] == "TestAgent"
        assert info["provider"] == "openai"
        # With Azure OpenAI priority, deployment name is used
        assert info["model_name"] == "gpt-4.1-nano"
        assert "model_config" in info
        assert "system_message" in info


class TestAgentFactory:
    """Test AgentFactory functionality."""
    
    def test_create_data_analyst_agent(self):
        """Test creating data analyst agent."""
        with patch.object(AgentFactory, 'AGENT_TYPES') as mock_types:
            mock_class = Mock()
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            mock_types.__getitem__ = Mock(return_value=mock_class)
            mock_types.__contains__ = Mock(return_value=True)
            
            agent = AgentFactory.create_agent("data_analyst", provider="openai")
            
            assert agent == mock_instance
            mock_class.assert_called_once()
    
    def test_create_agent_with_model_config(self):
        """Test creating agent with custom model configuration."""
        model_config = {"model_name": "custom-model", "max_tokens": 2000, "temperature": 0.8}
        
        with patch.object(AgentFactory, 'AGENT_TYPES') as mock_types:
            mock_class = Mock()
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            mock_types.__getitem__ = Mock(return_value=mock_class)
            mock_types.__contains__ = Mock(return_value=True)
            
            agent = AgentFactory.create_agent(
                "data_analyst",
                provider="openai",
                model_config=model_config
            )
            
            assert agent == mock_instance
            # Verify ModelConfig was created and passed
            call_args = mock_class.call_args
            assert "model_config" in call_args.kwargs
            assert isinstance(call_args.kwargs["model_config"], ModelConfig)
    
    def test_create_custom_agent(self):
        """Test creating custom agent."""
        with patch('agents.agent_factory.CustomAgent') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            agent = AgentFactory.create_custom_agent(
                name="MyAgent",
                template_name="my_template.txt",
                system_message="Custom system message",
                provider="openai"
            )
            
            assert agent == mock_instance
            mock_class.assert_called_once_with(
                name="MyAgent",
                template_name="my_template.txt",
                system_message="Custom system message",
                provider="openai",
                model_name=None,
                model_config=None
            )
    
    def test_create_invalid_agent_type(self):
        """Test error for invalid agent type."""
        with pytest.raises(ValueError, match="Unsupported agent type: invalid"):
            AgentFactory.create_agent("invalid")
    
    def test_register_agent_type(self):
        """Test registering new agent type."""
        class TestAgent(BaseAgent):
            def prepare_task(self, task_data):
                return "test"
        
        AgentFactory.register_agent_type("test_agent", TestAgent)
        
        assert "test_agent" in AgentFactory.AGENT_TYPES
        assert AgentFactory.AGENT_TYPES["test_agent"] == TestAgent
    
    def test_list_agent_types(self):
        """Test listing agent types."""
        types = AgentFactory.list_agent_types()
        
        assert "data_analyst" in types
        assert "content_writer" in types
        assert "code_reviewer" in types
        assert "custom" in types
    
    def test_list_providers(self):
        """Test listing providers."""
        providers = AgentFactory.list_providers()
        
        assert "openai" in providers
        assert isinstance(providers["openai"], list)
        assert "gpt-4" in providers["openai"]
    
    def test_create_model_config(self):
        """Test creating model configuration."""
        config = AgentFactory.create_model_config(
            "custom-model",
            max_tokens=2000,
            temperature=0.5
        )
        
        assert isinstance(config, ModelConfig)
        assert config.model_name == "custom-model"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5


class TestEnhancedSampleAgents:
    """Test enhanced sample agents."""
    
    @patch('agents.sample_agents.BaseAgent.__init__')
    @patch.object(DataAnalystAgent, 'update_model_config')
    def test_data_analyst_agent_defaults(self, mock_update, mock_base_init):
        """Test DataAnalystAgent with Azure OpenAI provider-specific defaults."""
        mock_base_init.return_value = None
        
        # Test with OpenAI provider (Azure OpenAI priority)
        agent = DataAnalystAgent(provider="openai")
        
        # Verify BaseAgent was called with gpt-4 model (will be mapped to Azure deployment)
        call_args = mock_base_init.call_args
        assert call_args.kwargs["model_name"] == "gpt-4"
        assert call_args.kwargs["provider"] == "openai"
        
        # Verify update_model_config was called
        mock_update.assert_called_once_with(temperature=0.3)
    
    @patch('agents.sample_agents.BaseAgent.__init__')
    def test_content_writer_agent_temperature(self, mock_base_init):
        """Test ContentWriterAgent temperature adjustment (Azure OpenAI priority)."""
        mock_base_init.return_value = None
        
        with patch.object(ContentWriterAgent, 'update_model_config') as mock_update:
            agent = ContentWriterAgent(provider="openai")
            
            # Verify temperature was adjusted for creativity
            mock_update.assert_called_once_with(temperature=0.8)
    
    @patch('agents.sample_agents.BaseAgent.__init__')
    def test_code_reviewer_agent_precision(self, mock_base_init):
        """Test CodeReviewerAgent precision settings (Azure OpenAI priority)."""
        mock_base_init.return_value = None
        
        with patch.object(CodeReviewerAgent, 'update_model_config') as mock_update:
            agent = CodeReviewerAgent(provider="openai")
            
            # Verify temperature was set low for precision
            mock_update.assert_called_once_with(temperature=0.2)
    
    @patch('agents.sample_agents.BaseAgent.__init__')
    def test_custom_agent_flexibility(self, mock_base_init):
        """Test CustomAgent flexible initialization."""
        mock_base_init.return_value = None
        
        agent = CustomAgent(
            name="MyCustomAgent",
            template_name="custom.txt",
            system_message="Custom message",
            provider="anthropic"
        )
        
        call_args = mock_base_init.call_args
        assert call_args.kwargs["name"] == "MyCustomAgent"
        assert call_args.kwargs["template_name"] == "custom.txt"
        assert call_args.kwargs["system_message"] == "Custom message"
        assert call_args.kwargs["provider"] == "anthropic"
