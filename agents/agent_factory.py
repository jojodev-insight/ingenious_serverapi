"""Agent factory for dynamic agent creation and management."""

from typing import Any

from .base_agent import BaseAgent, ModelConfig, ProviderConfig
from .file_data_analyst import FileDataAnalyst
from .sample_agents import (
    CodeReviewerAgent,
    ContentWriterAgent,
    CustomAgent,
    DataAnalystAgent,
)


class AgentFactory:
    """Factory for creating and managing agents with custom configurations."""

    # Registry of available agent types
    AGENT_TYPES = {
        "data_analyst": DataAnalystAgent,
        "content_writer": ContentWriterAgent,
        "code_reviewer": CodeReviewerAgent,
        "file_data_analyst": FileDataAnalyst,
        "custom": CustomAgent,
        "autogen": BaseAgent,
        "autogen_workflow": BaseAgent
    }

    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        provider: str | None = None,
        model_name: str | None = None,
        model_config: ModelConfig | dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None
    ) -> BaseAgent:
        """Create an agent with specified configuration.
        
        Args:
            agent_type: Type of agent to create ('data_analyst', 'content_writer', etc.).
            provider: LLM provider ('openai', 'deepseek', 'anthropic').
            model_name: Specific model to use.
            model_config: Model configuration (ModelConfig or dict).
            agent_config: Additional agent configuration parameters.
            
        Returns:
            Configured agent instance.
            
        Raises:
            ValueError: If agent_type is not supported.
        """
        if agent_type not in cls.AGENT_TYPES:
            raise ValueError(f"Unsupported agent type: {agent_type}. Available: {list(cls.AGENT_TYPES.keys())}")

        agent_class = cls.AGENT_TYPES[agent_type]
        agent_config = agent_config or {}

        # Convert dict model_config to ModelConfig if needed
        if isinstance(model_config, dict):
            # Need a model_name for ModelConfig
            if "model_name" not in model_config:
                # If no model_name provided, we'll let the agent use its default
                model_config = None
            else:
                model_config = ModelConfig(**model_config)

        # Special handling for AutoGen agent types
        if agent_type == "autogen":
            init_params = {
                "name": "AutoGenAgent",
                "template_name": "autogen_dummy.txt",
                "provider": provider,
                "model_name": model_name,
                "model_config": model_config,
                "agent_type": "autogen",
                "agent_config": agent_config,
                "use_templates": False
            }
        elif agent_type == "autogen_workflow":
            init_params = {
                "name": "AutoGenWorkflowAgent", 
                "template_name": "autogen_dummy.txt",
                "provider": provider,
                "model_name": model_name,
                "model_config": model_config,
                "agent_type": "autogen_workflow",
                "agent_config": agent_config,
                "use_templates": False
            }
        else:
            # Standard agent initialization
            init_params = {
                "provider": provider,
                "model_name": model_name,
                "model_config": model_config,
                **agent_config
            }

        return agent_class(**init_params)

    @classmethod
    def create_custom_agent(
        cls,
        name: str,
        template_name: str,
        system_message: str,
        provider: str | None = None,
        model_name: str | None = None,
        model_config: ModelConfig | dict[str, Any] | None = None,
        **kwargs
    ) -> CustomAgent:
        """Create a custom agent with full configuration control.
        
        Args:
            name: Agent name.
            template_name: Template file name.
            system_message: System message for the agent.
            provider: LLM provider.
            model_name: Specific model to use.
            model_config: Model configuration.
            **kwargs: Additional parameters.
            
        Returns:
            Configured CustomAgent instance.
        """
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)

        return CustomAgent(
            name=name,
            template_name=template_name,
            system_message=system_message,
            provider=provider,
            model_name=model_name,
            model_config=model_config,
            **kwargs
        )

    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: type[BaseAgent]) -> None:
        """Register a new agent type.
        
        Args:
            agent_type: Name for the agent type.
            agent_class: Agent class to register.
        """
        cls.AGENT_TYPES[agent_type] = agent_class

    @classmethod
    def list_agent_types(cls) -> dict[str, str]:
        """List all available agent types.
        
        Returns:
            Dictionary mapping agent type names to class names.
        """
        return {name: cls.__name__ for name, cls in cls.AGENT_TYPES.items()}

    @classmethod
    def list_providers(cls) -> dict[str, list]:
        """List all available providers and their models.
        
        Returns:
            Dictionary mapping providers to their available models.
        """
        return ProviderConfig.list_models()

    @classmethod
    def create_model_config(
        cls,
        model_name: str,
        provider: str | None = None,
        **kwargs
    ) -> ModelConfig:
        """Create a model configuration.
        
        Args:
            model_name: Name of the model.
            provider: Provider name (to get defaults).
            **kwargs: Additional model parameters to override.
            
        Returns:
            ModelConfig instance.
        """
        if provider:
            try:
                # Get base config from provider defaults
                base_config = ProviderConfig.get_model_config(provider, model_name)
                # Override with custom parameters
                config_dict = base_config.to_dict()
                config_dict.update(kwargs)
                return ModelConfig(**{k: v for k, v in config_dict.items() if k != 'model'})
            except ValueError:
                # Provider/model not in predefined configs, create custom
                pass

        # Create custom model config
        return ModelConfig(model_name=model_name, **kwargs)


# Convenience functions for quick agent creation
def create_data_analyst(
    provider: str | None = None,
    model_name: str | None = None,
    **kwargs
) -> DataAnalystAgent:
    """Quick creation of a data analyst agent."""
    return AgentFactory.create_agent("data_analyst", provider, model_name, **kwargs)


def create_content_writer(
    provider: str | None = None,
    model_name: str | None = None,
    **kwargs
) -> ContentWriterAgent:
    """Quick creation of a content writer agent."""
    return AgentFactory.create_agent("content_writer", provider, model_name, **kwargs)


def create_code_reviewer(
    provider: str | None = None,
    model_name: str | None = None,
    **kwargs
) -> CodeReviewerAgent:
    """Quick creation of a code reviewer agent."""
    return AgentFactory.create_agent("code_reviewer", provider, model_name, **kwargs)
