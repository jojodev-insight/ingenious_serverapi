"""Sample agent implementations."""

from typing import Any

from .base_agent import BaseAgent, ModelConfig


class DataAnalystAgent(BaseAgent):
    """Agent specialized in data analysis tasks."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        model_config: ModelConfig | None = None,
        **kwargs
    ) -> None:
        # Use a model config optimized for analytical tasks
        if model_config is None and model_name is None:
            # Default to models good for analysis
            if provider == "deepseek":
                model_name = "deepseek-coder"  # Better for analytical reasoning
            elif provider == "openai":
                model_name = "gpt-4"  # Good balance of reasoning and analysis

        super().__init__(
            name="DataAnalyst",
            template_name="data_analyst_prompt.txt",
            provider=provider,
            model_name=model_name,
            model_config=model_config,
            system_message="You are a data analyst AI specialized in interpreting data, finding patterns, and providing insights.",
            **kwargs
        )

        # Adjust model config for analytical tasks (only if no custom config provided)
        if model_config is None:
            self.update_model_config(temperature=0.3)  # Lower temperature for more analytical responses

    def prepare_task(self, task_data: dict[str, Any]) -> str:
        """Prepare data analysis task prompt."""
        return self.render_prompt(
            data=task_data.get("data", ""),
            context=task_data.get("context", ""),
            analysis_type=task_data.get("analysis_type", "general"),
            output_format=task_data.get("output_format", "markdown")
        )


class ContentWriterAgent(BaseAgent):
    """Agent specialized in content writing tasks."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        model_config: ModelConfig | None = None,
        **kwargs
    ) -> None:
        # Use models good for creative writing
        if model_config is None and model_name is None:
            if provider == "openai":
                model_name = "gpt-4o"  # Excellent for creative content
            elif provider == "deepseek":
                model_name = "deepseek-chat"  # Good for conversational content

        super().__init__(
            name="ContentWriter",
            template_name="content_writer_prompt.txt",
            provider=provider,
            model_name=model_name,
            model_config=model_config,
            system_message="You are a creative content writer AI specialized in producing engaging, well-structured content.",
            **kwargs
        )

        # Adjust model config for creative tasks
        if model_config is None:
            self.update_model_config(temperature=0.8)  # Higher temperature for creativity

    def prepare_task(self, task_data: dict[str, Any]) -> str:
        """Prepare content writing task prompt."""
        return self.render_prompt(
            topic=task_data.get("topic", ""),
            audience=task_data.get("audience", "general"),
            content_type=task_data.get("content_type", "article"),
            tone=task_data.get("tone", "professional"),
            word_count=task_data.get("word_count", "500-1000"),
            requirements=task_data.get("requirements", ""),
            style=task_data.get("style", "informative")
        )


class CodeReviewerAgent(BaseAgent):
    """Agent specialized in code review tasks."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        model_config: ModelConfig | None = None,
        **kwargs
    ) -> None:
        # Use models optimized for code analysis
        if model_config is None and model_name is None:
            if provider == "deepseek":
                model_name = "deepseek-coder"  # Specialized for code
            elif provider == "openai":
                model_name = "gpt-4"  # Good at code analysis

        super().__init__(
            name="CodeReviewer",
            template_name="code_reviewer_prompt.txt",
            provider=provider,
            model_name=model_name,
            model_config=model_config,
            system_message="You are a senior software engineer AI specialized in code review, best practices, and security analysis.",
            **kwargs
        )

        # Adjust model config for precise code analysis
        if model_config is None:
            self.update_model_config(temperature=0.2)  # Very low temperature for accuracy

    def prepare_task(self, task_data: dict[str, Any]) -> str:
        """Prepare code review task prompt."""
        return self.render_prompt(
            code=task_data.get("code", ""),
            language=task_data.get("language", "unknown"),
            context=task_data.get("context", ""),
            review_type=task_data.get("review_type", "general"),
            focus_areas=task_data.get("focus_areas", "security, performance, maintainability")
        )


class CustomAgent(BaseAgent):
    """Customizable agent for flexible use cases."""

    def __init__(
        self,
        name: str,
        template_name: str,
        system_message: str,
        provider: str | None = None,
        model_name: str | None = None,
        model_config: ModelConfig | None = None,
        **kwargs
    ) -> None:
        """Initialize a custom agent.
        
        Args:
            name: Agent name.
            template_name: Template file name.
            system_message: Custom system message.
            provider: LLM provider.
            model_name: Specific model to use.
            model_config: Custom model configuration.
            **kwargs: Additional parameters.
        """
        super().__init__(
            name=name,
            template_name=template_name,
            provider=provider,
            model_name=model_name,
            model_config=model_config,
            system_message=system_message,
            **kwargs
        )

    def prepare_task(self, task_data: dict[str, Any]) -> str:
        """Prepare task prompt with flexible variable substitution."""
        # Use all task_data keys as template variables
        return self.render_prompt(**task_data)
