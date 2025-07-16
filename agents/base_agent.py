"""Base agent implementation for the Autogen project."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import json

import openai
import pandas as pd
import PyPDF2
from docx import Document
from jinja2 import Template

# AutoGen imports with availability check
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.messages import ChatMessage, TextMessage
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_core.models import ChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError as e:
    AUTOGEN_AVAILABLE = False
    AUTOGEN_IMPORT_ERROR = str(e)

from core import agent_logger, config


class FileProcessor:
    """Utility class for processing various file formats."""

    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Read text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Extracted text content.
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise ValueError(f"Error reading PDF file {file_path}: {str(e)}")

    @staticmethod
    def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """Read data from a CSV file.
        
        Args:
            file_path: Path to the CSV file.
            **kwargs: Additional parameters for pandas.read_csv().
            
        Returns:
            DataFrame containing the CSV data.
        """
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading CSV file {file_path}: {str(e)}")

    @staticmethod
    def read_excel(file_path: str, sheet_name: str | None = 0, **kwargs) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Read data from an Excel file.
        
        Args:
            file_path: Path to the Excel file.
            sheet_name: Specific sheet name or index, or None for all sheets (default: 0 for first sheet).
            **kwargs: Additional parameters for pandas.read_excel().
            
        Returns:
            DataFrame or dictionary of DataFrames.
        """
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading Excel file {file_path}: {str(e)}")

    @staticmethod
    def read_docx(file_path: str) -> str:
        """Read text content from a Word document.
        
        Args:
            file_path: Path to the Word document.
            
        Returns:
            Extracted text content.
        """
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error reading Word document {file_path}: {str(e)}")

    @staticmethod
    def get_file_info(file_path: str) -> dict[str, Any]:
        """Get information about a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Dictionary containing file information.
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            return {
                "name": path.name,
                "size": stat.st_size,
                "extension": path.suffix.lower(),
                "modified": stat.st_mtime,
                "exists": path.exists()
            }
        except Exception as e:
            return {"error": str(e), "exists": False}

    @staticmethod
    def process_file(file_path: str, **kwargs) -> dict[str, Any]:
        """Process a file based on its extension.
        
        Args:
            file_path: Path to the file.
            **kwargs: Additional parameters for specific file readers.
            
        Returns:
            Dictionary containing processed file data and metadata.
        """
        file_info = FileProcessor.get_file_info(file_path)

        if not file_info.get("exists", False):
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_info["extension"]
        result = {"file_info": file_info, "content": None, "data": None}

        try:
            if extension == ".pdf":
                result["content"] = FileProcessor.read_pdf(file_path)
                result["type"] = "text"
            elif extension == ".csv":
                result["data"] = FileProcessor.read_csv(file_path, **kwargs)
                result["type"] = "dataframe"
            elif extension in [".xlsx", ".xls"]:
                result["data"] = FileProcessor.read_excel(file_path, **kwargs)
                result["type"] = "dataframe"
            elif extension == ".docx":
                result["content"] = FileProcessor.read_docx(file_path)
                result["type"] = "text"
            else:
                # Try to read as text file
                with open(file_path, encoding='utf-8') as f:
                    result["content"] = f.read()
                result["type"] = "text"

        except Exception as e:
            result["error"] = str(e)
            result["type"] = "error"

        return result


class ModelConfig:
    """Configuration for LLM models."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ):
        """Initialize model configuration.
        
        Args:
            model_name: Name of the model to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 to 2.0).
            top_p: Nucleus sampling parameter (0.0 to 1.0).
            frequency_penalty: Frequency penalty (-2.0 to 2.0).
            presence_penalty: Presence penalty (-2.0 to 2.0).
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }


class ProviderConfig:
    """Configuration for LLM providers."""

    # Predefined model configurations for different providers
    PROVIDER_MODELS = {
        "openai": {
            "gpt-4": ModelConfig("gpt-4", max_tokens=2000, temperature=0.7),
            "gpt-4-turbo": ModelConfig("gpt-4-turbo", max_tokens=4000, temperature=0.7),
            "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", max_tokens=1500, temperature=0.7),
            "gpt-4o": ModelConfig("gpt-4o", max_tokens=4000, temperature=0.7),
            "gpt-4o-mini": ModelConfig("gpt-4o-mini", max_tokens=2000, temperature=0.7)
        },
        "deepseek": {
            "deepseek-chat": ModelConfig("deepseek-chat", max_tokens=2000, temperature=0.7),
            "deepseek-coder": ModelConfig("deepseek-coder", max_tokens=2000, temperature=0.3),
            "deepseek-math": ModelConfig("deepseek-math", max_tokens=1500, temperature=0.5)
        },
        "anthropic": {
            "claude-3-opus": ModelConfig("claude-3-opus-20240229", max_tokens=2000, temperature=0.7),
            "claude-3-sonnet": ModelConfig("claude-3-sonnet-20240229", max_tokens=2000, temperature=0.7),
            "claude-3-haiku": ModelConfig("claude-3-haiku-20240307", max_tokens=1500, temperature=0.7)
        }
    }

    @classmethod
    def get_model_config(cls, provider: str, model_name: str) -> ModelConfig:
        """Get model configuration for a provider and model.
        
        Args:
            provider: Provider name (e.g., 'openai', 'deepseek', 'anthropic').
            model_name: Model name or deployment name.
            
        Returns:
            ModelConfig instance.
            
        Raises:
            ValueError: If provider or model is not supported.
        """
        provider = provider.lower()
        if provider not in cls.PROVIDER_MODELS:
            raise ValueError(f"Unsupported provider: {provider}")

        # For Azure OpenAI, the model_name might be a deployment name
        # Try to find a matching model config, or create a default one
        if provider == "openai" and model_name not in cls.PROVIDER_MODELS[provider]:
            # This might be an Azure deployment name
            # Try to match common patterns or use a default GPT-4 config
            if "gpt-4" in model_name.lower():
                base_config = cls.PROVIDER_MODELS[provider].get("gpt-4")
            elif "gpt-3.5" in model_name.lower():
                base_config = cls.PROVIDER_MODELS[provider].get("gpt-3.5-turbo")
            else:
                # Default to gpt-4 config for unknown deployment names
                base_config = cls.PROVIDER_MODELS[provider].get("gpt-4")

            if base_config:
                # Create a new config with the deployment name
                return ModelConfig(
                    model_name=model_name,
                    max_tokens=base_config.max_tokens,
                    temperature=base_config.temperature,
                    top_p=base_config.top_p,
                    frequency_penalty=base_config.frequency_penalty,
                    presence_penalty=base_config.presence_penalty
                )

        if model_name not in cls.PROVIDER_MODELS[provider]:
            raise ValueError(f"Unsupported model '{model_name}' for provider '{provider}'")

        return cls.PROVIDER_MODELS[provider][model_name]

    @classmethod
    def list_models(cls, provider: str | None = None) -> dict[str, list]:
        """List available models for providers.
        
        Args:
            provider: Specific provider to list models for, or None for all.
            
        Returns:
            Dictionary mapping providers to their available models.
        """
        if provider:
            provider = provider.lower()
            if provider in cls.PROVIDER_MODELS:
                return {provider: list(cls.PROVIDER_MODELS[provider].keys())}
            else:
                return {}

        return {p: list(models.keys()) for p, models in cls.PROVIDER_MODELS.items()}


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(
        self,
        name: str,
        template_name: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        model_config: ModelConfig | None = None,
        system_message: str | None = None,
        custom_client_params: dict[str, Any] | None = None,
        data_dir: str | None = None,
        use_templates: bool = True,
        agent_type: str = "standard",
        agent_config: dict[str, Any] | None = None
    ) -> None:
        """Initialize the base agent.
        
        Args:
            name: Agent name.
            template_name: Name of the prompt template file (optional for some agent types).
            provider: LLM provider ('openai', 'deepseek', 'anthropic').
            model_name: Specific model to use (overrides config defaults).
            model_config: Custom model configuration (overrides defaults).
            system_message: System message for the agent.
            custom_client_params: Additional parameters for the client.
            data_dir: Directory path for data files (defaults to 'data' in project root).
            use_templates: Whether this agent uses template files (False for AutoGen-style agents).
            agent_type: Type of agent ('standard', 'autogen', 'autogen_workflow').
            agent_config: Additional configuration for specific agent types.
        """
        self.name = name
        self.template_name = template_name
        self.provider = provider or config.default_provider
        self.custom_client_params = custom_client_params or {}
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.use_templates = use_templates
        self.agent_type = agent_type
        self.agent_config = agent_config or {}

        # AutoGen-specific initialization
        if self.agent_type in ["autogen", "autogen_workflow"]:
            if not AUTOGEN_AVAILABLE:
                agent_logger.warning(f"AutoGen not available for agent '{name}': {AUTOGEN_IMPORT_ERROR}")
            
            # AutoGen configuration
            self.max_turns = self.agent_config.get("max_turns", 10)
            self.temperature = self.agent_config.get("temperature", 0.7)
            self.agents_config = self.agent_config.get("agents", [])
            
            # For AutoGen workflow agents
            if self.agent_type == "autogen_workflow":
                self.workflow_type = self.agent_config.get("workflow_type", "research")
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)

        # Set up model configuration
        if model_config:
            self.model_config = model_config
        elif model_name:
            self.model_config = ProviderConfig.get_model_config(self.provider, model_name)
        else:
            # Use default model from config
            default_model = self._get_default_model_name()
            self.model_config = ProviderConfig.get_model_config(self.provider, default_model)

        # Handle Azure OpenAI deployment names
        if self.provider.lower() == "openai" and config.is_azure_openai:
            # For Azure OpenAI, use deployment name from config
            original_model = self.model_config.model_name
            deployment_name = config.azure_deployment_name

            # If a specific model was requested, try to map it to deployment name
            # Otherwise use the configured deployment name
            if model_name and model_name != original_model:
                # User specified a different model, but for Azure we still need to use deployment name
                # Log this mapping
                agent_logger.info(f"Azure OpenAI: Requested model '{model_name}' will use deployment '{deployment_name}'")

            self.model_config.model_name = deployment_name
            agent_logger.info(f"Azure OpenAI detected: Using deployment '{deployment_name}' (original model: '{original_model}')")

        self.system_message = system_message or f"You are {name}, a helpful AI assistant."

        # Load prompt template only if templates are used
        if self.use_templates:
            if not template_name:
                raise ValueError(f"template_name is required when use_templates=True for agent '{name}'")
            self.template = self._load_template()
        else:
            self.template = None

        # Initialize client
        self.client = self._create_client()

    def _get_client_params_from_config(self) -> dict[str, Any]:
        """Get client parameters from configuration for this provider."""
        llm_config = config.get_llm_config(self.provider)
        if llm_config.get("config_list"):
            # Extract the first config item (should be the only one for single provider)
            config_item = llm_config["config_list"][0]
            return {k: v for k, v in config_item.items() if k != "model"}
        return {}

    def _get_default_model_name(self) -> str:
        """Get the default model name for the current provider."""
        if self.provider.lower() == "openai":
            return config.openai_model
        elif self.provider.lower() == "deepseek":
            return config.deepseek_model
        else:
            # For other providers, use a sensible default
            provider_models = ProviderConfig.PROVIDER_MODELS.get(self.provider.lower(), {})
            if provider_models:
                return list(provider_models.keys())[0]
            else:
                raise ValueError(f"No default model available for provider: {self.provider}")

    def _load_template(self) -> Template:
        """Load prompt template from file."""
        if not self.template_name:
            raise ValueError("template_name is required for template loading")
            
        template_path = Path(config.templates_dir) / self.template_name

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, encoding='utf-8') as f:
            template_content = f.read()

        return Template(template_content)

    def _create_client(self) -> openai.OpenAI | Any:
        """Create the appropriate client based on provider."""
        # Start with custom client params, then add config params
        base_params = self.custom_client_params.copy()
        config_params = self._get_client_params_from_config()

        # Merge config params with custom params (custom params take precedence)
        for key, value in config_params.items():
            base_params.setdefault(key, value)

        if self.provider.lower() == "openai":
            # Check if this is Azure OpenAI based on the config params
            if "azure_endpoint" in base_params and "api_version" in base_params:
                # Azure OpenAI configuration
                try:
                    from openai import AzureOpenAI
                    agent_logger.info(f"Creating Azure OpenAI client for agent '{self.name}' with endpoint: {base_params.get('azure_endpoint')}, API version: {base_params.get('api_version')}")
                    client = AzureOpenAI(**base_params)

                    # Test the client with a simple call to verify it works
                    try:
                        # This is a minimal test that shouldn't cost tokens
                        client.models.list()
                        return client
                    except Exception as e:
                        agent_logger.warning(f"Azure OpenAI client test failed: {e}. Falling back to regular OpenAI.")
                        # Fall through to the fallback logic below

                except ImportError:
                    agent_logger.warning("AzureOpenAI not available, falling back to regular OpenAI client")

                # Fallback: Get regular OpenAI config and create standard client
                try:
                    fallback_config = config.get_llm_config(self.provider, force_fallback=True)
                    fallback_params = self.custom_client_params.copy()
                    if fallback_config.get("config_list"):
                        fallback_config_item = fallback_config["config_list"][0]
                        for key, value in fallback_config_item.items():
                            if key != "model":
                                fallback_params.setdefault(key, value)

                    agent_logger.info(f"Creating fallback OpenAI client for agent '{self.name}' with base URL: {fallback_params.get('base_url', 'https://api.openai.com/v1')}")
                    return openai.OpenAI(**fallback_params)
                except Exception as e:
                    agent_logger.error(f"Fallback OpenAI client creation failed: {e}")
                    # Final fallback with minimal config
                    minimal_params = {
                        "api_key": config.openai_api_key,
                        "base_url": "https://api.openai.com/v1"
                    }
                    agent_logger.info(f"Creating minimal OpenAI client for agent '{self.name}'")
                    return openai.OpenAI(**minimal_params)
            else:
                # Regular OpenAI configuration
                agent_logger.info(f"Creating OpenAI client for agent '{self.name}' with base URL: {base_params.get('base_url', 'https://api.openai.com/v1')}")
                return openai.OpenAI(**base_params)
        elif self.provider.lower() == "deepseek":
            agent_logger.info(f"Creating DeepSeek client for agent '{self.name}' with base URL: {base_params.get('base_url')}")
            return openai.OpenAI(**base_params)
        elif self.provider.lower() == "anthropic":
            # Note: This would require anthropic client, but using OpenAI-compatible interface for now
            anthropic_base = base_params.get("base_url", "")
            agent_logger.info(f"Creating Anthropic client for agent '{self.name}' with base URL: {anthropic_base}")
            return openai.OpenAI(**base_params)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def update_model_config(self, **kwargs: Any) -> None:
        """Update model configuration parameters.
        
        Args:
            **kwargs: Model configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
            else:
                raise ValueError(f"Invalid model configuration parameter: {key}")

        agent_logger.info(f"Updated model config for agent '{self.name}': {kwargs}")

    def get_model_info(self) -> dict[str, Any]:
        """Get current model information.
        
        Returns:
            Dictionary containing model and provider information.
        """
        return {
            "agent_name": self.name,
            "provider": self.provider,
            "model_name": self.model_config.model_name,
            "model_config": self.model_config.to_dict(),
            "system_message": self.system_message
        }

    def render_prompt(self, **kwargs: Any) -> str:
        """Render the prompt template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template.
            
        Returns:
            Rendered prompt string.
        """
        return self.template.render(**kwargs)

    def load_data_file(self, filename: str, **kwargs) -> dict[str, Any]:
        """Load a data file from the data directory.
        
        Args:
            filename: Name of the file to load.
            **kwargs: Additional parameters for file processing.
            
        Returns:
            Dictionary containing file data and metadata.
        """
        file_path = self.data_dir / filename
        return FileProcessor.process_file(str(file_path), **kwargs)

    def list_data_files(self, pattern: str = "*") -> list[dict[str, Any]]:
        """List available data files in the data directory.
        
        Args:
            pattern: File pattern to match (e.g., "*.csv", "*.pdf").
            
        Returns:
            List of file information dictionaries.
        """
        files = []
        for file_path in self.data_dir.glob(pattern):
            if file_path.is_file():
                files.append(FileProcessor.get_file_info(str(file_path)))
        return files

    def process_multiple_files(self, filenames: list[str], **kwargs) -> dict[str, Any]:
        """Process multiple data files.
        
        Args:
            filenames: List of filenames to process.
            **kwargs: Additional parameters for file processing.
            
        Returns:
            Dictionary mapping filenames to their processed data.
        """
        results = {}
        for filename in filenames:
            try:
                results[filename] = self.load_data_file(filename, **kwargs)
            except Exception as e:
                results[filename] = {"error": str(e), "type": "error"}
        return results

    def get_data_summary(self, filename: str) -> str:
        """Get a text summary of a data file suitable for LLM processing.
        
        Args:
            filename: Name of the file to summarize.
            
        Returns:
            Text summary of the file contents.
        """
        try:
            file_data = self.load_data_file(filename)

            if file_data.get("type") == "error":
                return f"Error processing file {filename}: {file_data.get('error', 'Unknown error')}"

            summary = f"File: {filename}\n"
            summary += f"Size: {file_data['file_info']['size']} bytes\n"
            summary += f"Type: {file_data['file_info']['extension']}\n\n"

            if file_data.get("type") == "text":
                content = file_data.get("content", "")
                if len(content) > 2000:
                    summary += f"Content preview (first 2000 chars):\n{content[:2000]}...\n"
                else:
                    summary += f"Content:\n{content}\n"

            elif file_data.get("type") == "dataframe":
                df = file_data.get("data")

                # Handle case where Excel file returns dict of DataFrames (multiple sheets)
                if isinstance(df, dict):
                    summary += "Excel file with multiple sheets:\n"
                    for sheet_name, sheet_df in df.items():
                        summary += f"\nSheet '{sheet_name}':\n"
                        summary += f"  Shape: {sheet_df.shape[0]} rows × {sheet_df.shape[1]} columns\n"
                        summary += f"  Columns: {', '.join(sheet_df.columns.tolist())}\n"
                        summary += f"  First 3 rows:\n{sheet_df.head(3).to_string()}\n"
                elif hasattr(df, 'shape'):  # Single DataFrame
                    summary += f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    summary += f"Columns: {', '.join(df.columns.tolist())}\n"
                    summary += f"Data types:\n{df.dtypes.to_string()}\n\n"
                    summary += f"First 5 rows:\n{df.head().to_string()}\n"
                    if df.shape[0] > 5:
                        summary += f"\nStatistical summary:\n{df.describe().to_string()}\n"
                else:
                    summary += f"Data type: {type(df)}\n"
                    summary += f"Data preview: {str(df)[:500]}\n"

            return summary

        except Exception as e:
            return f"Error getting summary for {filename}: {str(e)}"

    def prepare_task(self, task_data: dict[str, Any]) -> str:
        """Prepare the task prompt from input data.
        
        This method can be overridden by subclasses. For template-based agents,
        this should use self.template. For non-template agents (like AutoGen),
        this can be implemented differently or raise NotImplementedError.
        
        Args:
            task_data: Input data for the task.
            
        Returns:
            Prepared prompt string.
        """
        # Handle AutoGen agent types
        if self.agent_type in ["autogen", "autogen_workflow"]:
            return self._prepare_autogen_task(task_data)
        
        # Handle standard template-based agents
        if not self.use_templates:
            raise NotImplementedError("prepare_task must be implemented by non-template agents")
        
        if not self.template:
            raise ValueError("Template not loaded for template-based agent")
            
        # Default implementation for template-based agents
        return self.template.render(**task_data)

    def _prepare_autogen_task(self, task_data: dict[str, Any]) -> str:
        """Prepare task for AutoGen agents.
        
        Args:
            task_data: Input data containing conversation parameters.
            
        Returns:
            Prepared prompt string for the conversation.
        """
        # For AutoGen agents, the primary input is the conversation message
        message = task_data.get("message", task_data.get("prompt", ""))
        
        if not message:
            return "Please provide a message to start the conversation."
        
        # Add context about the agents if available
        agents_config = task_data.get("agents", self.agents_config)
        if agents_config:
            agent_names = [agent.get("name", "Unknown") for agent in agents_config]
            context = f"Starting conversation with agents: {', '.join(agent_names)}\n\n"
            return context + message
        
        return message

    def execute(self, task_data: dict[str, Any], stream: bool = False) -> dict[str, Any]:
        """Execute the agent task.
        
        Args:
            task_data: Input data for the task.
            stream: Whether to stream the response or return complete response.
            
        Returns:
            Dictionary containing execution results.
        """
        if stream:
            return self.execute_stream(task_data)
        else:
            return self.execute_sync(task_data)

    def execute_sync(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent task synchronously (non-streaming).
        
        Args:
            task_data: Input data for the task.
            
        Returns:
            Dictionary containing execution results.
        """
        agent_logger.log_agent_start(self.name, str(task_data))

        try:
            # Handle AutoGen agent types with their own execution logic
            if self.agent_type == "autogen":
                return self._execute_autogen_sync(task_data)
            elif self.agent_type == "autogen_workflow":
                return self._execute_autogen_workflow_sync(task_data)
            
            # Standard template-based execution for regular agents
            # All agents must implement prepare_task - no exceptions
            prompt = self.prepare_task(task_data)

            # Create messages
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]

            # Prepare API call parameters
            api_params = self.model_config.to_dict()
            api_params["messages"] = messages
            api_params["stream"] = False

            # Call the LLM
            response = self.client.chat.completions.create(**api_params)

            result = {
                "success": True,
                "response": response.choices[0].message.content,
                "agent_name": self.name,
                "provider": self.provider,
                "model_name": self.model_config.model_name,
                "usage": getattr(response, 'usage', None),
                "stream": False
            }

            agent_logger.log_agent_end(self.name, str(task_data), True)
            return result

        except Exception as e:
            agent_logger.error(f"Agent '{self.name}' execution failed", {"error": str(e)})
            agent_logger.log_agent_end(self.name, str(task_data), False)

            return {
                "success": False,
                "error": str(e),
                "agent_name": self.name,
                "provider": self.provider,
                "model_name": getattr(self.model_config, 'model_name', 'unknown'),
                "stream": False
            }

    def execute_stream(self, task_data: dict[str, Any]):
        """Execute the agent task with streaming response.
        
        Args:
            task_data: Input data for the task.
            
        Yields:
            Dictionary chunks containing streaming execution results.
        """
        agent_logger.log_agent_start(self.name, str(task_data))

        try:
            # Handle AutoGen agent types with their own streaming logic
            if self.agent_type == "autogen":
                yield from self._execute_autogen_stream(task_data)
                return
            elif self.agent_type == "autogen_workflow":
                yield from self._execute_autogen_workflow_stream(task_data)
                return
            
            # Standard template-based streaming for regular agents
            # All agents must implement prepare_task - no exceptions
            prompt = self.prepare_task(task_data)

            # Create messages
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]

            # Prepare API call parameters
            api_params = self.model_config.to_dict()
            api_params["messages"] = messages
            api_params["stream"] = True

            # Call the LLM with streaming
            stream = self.client.chat.completions.create(**api_params)

            full_content = ""

            # Yield initial metadata
            yield {
                "success": True,
                "agent_name": self.name,
                "provider": self.provider,
                "model_name": self.model_config.model_name,
                "stream": True,
                "type": "start"
            }

            # Stream the response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content

                    yield {
                        "success": True,
                        "chunk": content,
                        "agent_name": self.name,
                        "provider": self.provider,
                        "model_name": self.model_config.model_name,
                        "stream": True,
                        "type": "chunk"
                    }

            # Yield final result
            yield {
                "success": True,
                "response": full_content,
                "agent_name": self.name,
                "provider": self.provider,
                "model_name": self.model_config.model_name,
                "stream": True,
                "type": "complete"
            }

            agent_logger.log_agent_end(self.name, str(task_data), True)

        except Exception as e:
            agent_logger.error(f"Agent '{self.name}' streaming execution failed", {"error": str(e)})
            agent_logger.log_agent_end(self.name, str(task_data), False)

            yield {
                "success": False,
                "error": str(e),
                "agent_name": self.name,
                "provider": self.provider,
                "model_name": getattr(self.model_config, 'model_name', 'unknown'),
                "stream": True,
                "type": "error"
            }

    def _create_simple_conversation(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Create a simple multi-agent conversation simulation.
        
        This is a simplified version that demonstrates the AutoGen integration concept
        without requiring the full AutoGen model client setup complexity.
        
        Args:
            task_data: Task data containing conversation parameters.
            
        Returns:
            Conversation results.
        """
        try:
            message = task_data.get("message", task_data.get("prompt", "Hello!"))
            agents_config = task_data.get("agents", self.agents_config)
            max_turns = task_data.get("max_turns", self.max_turns)
            
            # Default agent configuration if none provided
            if not agents_config:
                agents_config = [
                    {"type": "user_proxy", "name": "User", "system_message": "You are a user proxy agent."},
                    {"type": "assistant", "name": "Assistant", "system_message": "You are a helpful assistant."}
                ]
            
            conversation = []
            agents_used = []
            
            # Simulate conversation between agents
            current_message = message
            for turn in range(min(max_turns, 4)):  # Limit to 4 turns for simulation
                for i, agent_config in enumerate(agents_config):
                    agent_name = agent_config.get("name", f"Agent_{i}")
                    agent_type = agent_config.get("type", "assistant")
                    
                    if agent_name not in agents_used:
                        agents_used.append(agent_name)
                    
                    # Simulate agent response using the LLM
                    if agent_type == "user_proxy":
                        # User proxy just forwards the message
                        response_content = f"[{agent_name}] {current_message}"
                    else:
                        # Assistant agents generate responses
                        try:
                            agent_system_message = agent_config.get("system_message", "You are a helpful assistant.")
                            messages = [
                                {"role": "system", "content": agent_system_message},
                                {"role": "user", "content": current_message}
                            ]
                            
                            api_params = self.model_config.to_dict()
                            api_params["messages"] = messages
                            api_params["stream"] = False
                            api_params["max_tokens"] = min(200, api_params.get("max_tokens", 200))  # Limit for simulation
                            
                            response = self.client.chat.completions.create(**api_params)
                            response_content = response.choices[0].message.content
                        except Exception as e:
                            response_content = f"[{agent_name}] Error generating response: {str(e)}"
                    
                    conversation.append({
                        "source": agent_name,
                        "content": response_content,
                        "turn": turn,
                        "agent_type": agent_type
                    })
                    
                    current_message = response_content
                    
                    # Break if we have enough conversation
                    if len(conversation) >= max_turns:
                        break
                
                if len(conversation) >= max_turns:
                    break
            
            # Create summary response
            if conversation:
                last_message = conversation[-1]["content"]
                summary = f"Multi-agent conversation completed with {len(agents_used)} agents over {len(conversation)} messages."
            else:
                last_message = "No conversation generated."
                summary = "Conversation simulation failed."
            
            return {
                "success": True,
                "response": last_message,
                "conversation": conversation,
                "summary": summary,
                "total_messages": len(conversation),
                "agents_used": agents_used,
                "agent_type": "autogen_simulation"
            }
            
        except Exception as e:
            agent_logger.error(f"AutoGen conversation simulation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"AutoGen conversation failed: {str(e)}",
                "agent_type": "autogen_simulation"
            }

    def _get_workflow_config(self, workflow_type: str) -> dict[str, Any]:
        """Get predefined workflow configuration for AutoGen workflow agents.
        
        Args:
            workflow_type: Type of workflow ("research", "creative", "problem_solving").
            
        Returns:
            Workflow configuration dictionary.
        """
        workflows = {
            "research": {
                "agents": [
                    {"type": "user_proxy", "name": "Researcher", "system_message": "You are a research coordinator. Ask questions and gather information."},
                    {"type": "assistant", "name": "DataAnalyst", "system_message": "You analyze data and provide insights."},
                    {"type": "assistant", "name": "Writer", "system_message": "You synthesize information into clear reports."}
                ],
                "max_turns": 6
            },
            "creative": {
                "agents": [
                    {"type": "user_proxy", "name": "Client", "system_message": "You represent the client's creative vision."},
                    {"type": "assistant", "name": "Ideator", "system_message": "You generate creative ideas and concepts."},
                    {"type": "assistant", "name": "Critic", "system_message": "You provide constructive feedback and refinements."}
                ],
                "max_turns": 5
            },
            "problem_solving": {
                "agents": [
                    {"type": "user_proxy", "name": "ProblemOwner", "system_message": "You present the problem and requirements."},
                    {"type": "assistant", "name": "Analyzer", "system_message": "You break down problems and identify key issues."},
                    {"type": "assistant", "name": "Solver", "system_message": "You propose solutions and implementation strategies."}
                ],
                "max_turns": 5
            }
        }
        
        return workflows.get(workflow_type, workflows["research"])

    def _execute_autogen_sync(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute AutoGen conversation task synchronously.
        
        Args:
            task_data: Task data containing conversation parameters.
            
        Returns:
            Execution results.
        """
        agent_logger.info("Executing AutoGen conversation task")
        
        try:
            if not AUTOGEN_AVAILABLE:
                return {
                    "success": False,
                    "error": f"AutoGen not available: {AUTOGEN_IMPORT_ERROR}",
                    "response": "AutoGen integration is not fully configured. Please check the installation."
                }
            
            # Use the simplified conversation simulation
            result = self._create_simple_conversation(task_data)
            
            agent_logger.info("AutoGen conversation completed")
            return result
            
        except Exception as e:
            error_msg = f"AutoGen execution failed: {str(e)}"
            agent_logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "response": error_msg
            }

    def _execute_autogen_workflow_sync(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute AutoGen workflow task synchronously.
        
        Args:
            task_data: Task data containing workflow parameters.
            
        Returns:
            Execution results.
        """
        agent_logger.info(f"Executing AutoGen {self.workflow_type} workflow")
        
        # Get workflow configuration
        workflow_type = task_data.get("workflow_type", self.workflow_type)
        workflow_config = self._get_workflow_config(workflow_type)
        
        # Merge with task data
        enhanced_task_data = {**task_data, **workflow_config}
        
        # Execute using the AutoGen conversation logic
        result = self._create_simple_conversation(enhanced_task_data)
        
        # Add workflow metadata
        if result.get("success"):
            result["workflow_type"] = workflow_type
            result["workflow_agent"] = "AutoGenWorkflowAgent"
        
        return result

    def _execute_autogen_stream(self, task_data: dict[str, Any]):
        """Execute AutoGen conversation with streaming.
        
        Args:
            task_data: Task data containing conversation parameters.
            
        Yields:
            Streaming execution results.
        """
        agent_logger.info("Starting AutoGen streaming conversation")
        
        try:
            if not AUTOGEN_AVAILABLE:
                yield {
                    "success": False,
                    "type": "error",
                    "error": f"AutoGen not available: {AUTOGEN_IMPORT_ERROR}",
                    "stream": True
                }
                return
            
            # Yield start event
            yield {
                "success": True,
                "type": "start",
                "message": "Starting AutoGen conversation simulation",
                "stream": True
            }
            
            # Get configuration
            agents_config = task_data.get("agents", self.agents_config)
            if not agents_config:
                agents_config = [
                    {"type": "user_proxy", "name": "User"},
                    {"type": "assistant", "name": "Assistant", "system_message": "You are helpful."}
                ]
            
            # Yield agent creation
            yield {
                "success": True,
                "type": "agents_created",
                "message": f"Created {len(agents_config)} agents",
                "agents": [agent.get("name", "Unknown") for agent in agents_config],
                "stream": True
            }
            
            # Execute the conversation
            result = self._create_simple_conversation(task_data)
            
            if result.get("success"):
                # Stream each message
                for i, message in enumerate(result.get("conversation", [])):
                    yield {
                        "success": True,
                        "type": "message", 
                        "message_index": i,
                        "source": message["source"],
                        "content": message["content"],
                        "stream": True
                    }
                
                # Final result
                yield {
                    "success": True,
                    "type": "complete",
                    "response": result.get("response", ""),
                    "total_messages": result.get("total_messages", 0),
                    "agents_used": result.get("agents_used", []),
                    "stream": True
                }
            else:
                yield {
                    "success": False,
                    "type": "error",
                    "error": result.get("error", "Unknown error"),
                    "stream": True
                }
            
        except Exception as e:
            error_msg = f"AutoGen streaming failed: {str(e)}"
            agent_logger.error(error_msg)
            yield {
                "success": False,
                "type": "error",
                "error": error_msg,
                "stream": True
            }

    def _execute_autogen_workflow_stream(self, task_data: dict[str, Any]):
        """Execute AutoGen workflow with streaming.
        
        Args:
            task_data: Task data containing workflow parameters.
            
        Yields:
            Streaming execution results.
        """
        workflow_type = task_data.get("workflow_type", self.workflow_type)
        workflow_config = self._get_workflow_config(workflow_type)
        enhanced_task_data = {**task_data, **workflow_config}
        
        for chunk in self._execute_autogen_stream(enhanced_task_data):
            chunk["workflow_type"] = workflow_type
            chunk["workflow_agent"] = "AutoGenWorkflowAgent"
            yield chunk
