"""Simple text processor agent for pipeline testing."""

from agents.base_agent import BaseAgent
from core import agent_logger


class TextProcessorAgent(BaseAgent):
    """Agent that performs simple text processing operations."""
    
    def __init__(self, provider: str = "deepseek"):
        """Initialize the text processor agent."""
        super().__init__(
            name="TextProcessor",
            template_name="text_processor.txt",
            provider=provider,
            model_name="deepseek-chat"
        )
        self.update_model_config(temperature=0.1)  # Very low temperature for consistency
    
    def prepare_task(self, task_data: dict) -> str:
        """Prepare task prompt for text processing."""
        task = task_data.get("task", "")
        operation = task_data.get("operation", "echo")
        text_input = task_data.get("text_input", "")
        
        # Handle different operations
        if operation == "echo":
            prompt = f"Return exactly this text with no additional words or formatting: {text_input}"
        elif operation == "uppercase":
            prompt = f"Convert this text to uppercase and return only the result: {text_input}"
        elif operation == "lowercase":
            prompt = f"Convert this text to lowercase and return only the result: {text_input}"
        elif operation == "extract_numbers":
            prompt = f"Extract only the numbers from this text and return them as a comma-separated list: {text_input}"
        elif operation == "count_words":
            prompt = f"Count the words in this text and return only the number: {text_input}"
        else:
            # Default to the task description
            prompt = task
        
        # Add any additional context
        if "context" in task_data:
            prompt += f"\n\nContext: {task_data['context']}"
        
        # Add strict instruction
        prompt += "\n\nIMPORTANT: Return only the requested result with no explanations, analysis, or additional text."
        
        return prompt
