"""Simple calculator agent for pipeline testing."""

from agents.base_agent import BaseAgent
from core import agent_logger


class CalculatorAgent(BaseAgent):
    """Agent that performs simple mathematical calculations."""
    
    def __init__(self, provider: str = "deepseek"):
        """Initialize the calculator agent."""
        super().__init__(
            name="Calculator",
            template_name="calculator.txt",
            provider=provider,
            model_name="deepseek-chat"
        )
        self.update_model_config(temperature=0.0)  # Zero temperature for mathematical accuracy
    
    def prepare_task(self, task_data: dict) -> str:
        """Prepare task prompt for calculations."""
        operation = task_data.get("operation", "add")
        numbers = task_data.get("numbers", [])
        expression = task_data.get("expression", "")
        
        if expression:
            prompt = f"Calculate this mathematical expression and return only the numerical result: {expression}"
        elif numbers:
            if operation == "add" or operation == "sum":
                nums_str = ", ".join(map(str, numbers))
                prompt = f"Add these numbers and return only the sum: {nums_str}"
            elif operation == "multiply":
                nums_str = ", ".join(map(str, numbers))
                prompt = f"Multiply these numbers and return only the product: {nums_str}"
            elif operation == "average" or operation == "mean":
                nums_str = ", ".join(map(str, numbers))
                prompt = f"Calculate the average of these numbers and return only the result: {nums_str}"
            elif operation == "max":
                nums_str = ", ".join(map(str, numbers))
                prompt = f"Find the maximum of these numbers and return only the result: {nums_str}"
            elif operation == "min":
                nums_str = ", ".join(map(str, numbers))
                prompt = f"Find the minimum of these numbers and return only the result: {nums_str}"
            else:
                nums_str = ", ".join(map(str, numbers))
                prompt = f"Perform {operation} on these numbers: {nums_str}"
        else:
            # Fallback to task description
            task = task_data.get("task", "")
            prompt = task
        
        # Add strict mathematical instruction
        prompt += "\n\nIMPORTANT: Return only the numerical result. No explanations, no text, just the number."
        
        return prompt
