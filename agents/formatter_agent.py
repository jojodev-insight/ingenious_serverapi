"""Simple formatter agent for pipeline testing."""

from agents.base_agent import BaseAgent


class FormatterAgent(BaseAgent):
    """Agent that formats text in specific ways."""

    def __init__(self, provider: str = "deepseek"):
        """Initialize the formatter agent."""
        super().__init__(
            name="Formatter",
            template_name="formatter.txt",
            provider=provider,
            model_name="deepseek-chat"
        )
        self.update_model_config(temperature=0.1)  # Low temperature for consistent formatting

    def prepare_task(self, task_data: dict) -> str:
        """Prepare task prompt for formatting."""
        format_type = task_data.get("format", "json")
        data = task_data.get("data", "")
        template = task_data.get("template", "")

        if template:
            prompt = f"Format this data using the template. Return only the formatted result:\n\nData: {data}\nTemplate: {template}"
        elif format_type == "json":
            prompt = f"Format this data as a JSON object and return only the JSON: {data}"
        elif format_type == "csv":
            prompt = f"Format this data as CSV and return only the CSV: {data}"
        elif format_type == "list":
            prompt = f"Format this data as a comma-separated list and return only the list: {data}"
        elif format_type == "table":
            prompt = f"Format this data as a simple table and return only the table: {data}"
        elif format_type == "sentence":
            prompt = f"Format this data into a single sentence and return only the sentence: {data}"
        elif format_type == "report":
            title = task_data.get("title", "Report")
            prompt = f"Create a brief report with title '{title}' using this data. Return only the formatted report: {data}"
        else:
            # Fallback to task description
            task = task_data.get("task", "")
            prompt = task

        # Add formatting instruction
        prompt += "\n\nIMPORTANT: Return only the formatted result with no additional explanations or commentary."

        return prompt
