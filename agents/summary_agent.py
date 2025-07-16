"""Simple summary agent for providing concise summary reports."""

from agents.base_agent import BaseAgent


class SummaryAgent(BaseAgent):
    """Agent that provides concise summary reports."""

    def __init__(self, provider: str = "deepseek"):
        """Initialize the summary agent."""
        super().__init__(
            name="Summary",
            template_name="summary_prompt.txt",
            provider=provider,
            model_name="deepseek-chat"
        )
        self.update_model_config(temperature=0.5)  # Moderate temperature for summarization

    def prepare_task(self, task_data: dict) -> str:
        """Prepare task prompt for summarization."""
        text = task_data.get("text", "")

        if not text:
            return "No text provided for summarization."

        prompt = f"Please provide a concise summary of the following text:\n\n{text}\n\nIMPORTANT: Keep the summary brief and to the point."

        return prompt
