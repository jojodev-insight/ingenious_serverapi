# Save this as custom_agent_example.py and run with:
#   uv run custom_agent_example.py
from agents.base_agent import BaseAgent
from typing import Dict, Any

class MyCustomAgent(BaseAgent):
    def __init__(self, provider: str = None):
        super().__init__(
            name="MyCustomAgent",
            template_name="my_custom_prompt.txt",
            provider=provider
        )
    
    def prepare_task(self, task_data: Dict[str, Any]) -> str:
        return self.render_prompt(
            task_type=task_data.get("task_type", ""),
            requirements=task_data.get("requirements", ""),
            context=task_data.get("context", "")
        )

# Register and use custom agent
from api.orchestrator import TaskOrchestrator

orchestrator = TaskOrchestrator()
orchestrator.register_agent("my_custom", MyCustomAgent)

# Use your custom agent
result = orchestrator.run_agent(
    "my_custom",
    {
        "task_type": "analysis",
        "requirements": "Detailed breakdown needed",
        "context": "Business process optimization"
    }
)
print("Custom agent result:", result)