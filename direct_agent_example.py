
from agents import FileDataAnalyst

# Create and configure an agent (defaults to OpenAI, falls back to DeepSeek)
agent = FileDataAnalyst(provider="openai")

# Prepare a task
task_data = {
    "analysis_request": "Analyze sales trends and identify opportunities",
    "files": ["sales_data.csv", "market_report.txt"]
}

# Generate the analysis prompt
prompt = agent.prepare_task(task_data)
print("Generated prompt:", prompt)

# Execute with actual LLM (requires API keys)
try:
    response = agent.execute(task_data)
    print("Analysis result:", response)
except Exception as e:
    print(f"Execution failed: {e}")