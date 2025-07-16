# 🧠 Autogen Agent Orchestration Project

A modular and testable Python project that uses the **Autogen API**, supporting both **OpenAI** and **DeepSeek** as language model providers with a FastAPI-based orchestration system.

## 🚀 Features

- **Modular Agent System**: Extensible base agent class with sample implementations
- **Multi-Provider Support**: Compatible with OpenAI and DeepSeek language models
- **HTTP API**: FastAPI-based REST API for agent orchestration
- **Template Management**: Jinja2-based prompt templates stored separately
- **Comprehensive Logging**: Thread-aware logging with execution tracking
- **Configuration Management**: Environment-based configuration with validation
- **Testing Suite**: Unit tests for all major components
- **Modern Python**: Uses `uv` for dependency management and follows PEP8 standards

## 📁 Project Structure

```
autogen_project/
├── agents/                 # Agent logic modules
│   ├── __init__.py
│   ├── base_agent.py      # Base agent class
│   └── sample_agents.py   # Sample agent implementations
├── api/                   # API routes and orchestrator logic
│   ├── __init__.py
│   ├── orchestrator.py    # Task orchestration logic
│   └── app.py            # FastAPI application
├── templates/             # Prompt templates (Jinja2)
│   ├── data_analyst_prompt.txt
│   ├── content_writer_prompt.txt
│   └── code_reviewer_prompt.txt
├── core/                  # Core utilities
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   └── logger.py         # Logging utilities
├── tests/                 # Test suite
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_logger.py
│   └── test_orchestrator.py
├── logs/                  # Default logging output directory
├── .env                   # Environment configuration
├── pyproject.toml         # UV dependency definition
├── main.py               # Application entry point
└── README.md             # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone or create the project directory**:
   ```bash
   cd autogen_project
   ```

2. **Install uv** (if not already installed):
   ```bash
   # On Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create and activate virtual environment**:
   ```bash
   uv venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

5. **Install development dependencies** (optional):
   ```bash
   uv pip install -e ".[dev]"
   ```

### Configuration

1. **Update the `.env` file** with your API keys:
   ```env
   # API Keys
   OPENAI_API_KEY=your_openai_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   
   # Model Provider Configuration
   DEFAULT_PROVIDER=openai
   OPENAI_MODEL=gpt-4
   DEEPSEEK_MODEL=deepseek-chat
   DEEPSEEK_API_BASE=https://api.deepseek.com/v1
   
   # Logging Configuration
   LOG_DIR=logs
   LOG_LEVEL=INFO
   
   # API Configuration
   API_HOST=localhost
   API_PORT=8000
   ```

2. **Ensure the logs directory exists**:
   ```bash
   mkdir -p logs
   ```

## 🏃 Running the Application

### Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative docs**: `http://localhost:8000/redoc`

## 📝 Usage Examples

### 1. List Available Agents

```bash
curl http://localhost:8000/agents
```

### 2. Run a Single Agent

```bash
curl -X POST "http://localhost:8000/run-agent" \
     -H "Content-Type: application/json" \
     -d '{
       "agent_name": "data_analyst",
       "task_data": {
         "data": "Sales data: Q1=100, Q2=150, Q3=200, Q4=180",
         "context": "Quarterly sales analysis for 2024"
       },
       "provider": "openai"
     }'
```

### 3. Run a Multi-Agent Workflow

```bash
curl -X POST "http://localhost:8000/run-workflow" \
     -H "Content-Type: application/json" \
     -d '{
       "workflow": [
         {
           "agent": "data_analyst",
           "task_data": {
             "data": "User engagement metrics: DAU=1000, MAU=25000, retention=75%",
             "context": "Monthly product analytics"
           }
         },
         {
           "agent": "content_writer",
           "task_data": {
             "topic": "Product Performance Report",
             "audience": "stakeholders",
             "content_type": "executive summary"
           }
         }
       ],
       "provider": "deepseek"
     }'
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

## 🔧 Adding New Agents

### 1. Create a New Prompt Template

Create a new file in `templates/` directory:

```text
# templates/my_agent_prompt.txt
You are a specialized AI assistant for {{ task_type }}.

Your task: {{ task_description }}

Context: {{ context if context else "No additional context" }}

Please provide your response below:
```

### 2. Implement the Agent Class

```python
# agents/my_custom_agent.py
from typing import Dict, Any
from .base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    """Agent specialized in custom tasks."""
    
    def __init__(self, provider: str = None) -> None:
        super().__init__(
            name="MyCustomAgent",
            template_name="my_agent_prompt.txt",
            provider=provider
        )
    
    def prepare_task(self, task_data: Dict[str, Any]) -> str:
        """Prepare custom task prompt."""
        return self.render_prompt(
            task_type=task_data.get("task_type", ""),
            task_description=task_data.get("task_description", ""),
            context=task_data.get("context", "")
        )
```

### 3. Register the Agent

```python
# Update api/orchestrator.py
from agents.my_custom_agent import MyCustomAgent

# In TaskOrchestrator.__init__()
self.agents["my_custom"] = MyCustomAgent
```

## 📊 Logging

The application provides comprehensive logging:

- **Agent logs**: Track individual agent execution
- **Orchestrator logs**: Track job orchestration and workflows
- **Thread-aware**: Each log entry includes thread information
- **Configurable**: Log level and directory configurable via `.env`

Log files are created in the `logs/` directory:
- `agent.log`: Agent execution logs
- `orchestrator.log`: Orchestrator and API logs

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed with `uv pip install -e .`
2. **API key errors**: Verify your API keys are correctly set in the `.env` file
3. **Module not found**: Make sure you're running from the project root directory
4. **Port already in use**: Change the `API_PORT` in `.env` or stop other services on port 8000

### Debug Mode

Enable debug logging by setting in `.env`:
```env
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Follow PEP8 style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Use type hints in function signatures
5. Ensure all tests pass before submitting

## 📄 License

This project is open source and available under the MIT License.
