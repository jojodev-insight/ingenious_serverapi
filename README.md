# 🧠 Ingenious Lite: Simple Autogen Agent Orchestration Project

A modular and testable Python project that uses the **Autogen API**, supporting both **OpenAI** and **DeepSeek** as language model providers with a FastAPI-based orchestration system. The focus of this project is to make prototyping software testable and highly accessible via API or programmatic calls. It's structured to keep invocation simple as we navigate through complex workflows.

The main objective is to make the creation and chaining of agents and fetching the output highly visible and testable through prototype creation. You only need to specify the API key and model name to get this running.


## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.12 or higher** (Python 3.13+ recommended for best compatibility)
- **Git** (for cloning the repository)
- **[uv](https://github.com/astral-sh/uv)** package manager (cross-platform, recommended)

### Installation (All Platforms)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd autogen_project
   ```
   
   *If you're working with a local copy, navigate to the project directory:*
   ```bash
   cd path/to/autogen_project
   ```

2. **Verify Python version:**
   ```bash
   python --version
   # Should show Python 3.12.x or higher (3.13+ recommended)
   ```

3. **Install uv** (if not already installed):

   - **Windows (PowerShell):**
     ```powershell
     irm https://astral.sh/uv/install.ps1 | iex
     ```

   - **macOS/Linux (bash/zsh):**
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

   - **Arch Linux (pacman):**
     ```bash
     sudo pacman -S uv
     ```

   - **Homebrew (macOS/Linux):**
     ```bash
     brew install uv
     ```

   See the [uv installation guide](https://github.com/astral-sh/uv#installation) for more options.

4. **Verify uv installation:**
   ```bash
   uv --version
   # Should show uv version
   ```

5. **Install dependencies and create virtual environment:**
   ```bash
   # This creates a virtual environment and installs all dependencies
   uv sync
   ```

6. **Install development dependencies** (optional, for testing/linting):
   ```bash
   uv sync --extra dev
   ```

### Configuration

1. **Create your environment configuration:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # On Windows (Command Prompt)
   copy .env.example .env
   
   # On Windows (PowerShell)
   Copy-Item .env.example .env
   ```

2. **Update the `.env` file** with your API keys and settings:

   Open the `.env` file in your text editor and configure the following:

   ```env
   # ====================
   # API KEYS (Required)
   # ====================
   # Get your DeepSeek API key from: https://platform.deepseek.com/api_keys
   DEEPSEEK_API_KEY=your_deepseek_api_key_here

   # For Azure OpenAI, get your key from Azure Portal
   # For regular OpenAI, get your key from: https://platform.openai.com/api-keys
   OPENAI_API_KEY=your_openai_api_key_here

   # ====================
   # OPENAI/AZURE OPENAI CONFIGURATION
   # ====================
   # For Azure OpenAI: Your Azure OpenAI endpoint URL
   # For regular OpenAI: Leave blank or set to https://api.openai.com/v1
   OPENAI_API_BASE=https://your-resource-name.openai.azure.com/

   # Model name to use (e.g., gpt-4, gpt-3.5-turbo, gpt-4o)
   OPENAI_MODEL=gpt-4

   # Azure-specific settings (only needed for Azure OpenAI)
   AZURE_DEPLOYMENT_NAME=your_deployment_name
   AZURE_API_VERSION=2024-12-01-preview

   # ====================
   # PROVIDER SETTINGS
   # ====================
   # Default LLM provider: "openai" or "deepseek"
   DEFAULT_PROVIDER=openai
   ```

3. **Verify required directories exist:**
   ```bash
   # These directories should already exist, but verify:
   ls -la logs/     # Should contain log files
   ls -la data/     # Should contain sample data files
   ls -la templates/ # Should contain prompt templates
   
   # On Windows:
   dir logs
   dir data
   dir templates
   
   # Verify sample data files are present:
   ls data/employee_data.xlsx data/sales_data.csv data/market_report.txt
   # On Windows: dir data\employee_data.xlsx data\sales_data.csv data\market_report.txt
   ```

4. **Test the installation:**
   ```bash
   # Test configuration loading
   uv run python -c "from core import config; print('Configuration loaded successfully')"
   
   # Test agent creation
   uv run python -c "from agents import DataAnalystAgent; print('Agents loaded successfully')"
   
   # Test API dependencies
   uv run python -c "from api.app import app; print('API loaded successfully')"
   ```

### Verification

1. **Run quick functionality tests:**
   ```bash
   # Test configuration and agents
   uv run python -c "
   from api.orchestrator import TaskOrchestrator
   orch = TaskOrchestrator()
   print('✅ Available agents:', orch.list_agents())
   print('✅ Configuration loaded successfully')
   "
   ```

2. **Test with sample data:**
   ```bash
   # Run a simple demo to verify everything works
   uv run examples/file_data_analyst_demo.py
   ```

3. **Start the API server (optional test):**
   ```bash
   uv run main.py
   ```
   
   Then in another terminal:
   ```bash
   curl http://localhost:8000/health
   # Should return: {"status": "healthy", "timestamp": "..."}
   
   curl http://localhost:8000/agents
   # Should return list of available agents
   ```

4. **Run the streaming demo:**
   ```bash
   uv run advanced_streaming_demo.py
   ```

### Troubleshooting Installation

**Common Issues:**

1. **Python version too old:**
   ```bash
   # Check your Python version
   python --version
   # If < 3.12, install Python 3.12+ from python.org
   # Python 3.13+ is recommended for best compatibility
   ```

2. **uv command not found:**
   ```bash
   # Restart your terminal after installing uv
   # Or add uv to your PATH manually
   ```

3. **Permission errors (Linux/macOS):**
   ```bash
   # Use user install for uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Then restart your terminal
   ```

4. **Virtual environment issues:**
   ```bash
   # Remove and recreate the virtual environment
   rm -rf .venv  # On Windows: rmdir /s .venv
   uv sync
   ```

5. **Missing API keys:**
   ```bash
   # Verify your .env file exists and has the required keys
   cat .env  # On Windows: type .env
   ```

## 🚀 Quick Start

### Option 1: API Usage (Recommended for external integrations)

1. **Start the server (using uv):**
   ```bash
   uv run main.py
   ```

2. **Test with a simple API call (using uv):**
   ```bash
   uv run curl_equivalent.py
   ```

3. **View API documentation:**
   Open `http://localhost:8000/docs` in your browser


### Option 2: Local Python Usage (Recommended for Python applications)

1. **Run the demo (using uv):**
   ```bash
   uv run examples/file_data_analyst_demo.py
   ```

2. **Use in your code:**
   ```python
   from agents import FileDataAnalyst
   
   agent = FileDataAnalyst(provider="openai")
   result = agent.execute({
       "analysis_request": "Analyze this data",
       "files": ["your_data.csv"]
   })
   ```

### Option 3: AutoGen Multi-Agent Conversations 🆕

1. **Run AutoGen demos:**
   ```bash
   uv run autogen_demo.py
   uv run advanced_streaming_demo.py
   ```

2. **Use AutoGen in your code:**
   ```python
   from api.orchestrator import TaskOrchestrator
   
   orchestrator = TaskOrchestrator()
   result = orchestrator.run_agent("autogen", {
       "message": "Explain quantum computing",
       "agents": [
           {"type": "user_proxy", "name": "Student"},
           {"type": "assistant", "name": "Expert", 
            "system_message": "You are a quantum computing expert."}
       ]
   })
   ```

3. **📖 For detailed AutoGen usage, see [AUTOGEN_INTEGRATION.md](AUTOGEN_INTEGRATION.md)**

## 🚀 Features

- **Modular Agent System**: Extensible base agent class with sample implementations
- **Multi-Provider Support**: Compatible with OpenAI and DeepSeek language models
- **🤖 AutoGen Integration**: Full multi-agent conversation support with AutoGen
- **HTTP API**: FastAPI-based REST API for agent orchestration
- **Template Management**: Jinja2-based prompt templates stored separately
- **Comprehensive Logging**: Thread-aware logging with execution tracking
- **Configuration Management**: Environment-based configuration with validation
- **Testing Suite**: Unit tests for all major components
- **Streaming Support**: Real-time streaming for long-running tasks and conversations
- **Modern Python**: Uses `uv` for dependency management and follows PEP8 standards

### 🆕 AutoGen Integration Features

- **Multi-Agent Conversations**: Create sophisticated conversations between multiple specialized agents
- **Specialized Workflows**: Pre-built workflows for research, code review, and brainstorming
- **Real-Time Streaming**: Watch multi-agent conversations unfold in real-time
- **Seamless Integration**: Works with existing orchestrator and API without breaking changes
- **Provider Flexibility**: Compatible with OpenAI and includes automatic DeepSeek fallback

## 📁 Project Structure

```
autogen_project/
├── agents/                      # Agent logic modules
│   ├── __init__.py             # Agent module exports
│   ├── agent_factory.py        # Factory for creating agents
│   ├── base_agent.py           # Base agent class
│   ├── calculator_agent.py     # Calculator/math operations agent
│   ├── file_data_analyst.py    # File analysis agent
│   ├── formatter_agent.py      # Text formatting agent
│   ├── sample_agents.py        # Sample agent implementations
│   ├── summary_agent.py        # Text summarization agent
│   └── text_processor_agent.py # Text processing agent
├── api/                        # API routes and orchestrator logic
│   ├── __init__.py             # API module initialization
│   ├── app.py                  # FastAPI application
│   └── orchestrator.py         # Task orchestration logic
├── core/                       # Core utilities
│   ├── __init__.py             # Core module exports
│   ├── config.py               # Configuration management
│   └── logger.py               # Logging utilities
├── data/                       # Sample data files
│   ├── employee_data.xlsx      # Sample employee dataset
│   ├── market_report.txt       # Sample market analysis text
│   └── sales_data.csv          # Sample sales metrics
├── examples/                   # Example implementations
│   └── file_data_analyst_demo.py # File analysis demo
├── logs/                       # Default logging output directory
│   ├── agent.log               # Agent execution logs
│   └── orchestrator.log        # Orchestrator and API logs
├── templates/                  # Prompt templates (Jinja2)
│   ├── calculator.txt          # Calculator agent prompt
│   ├── code_reviewer_prompt.txt # Code review prompt template
│   ├── content_writer_prompt.txt # Content writing prompt
│   ├── data_analyst_prompt.txt # Data analysis prompt
│   ├── data_analyst_with_files.txt # File analysis prompt
│   ├── formatter.txt           # Text formatting prompt
│   ├── summary_prompt.txt      # Summarization prompt
│   └── text_processor.txt      # Text processing prompt
├── tests/                      # Test suite
│   ├── __init__.py             # Test package initialization
│   ├── conftest.py             # Pytest configuration
│   ├── test_config.py          # Configuration tests
│   ├── test_enhanced_agents.py # Enhanced agent tests
│   ├── test_enhanced_orchestrator.py # Enhanced orchestrator tests
│   ├── test_file_processing.py # File processing tests
│   ├── test_logger.py          # Logger tests
│   └── test_orchestrator.py    # Orchestrator tests
├── .env                        # Environment configuration (create from .env.example)
├── .env.example                # Example environment configuration (safe for git)
├── .gitignore                  # Git ignore rules
├── cli.py                      # Command-line interface
├── curl_equivalent.py          # API usage examples (Python)
├── data_pipeline_example.py    # Data pipeline demonstration
├── main.py                     # Application entry point
├── pyproject.toml              # UV dependency definition
├── test_advanced_pipeline.py   # Advanced pipeline tests
├── test_data_pipeline.py       # Data pipeline tests
├── test_literal_pipeline.py    # Literal pipeline tests
├── test_pipeline_agents.py     # Pipeline agent tests
├── uv.lock                     # UV lock file
└── README.md                   # This file
```

## 🏃 Running the Application


### Start the API Server (using uv)

```bash
uv run main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative docs**: `http://localhost:8000/redoc`

## 📝 Usage Examples

There are two main ways to use the agents in this project:
1. **Via HTTP API** - For remote access and integration with other systems
2. **Via Local Python Code** - For direct integration in Python applications

### 🌐 API Usage (HTTP Endpoints)

#### 1. Start the API Server

```bash
uv run main.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

#### 2. List Available Agents

```bash
curl http://localhost:8000/agents
```

#### 3. Run a Single Agent

**Basic Data Analysis:**
```bash
curl -X POST "http://localhost:8000/run-agent" \
     -H "Content-Type: application/json" \
     -d '{
       "agent_name": "data_analyst",
       "task_data": {
         "data": "Sales data: Q1=100, Q2=150, Q3=200, Q4=180",
         "context": "Quarterly sales analysis for 2024"
       },
       "provider": "openai"  # Default to OpenAI, fallback to DeepSeek automatically
     }'
```

**File Data Analysis:**
```bash
curl -X POST "http://localhost:8000/run-agent" \
     -H "Content-Type: application/json" \
     -d '{
       "agent_name": "file_data_analyst",
       "task_data": {
         "analysis_request": "Examine employee performance, salary distribution, and department insights",
         "files": ["employee_data.xlsx"]
       },
       "provider": "openai",  # Default to OpenAI, fallback to DeepSeek automatically
       "model_name": "deepseek-chat",
       "llm_config": {
         "temperature": 0.2,
         "max_tokens": 1200
       }
     }'
```

**Python Equivalent of curl commands:**
```python
# Save this as run_agent_example.py and run with:
#   uv run run_agent_example.py
import requests

url = "http://localhost:8000/run-agent"
headers = {"Content-Type": "application/json"}
data = {
    "agent_name": "file_data_analyst",
    "task_data": {
        "analysis_request": "Analyze sales trends and performance metrics",
        "files": ["sales_data.csv"]
    },
    "provider": "openai",  # Default to OpenAI, fallback to DeepSeek automatically
    "model_name": "gpt-4"
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result['response'])
```

#### 4. Run a Multi-Agent Workflow

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
       "provider": "openai"  # Default to OpenAI, fallback to DeepSeek automatically
     }'
```

#### 5. Advanced Multi-Agent Data Analysis Workflow

**Complex Business Intelligence Pipeline:**
```bash
curl -X POST "http://localhost:8000/run-workflow" \
     -H "Content-Type: application/json" \
     -d '{
       "workflow": [
         {
           "agent": "file_data_analyst",
           "task_data": {
             "analysis_request": "Analyze Q4 financial performance and identify key trends",
             "files": ["sales_data.csv", "employee_data.xlsx"],
             "output_format": "structured_summary"
           }
         },
         {
           "agent": "data_analyst", 
           "task_data": {
             "data": "{{previous_result}}",
             "context": "Create predictive insights for Q1 planning",
             "analysis_type": "forecasting"
           }
         },
         {
           "agent": "content_writer",
           "task_data": {
             "topic": "Q4 Performance & Q1 Forecast Report",
             "audience": "C-suite executives",
             "content_type": "executive briefing",
             "data_source": "{{previous_result}}",
             "include_charts": true
           }
         }
       ],
       "provider": "openai",
       "model_name": "gpt-4",
       "llm_config": {
         "temperature": 0.3,
         "max_tokens": 2000
       }
     }'
```

**Python Equivalent of Advanced Workflow:**
```python
# Save this as advanced_workflow_example.py and run with:
#   uv run advanced_workflow_example.py
import requests
import json

def run_advanced_workflow():
    url = "http://localhost:8000/run-workflow"
    headers = {"Content-Type": "application/json"}
    
    workflow_data = {
        "workflow": [
            {
                "agent": "file_data_analyst",
                "task_data": {
                    "analysis_request": "Analyze Q4 financial performance and identify key trends",
                    "files": ["sales_data.csv", "employee_data.xlsx"],
                    "output_format": "structured_summary"
                }
            },
            {
                "agent": "data_analyst", 
                "task_data": {
                    "data": "{{previous_result}}",
                    "context": "Create predictive insights for Q1 planning",
                    "analysis_type": "forecasting"
                }
            },
            {
                "agent": "content_writer",
                "task_data": {
                    "topic": "Q4 Performance & Q1 Forecast Report",
                    "audience": "C-suite executives",
                    "content_type": "executive briefing",
                    "data_source": "{{previous_result}}",
                    "include_charts": True
                }
            }
        ],
        "provider": "openai",
        "model_name": "gpt-4",
        "llm_config": {
            "temperature": 0.3,
            "max_tokens": 2000
        }
    }
    
    print("🚀 Starting advanced workflow...")
    response = requests.post(url, headers=headers, json=workflow_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Workflow completed successfully!")
        print(f"📊 Total execution time: {result.get('total_execution_time', 'N/A')} seconds")
        print("\n📋 Final Report:")
        print("=" * 60)
        print(result.get('final_result', 'No result available'))
    else:
        print(f"❌ Workflow failed with status: {response.status_code}")
        print(f"Error: {response.text}")

if __name__ == "__main__":
    run_advanced_workflow()
```

#### 6. Streaming Endpoints (Real-time Response)

**Stream Single Agent Response:**
```bash
curl -X POST "http://localhost:8000/run-agent-stream" \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     --no-buffer \
     -d '{
       "agent_name": "content_writer",
       "task_data": {
         "topic": "The Future of AI in Healthcare",
         "style": "informative and engaging",
         "length": "300 words",
         "audience": "medical professionals"
       },
       "provider": "openai"
     }'
```

**Stream Multi-Agent Workflow:**
```bash
curl -X POST "http://localhost:8000/run-workflow-stream" \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     --no-buffer \
     -d '{
       "workflow": [
         {
           "agent": "summary",
           "task_data": {
             "text": "Machine learning is transforming healthcare through predictive analytics, medical imaging, drug discovery, and personalized treatment plans..."
           }
         },
         {
           "agent": "formatter",
           "task_data": {
             "text": "{{previous_result}}",
             "format": "bullet points with emojis"
           }
         }
       ],
       "provider": "openai"
     }'
```

**Python Streaming Client Example:**
```python
# Save this as streaming_client_example.py and run with:
#   uv run streaming_client_example.py
import requests
import json

def stream_agent_response():
    url = "http://localhost:8000/run-agent-stream"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    data = {
        "agent_name": "content_writer",
        "task_data": {
            "topic": "Quantum Computing Breakthroughs",
            "style": "technical but accessible",
            "length": "250 words",
            "audience": "technology enthusiasts"
        },
        "provider": "openai"
    }
    
    print("🚀 Starting streaming request...")
    
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    if response.status_code == 200:
        print("📡 Receiving stream...")
        content_chunks = []
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])  # Remove "data: " prefix
                    
                    if chunk.get("type") == "orchestrator_start":
                        print(f"🚀 Job {chunk['job_id']} started")
                        print(f"🤖 Agent: {chunk['agent_name']}")
                        print(f"⚡ Provider: {chunk['provider']}")
                        print("\n📄 Content:")
                        
                    elif chunk.get("type") == "chunk":
                        print(chunk["chunk"], end="", flush=True)
                        content_chunks.append(chunk["chunk"])
                        
                    elif chunk.get("type") == "complete":
                        print(f"\n\n✅ Stream completed")
                        print(f"📊 Total characters: {len(''.join(content_chunks))}")
                        break
                        
                    elif chunk.get("type") == "stream_error":
                        print(f"❌ Error: {chunk['error']}")
                        break
                        
                except json.JSONDecodeError:
                    pass  # Skip invalid JSON lines
    else:
        print(f"❌ Request failed: {response.status_code}")

if __name__ == "__main__":
    stream_agent_response()
```

### 🐍 Local Python Code Usage

#### 1. Direct Agent Usage

```python
# Save this as direct_agent_example.py and run with:
#   uv run direct_agent_example.py
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
```

#### 2. Using Agent Factory

```python
# Save this as agent_factory_example.py and run with:
#   uv run agent_factory_example.py
from agents import AgentFactory

# List available agent types
print("Available agents:", AgentFactory.list_agent_types())

# Create agent via factory (defaults to OpenAI, falls back to DeepSeek)
agent = AgentFactory.create_agent(
    "file_data_analyst",
    provider="openai",  # Default to OpenAI
    model_name="gpt-4"
)

# Use convenience methods for specific analysis types
sales_analysis = agent.analyze_sales_data("sales_data.csv")
employee_analysis = agent.analyze_employee_data("employee_data.xlsx")
print("Sales analysis:", sales_analysis)
print("Employee analysis:", employee_analysis)
```

#### 3. Using the Orchestrator Locally

```python
# Save this as orchestrator_example.py and run with:
#   uv run orchestrator_example.py
from api.orchestrator import TaskOrchestrator

# Create orchestrator
orchestrator = TaskOrchestrator()

# List available agents
print("Available agents:", orchestrator.list_agents())

# Execute a task directly
task_result = orchestrator.run_agent(
    agent_name="data_analyst",
    task_data={
        "data": "Revenue: Q1=$50k, Q2=$75k, Q3=$90k, Q4=$85k",
        "context": "Annual revenue analysis"
    },
    provider="openai"  # Default to OpenAI, fallback to DeepSeek automatically
)

print("Task result:", task_result)
```

#### 4. Async Usage for Better Performance

```python
# Save this as async_example.py and run with:
#   uv run async_example.py
import asyncio
from api.orchestrator import TaskOrchestrator

async def run_multiple_agents():
    orchestrator = TaskOrchestrator()
    
    # Run multiple agents concurrently using asyncio
    async def run_agent_async(agent_name, task_data, provider):
        # Wrap the sync method in an async function
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: orchestrator.run_agent(
                agent_name=agent_name,
                task_data=task_data,
                provider=provider
            )
        )
    
    tasks = [
        run_agent_async(
            agent_name="data_analyst",
            task_data={"data": "Q1 metrics", "context": "Financial analysis"},
            provider="openai"  # Default to OpenAI, fallback to DeepSeek automatically
        ),
        run_agent_async(
            agent_name="content_writer",
            task_data={"topic": "Q1 Report", "audience": "executives"},
            provider="openai"  # Default to OpenAI, fallback to DeepSeek automatically
        )
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# Run async tasks
results = asyncio.run(run_multiple_agents())
print("Results:", results)
```

#### 5. Custom Agent Development

```python
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
```

#### 6. Advanced Workflow - Programmatic Implementation

```python
# Save this as programmatic_workflow_example.py and run with:
#   uv run programmatic_workflow_example.py
import asyncio
from typing import Dict, Any
from agents import FileDataAnalyst, DataAnalystAgent, ContentWriterAgent
from core.logger import agent_logger

class AdvancedWorkflowOrchestrator:
    """
    Demonstrates advanced workflow implementation by directly managing 
    agent initialization and response handling.
    """
    
    def __init__(self):
        self.workflow_results = []
        self.execution_log = []
    
    async def execute_advanced_workflow(self) -> Dict[str, Any]:
        """
        Execute a complex 3-stage business intelligence workflow:
        1. File Data Analysis → 2. Predictive Analytics → 3. Executive Report
        """
        print("🚀 Starting Advanced Programmatic Workflow...")
        print("=" * 60)
        
        try:
            # Stage 1: File Data Analysis
            stage1_result = await self._stage1_file_analysis()
            
            # Stage 2: Predictive Analytics
            stage2_result = await self._stage2_predictive_analysis(stage1_result)
            
            # Stage 3: Executive Report Generation
            stage3_result = await self._stage3_executive_report(stage2_result)
            
            # Compile final results
            final_report = self._compile_final_report()
            
            print("\n✅ Workflow completed successfully!")
            return final_report
            
        except Exception as e:
            agent_logger.error(f"Workflow failed: {str(e)}")
            print(f"❌ Workflow failed: {str(e)}")
            return {"error": str(e), "completed_stages": len(self.workflow_results)}
    
    async def _stage1_file_analysis(self) -> str:
        """Stage 1: Analyze financial data from multiple files"""
        print("\n📊 Stage 1: File Data Analysis")
        print("-" * 30)
        
        # Initialize File Data Analyst (defaults to OpenAI, falls back to DeepSeek)
        analyst = FileDataAnalyst(provider="openai")
        
        task_data = {
            "analysis_request": "Analyze Q4 financial performance, identify key trends, and provide structured summary",
            "files": ["sales_data.csv", "employee_data.xlsx"],
            "output_format": "structured_json",
            "focus_areas": ["revenue_trends", "cost_analysis", "performance_metrics"]
        }
        
        print(f"🔍 Analyzing files: {task_data['files']}")
        
        # Execute analysis
        result = await asyncio.to_thread(analyst.execute, task_data)
        
        # Log and store result
        self.workflow_results.append({
            "stage": 1,
            "agent": "FileDataAnalyst",
            "task": "Q4 Financial Analysis",
            "result": result,
            "status": "completed"
        })
        
        print(f"✅ Stage 1 completed. Analysis length: {len(result)} characters")
        return result
    
    async def _stage2_predictive_analysis(self, previous_result: str) -> str:
        """Stage 2: Generate predictive insights based on Stage 1 results"""
        print("\n📈 Stage 2: Predictive Analytics")
        print("-" * 30)
        
        # Initialize Data Analyst for forecasting (defaults to OpenAI, falls back to DeepSeek)
        predictor = DataAnalystAgent(provider="openai")
        
        task_data = {
            "data": previous_result,
            "context": "Generate Q1 2025 forecasts and strategic recommendations",
            "analysis_type": "predictive_forecasting",
            "output_requirements": [
                "revenue_forecast",
                "risk_assessment", 
                "growth_opportunities",
                "resource_planning"
            ]
        }
        
        print("🔮 Generating predictive insights...")
        
        # Execute prediction
        result = await asyncio.to_thread(predictor.execute, task_data)
        
        # Log and store result
        self.workflow_results.append({
            "stage": 2,
            "agent": "DataAnalyst",
            "task": "Q1 Forecasting",
            "result": result,
            "status": "completed"
        })
        
        print(f"✅ Stage 2 completed. Forecast length: {len(result)} characters")
        return result
    
    async def _stage3_executive_report(self, forecast_data: str) -> str:
        """Stage 3: Generate executive-ready report"""
        print("\n📋 Stage 3: Executive Report Generation")
        print("-" * 30)
        
        # Initialize Content Writer for executive communication (defaults to OpenAI, falls back to DeepSeek)
        writer = ContentWriterAgent(provider="openai")
        
        # Combine previous results for comprehensive context
        combined_data = {
            "financial_analysis": self.workflow_results[0]["result"],
            "predictive_insights": forecast_data
        }
        
        task_data = {
            "topic": "Q4 2024 Performance Review & Q1 2025 Strategic Outlook",
            "audience": "C-Suite Executives and Board Members",
            "content_type": "executive_briefing",
            "data_sources": combined_data,
            "requirements": [
                "executive_summary",
                "key_findings",
                "strategic_recommendations", 
                "action_items",
                "risk_mitigation"
            ],
            "format": "professional_presentation"
        }
        
        print("📝 Generating executive briefing...")
        
        # Execute report generation
        result = await asyncio.to_thread(writer.execute, task_data)
        
        # Log and store result
        self.workflow_results.append({
            "stage": 3,
            "agent": "ContentWriter",
            "task": "Executive Report",
            "result": result,
            "status": "completed"
        })
        
        print(f"✅ Stage 3 completed. Report length: {len(result)} characters")
        return result
    
    def _compile_final_report(self) -> Dict[str, Any]:
        """Compile comprehensive workflow results"""
        return {
            "workflow_type": "Advanced Business Intelligence Pipeline",
            "total_stages": len(self.workflow_results),
            "execution_summary": {
                "stage_1": "Financial data analysis completed",
                "stage_2": "Predictive forecasting completed", 
                "stage_3": "Executive report generated"
            },
            "final_deliverable": self.workflow_results[-1]["result"],
            "supporting_analysis": {
                "financial_insights": self.workflow_results[0]["result"][:500] + "...",
                "forecast_summary": self.workflow_results[1]["result"][:500] + "..."
            },
            "workflow_metadata": {
                "agents_used": ["FileDataAnalyst", "DataAnalyst", "ContentWriter"],
                "providers": ["openai", "deepseek", "openai"],
                "total_processing_stages": 3
            }
        }

async def main():
    """Main execution function"""
    orchestrator = AdvancedWorkflowOrchestrator()
    
    # Execute the advanced workflow
    final_report = await orchestrator.execute_advanced_workflow()
    
    # Display results
    print("\n" + "=" * 60)
    print("📋 FINAL WORKFLOW REPORT")
    print("=" * 60)
    
    if "error" not in final_report:
        print(f"🎯 Workflow Type: {final_report['workflow_type']}")
        print(f"📊 Stages Completed: {final_report['total_stages']}/3")
        print(f"🤖 Agents Used: {', '.join(final_report['workflow_metadata']['agents_used'])}")
        
        print("\n📈 Executive Summary:")
        print("-" * 30)
        print(final_report['final_deliverable'][:800] + "...")
        
        print("\n🔍 Supporting Analysis Available:")
        print("- Financial Analysis:", len(final_report['supporting_analysis']['financial_insights']))
        print("- Forecast Data:", len(final_report['supporting_analysis']['forecast_summary']))
        
    else:
        print(f"❌ Workflow Error: {final_report['error']}")
        print(f"📊 Completed Stages: {final_report['completed_stages']}/3")

if __name__ == "__main__":
    asyncio.run(main())
```

### 📂 Example Files and Demos

The project includes several example files to help you get started:

#### Root Level Examples

- **`curl_equivalent.py`** - API Usage Example
  A complete Python script that demonstrates how to call the API endpoints programmatically. Includes the advanced 3-stage workflow implementation.
  ```bash
  uv run curl_equivalent.py
  ```

- **`data_pipeline_example.py`** - Pipeline Demo
  Shows how to chain multiple agents together for complex data processing workflows.
  ```bash
  uv run data_pipeline_example.py
  ```

- **`cli.py`** - Command-Line Interface
  Interactive CLI tool for running agents and workflows from the command line.
  ```bash
  uv run cli.py
  ```

#### Examples Directory

- **`examples/file_data_analyst_demo.py`** - Local Usage Example  
  Comprehensive demo showing local agent usage with file analysis capabilities.
  ```bash
  uv run examples/file_data_analyst_demo.py
  ```

#### Test Examples
The project includes comprehensive test files that also serve as usage examples:

- **`test_advanced_pipeline.py`** - Advanced workflow testing
- **`test_data_pipeline.py`** - Data pipeline testing
- **`test_literal_pipeline.py`** - Literal pipeline testing  
- **`test_pipeline_agents.py`** - Pipeline agent testing

Run any test file to see example implementations:
```bash
uv run python test_advanced_pipeline.py
```

> **Note:** All scripts above assume you have completed the setup and activated your uv environment. Always use `uv run ...` to ensure the correct environment and dependencies are used.

#### Sample Data Files
These examples work with the sample data files in the `data/` directory:
- **`employee_data.xlsx`** - Sample employee information with salary, performance scores
- **`sales_data.csv`** - Sample sales metrics with product categories and revenue
- **`market_report.txt`** - Sample market analysis text for natural language processing

## � API vs Local Code: When to Use What?

### Use the HTTP API when:
- ✅ Building web applications or microservices
- ✅ Need to integrate with non-Python systems
- ✅ Want to deploy agents as a service
- ✅ Need to scale horizontally across multiple servers
- ✅ Building REST APIs or webhooks
- ✅ Working with languages other than Python

### Use Local Python Code when:
- ✅ Building Python applications or scripts
- ✅ Need direct access to agent internals
- ✅ Want to avoid network overhead
- ✅ Building Jupyter notebooks or data science workflows
- ✅ Need to customize agent behavior extensively
- ✅ Working in offline environments

### Example Integration Patterns

**Web App with API:**
```javascript
// Frontend JavaScript calling the API
fetch('http://localhost:8000/run-agent', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        agent_name: 'data_analyst',
        task_data: {data: userInput}
    })
})
.then(response => response.json())
.then(result => displayResults(result.response));
```

**Python Data Science Workflow:**
```python
# Save this as data_science_example.py and run with:
#   uv run data_science_example.py
# Direct integration in Jupyter notebook or Python script
from agents import FileDataAnalyst
import pandas as pd

# Load your data
df = pd.read_csv('my_data.csv')

# Analyze with agent
agent = FileDataAnalyst()
insights = agent.execute({
    'analysis_request': 'Find patterns and anomalies',
    'files': ['my_data.csv']
})

# Continue with your analysis
print("Data shape:", df.shape)
print("Insights:", insights)
```

## �🧪 Testing

Run the test suite:

```bash
# Run all tests (using uv)
uv pip run pytest

# Run with coverage
uv pip run pytest --cov=.

# Run specific test file
uv pip run pytest tests/test_config.py

# Run with verbose output
uv pip run pytest -v
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
