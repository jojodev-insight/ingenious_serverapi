# 🧠 Autogen Agent Orchestration Project

A modular and testable Python project that uses the **Autogen API**, supporting both **OpenAI** and **DeepSeek** as language model providers with a FastAPI-based orchestration system.


## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager (cross-platform, recommended)

### Installation (All Platforms)

1. **Clone or create the project directory:**
   ```bash
   cd autogen_project
   ```

2. **Install uv** (if not already installed):

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


3. **Create and activate a virtual environment (all OSes):**
   ```bash
   uv venv
   # On Windows (cmd or PowerShell)
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies (using uv):**
   ```bash
   uv pip install -e .
   ```

5. **Install development dependencies** (optional, for testing/linting):
   ```bash
   uv pip install -e ".[dev]"
   ```

### Configuration

1. **Update the `.env` file** with your API keys and settings:
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
  2. **Create the logs directory:**
    ```bash
    # On macOS/Linux
    mkdir -p logs
    
    # On Windows (PowerShell)
    New-Item -ItemType Directory -Force -Path logs
    
    # On Windows (Command Prompt)
    if not exist logs mkdir logs
    ```

## �🚀 Quick Start


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
       "provider": "openai"
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
       "provider": "deepseek",
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
    "provider": "openai",
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
       "provider": "deepseek"
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

### 🐍 Local Python Code Usage

#### 1. Direct Agent Usage

```python
# Save this as direct_agent_example.py and run with:
#   uv run direct_agent_example.py
from agents import FileDataAnalyst

# Create and configure an agent
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

# Create agent via factory
agent = AgentFactory.create_agent(
    "file_data_analyst",
    provider="deepseek",
    model_name="deepseek-chat"
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
task_result = orchestrator.execute_agent_task(
    agent_name="data_analyst",
    task_data={
        "data": "Revenue: Q1=$50k, Q2=$75k, Q3=$90k, Q4=$85k",
        "context": "Annual revenue analysis"
    },
    provider="openai"
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
    
    # Run multiple agents concurrently
    tasks = [
        orchestrator.execute_agent_task(
            "data_analyst", 
            {"data": "Q1 metrics", "context": "Financial analysis"},
            "openai"
        ),
        orchestrator.execute_agent_task(
            "content_writer",
            {"topic": "Q1 Report", "audience": "executives"},
            "deepseek"
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
result = orchestrator.execute_agent_task(
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
from agents import FileDataAnalyst, DataAnalyst, ContentWriter
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
        
        # Initialize File Data Analyst
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
        
        # Initialize Data Analyst for forecasting
        predictor = DataAnalyst(provider="deepseek")
        
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
        
        # Initialize Content Writer for executive communication
        writer = ContentWriter(provider="openai")
        
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


#### `curl_equivalent.py` - API Usage Example
A complete Python script that demonstrates how to call the API endpoints. Run it directly after setup:
```bash
uv run curl_equivalent.py
```

#### `examples/file_data_analyst_demo.py` - Local Usage Example  
Comprehensive demo showing local agent usage. Run it independently:
```bash
uv run examples/file_data_analyst_demo.py
```

#### `data_pipeline_example.py` - Pipeline Demo
Shows how to chain multiple agents together for complex workflows. Run it independently:
```bash
uv run data_pipeline_example.py
```

> **Note:** All scripts above assume you have completed the setup and activated your uv environment. Always use `uv run ...` to ensure the correct environment and dependencies are used.

These examples work with the sample data files in the `data/` directory:
- `employee_data.xlsx` - Sample employee information
- `sales_data.csv` - Sample sales metrics  
- `market_report.txt` - Sample market analysis text

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
