"""Example demonstrating the FileDataAnalyst agent with sample data files."""

import asyncio
from agents import FileDataAnalyst, AgentFactory
from api.orchestrator import TaskOrchestrator


def demo_file_processing():
    """Demonstrate file processing capabilities."""
    print("=== File Data Analyst Demo ===\n")
    
    # Create the agent
    agent = FileDataAnalyst()
    print(f"Agent: {agent.name}")
    print(f"Data directory: {agent.data_dir}")
    
    # List available files
    files = agent.list_data_files()
    print(f"\nFound {len(files)} data files:")
    for file_info in files:
        print(f"  - {file_info['name']} ({file_info['extension']}) - {file_info['size']} bytes")
    
    # Get detailed file information
    print("\n=== File Details ===")
    detailed_info = agent.get_available_files_info()
    for filename, info in detailed_info.items():
        print(f"\n{filename}:")
        if "error" in info:
            print(f"  Error: {info['error']}")
        else:
            summary = info["summary"]
            print(f"  {summary[:200]}..." if len(summary) > 200 else f"  {summary}")


def demo_agent_analysis():
    """Demonstrate agent analysis with mocked responses."""
    print("\n=== Agent Analysis Demo ===\n")
    
    agent = FileDataAnalyst()
    
    # Example 1: Analyze sales data
    print("1. Sales Data Analysis Task:")
    task_data = {
        "analysis_request": "Analyze the sales data to identify top-performing products and revenue trends.",
        "files": ["sales_data.csv"]
    }
    
    prompt = agent.prepare_task(task_data)
    print("Generated prompt preview:")
    print(prompt[:500] + "...\n" if len(prompt) > 500 else prompt + "\n")
    
    # Example 2: Employee data analysis
    print("2. Employee Data Analysis Task:")
    task_data = {
        "analysis_request": "Analyze employee data to identify salary patterns and performance correlations.",
        "files": ["employee_data.xlsx"]
    }
    
    prompt = agent.prepare_task(task_data)
    print("Generated prompt preview:")
    print(prompt[:500] + "...\n" if len(prompt) > 500 else prompt + "\n")
    
    # Example 3: Multi-file analysis
    print("3. Multi-file Analysis Task:")
    task_data = {
        "analysis_request": "Compare and correlate insights from all available data sources.",
        "include_all_files": True
    }
    
    prompt = agent.prepare_task(task_data)
    print("Generated prompt preview:")
    print(prompt[:500] + "...\n" if len(prompt) > 500 else prompt + "\n")


def demo_orchestrator_integration():
    """Demonstrate integration with the orchestrator."""
    print("=== Orchestrator Integration Demo ===\n")
    
    # Create orchestrator and register the file data analyst
    orchestrator = TaskOrchestrator()
    orchestrator.register_agent("file_data_analyst", FileDataAnalyst)
    
    print("Registered agents:")
    for agent_name in orchestrator.list_agents():
        print(f"  - {agent_name}")
    
    # Example task for the orchestrator
    print("\nExample orchestrator task:")
    task_data = {
        "analysis_request": "Provide comprehensive analysis of our sales and employee data to identify business insights.",
        "files": ["sales_data.csv", "employee_data.xlsx"],
        "include_summary": True
    }
    
    print("Task data:")
    print(f"  Analysis request: {task_data['analysis_request']}")
    print(f"  Files: {task_data['files']}")
    
    # Note: We're not actually executing this since it would require API keys
    print("\n(Task prepared for execution - would require valid API keys to run)")


def demo_agent_factory_integration():
    """Demonstrate AgentFactory integration."""
    print("\n=== Agent Factory Integration Demo ===\n")
    
    # List available agent types
    agent_types = AgentFactory.list_agent_types()
    print("Available agent types:")
    for agent_type in agent_types:
        print(f"  - {agent_type}")
    
    # Create file data analyst through factory
    agent = AgentFactory.create_agent(
        "file_data_analyst",
        provider="deepseek",
        model_name="deepseek-chat"
    )
    
    print(f"\nCreated agent via factory:")
    print(f"  Name: {agent.name}")
    print(f"  Provider: {agent.provider}")
    print(f"  Model: {agent.model_config.model_name}")
    
    # Test convenience methods
    print("\nTesting convenience methods:")
    print("- analyze_sales_data() available")
    print("- analyze_employee_data() available") 
    print("- compare_multiple_datasets() available")


if __name__ == "__main__":
    try:
        demo_file_processing()
        demo_agent_analysis()
        demo_orchestrator_integration()
        demo_agent_factory_integration()
        
        print("\n=== Demo Complete ===")
        print("The FileDataAnalyst agent is ready to process your data files!")
        print("\nTo use with real API calls, ensure you have valid API keys set in your environment:")
        print("- DEEPSEEK_API_KEY for DeepSeek")
        print("- OPENAI_API_KEY for OpenAI")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
