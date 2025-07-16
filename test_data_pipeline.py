"""
Simple test for the new data pipeline workflow functionality.
"""

from api.orchestrator import TaskOrchestrator

def test_basic_data_pipeline():
    """Test basic data pipeline functionality."""
    
    orchestrator = TaskOrchestrator()
    
    # Simple 2-step workflow
    workflow = [
        {
            "agent": "data_analyst",
            "task_data": {
                "task": "Analyze the number 42 and explain its significance"
            },
            "output_key": "analysis"
        },
        {
            "agent": "content_writer", 
            "task_data": {
                "task": "Write a short summary based on the analysis"
            },
            "input_mapping": {
                "source_analysis": "analysis"
            },
            "output_key": "summary"
        }
    ]
    
    initial_data = {
        "context": "Testing data pipeline",
        "timestamp": "2025-07-15"
    }
    
    print("Testing basic data pipeline workflow...")
    
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        initial_data=initial_data,
        provider="deepseek"
    )
    
    print(f"Success: {result['success']}")
    print(f"Steps completed: {result['completed_steps']}/{result['workflow_steps']}")
    
    if result['success']:
        print("\nStep outputs:")
        for key, value in result.get('step_outputs', {}).items():
            print(f"  {key}: {str(value)[:100]}...")
        
        print("\nFinal data:")
        for key, value in result.get('final_data', {}).items():
            print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    test_basic_data_pipeline()
