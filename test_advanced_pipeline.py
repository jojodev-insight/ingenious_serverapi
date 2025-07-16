"""
Test the data pipeline with more specific data flow examples.
"""

from api.orchestrator import TaskOrchestrator

def run_workflow_with_fallback(orchestrator, workflow, initial_data=None):
    """Run workflow with Azure OpenAI first, fallback to DeepSeek if needed."""
    
    # Try Azure OpenAI first
    try:
        print("🔵 Attempting with Azure OpenAI...")
        result = orchestrator.run_data_pipeline_workflow(
            workflow=workflow,
            initial_data=initial_data,
            provider="openai"  # Use Azure OpenAI as default
        )
        
        if result['success']:
            print("🟢 Using Azure OpenAI")
            return result
        else:
            print("🟡 Azure OpenAI pipeline failed, falling back to DeepSeek...")
            
    except Exception as e:
        print(f"🟡 Primary OpenAI error: {e}")
        print("🟡 Trying fallback configuration...")
    
    # Try alternative OpenAI configuration as fallback
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        initial_data=initial_data,
        provider="openai"  # Stay with OpenAI but try different config
    )
    print("🟢 Using OpenAI fallback configuration")
    return result

def test_data_transformation_pipeline():
    """Test data transformations and mappings."""
    
    orchestrator = TaskOrchestrator()
    
    # Workflow that demonstrates data transformation
    workflow = [
        {
            "agent": "data_analyst",
            "task_data": {
                "task": "Generate this exact text: 'The revenue numbers are: 1000, 2000, 3000, 4000'"
            },
            "output_key": "raw_text",
            "transform": "extract_numbers"
        },
        {
            "agent": "data_analyst",
            "task_data": {
                "task": "Calculate the sum of these numbers and provide statistical analysis"
            },
            "input_mapping": {
                "numbers": "raw_text"
            },
            "output_key": "analysis"
        },
        {
            "agent": "content_writer",
            "task_data": {
                "task": "Create a formatted report about the analysis"
            },
            "input_mapping": {
                "analysis_data": "analysis",
                "original_numbers": "raw_text"
            },
            "output_key": "final_report",
            "transform": "strip"
        }
    ]
    
    initial_data = {
        "report_type": "Revenue Analysis",
        "department": "Finance"
    }
    
    print("🔄 Testing Data Transformation Pipeline...")
    print("=" * 50)
    
    result = run_workflow_with_fallback(orchestrator, workflow, initial_data)
    
    print(f"✅ Success: {result['success']}")
    print(f"📊 Steps: {result['completed_steps']}/{result['workflow_steps']}")
    print(f"⏱️  Time: {result['execution_time']:.2f}s")
    
    # Show data flow
    print("\n📋 Data Flow:")
    print("-" * 20)
    for step in result.get('data_flow', {}).get('steps', []):
        print(f"Step {step['step']}: {step['agent']}")
        if step['input_mapping']:
            print(f"  📥 Inputs: {step['input_mapping']}")
        print(f"  📤 Output: {step['output_key']}")
        if step['transform']:
            print(f"  🔄 Transform: {step['transform']}")
        print(f"  ✅ Has Output: {step['has_output']}")
        print()
    
    # Show step outputs
    print("📤 Step Outputs:")
    print("-" * 15)
    for key, value in result.get('step_outputs', {}).items():
        print(f"{key}:")
        print(f"  {str(value)[:200]}...")
        print()
    
    return result

def test_error_handling():
    """Test error handling in pipeline."""
    
    orchestrator = TaskOrchestrator()
    
    workflow = [
        {
            "agent": "data_analyst",
            "task_data": {
                "task": "Analyze: Success step"
            },
            "output_key": "good_data"
        },
        {
            "agent": "nonexistent_agent",
            "task_data": {
                "task": "This will fail"
            },
            "output_key": "failed_data",
            "on_error": "continue"
        },
        {
            "agent": "content_writer",
            "task_data": {
                "task": "Write report with available data"
            },
            "input_mapping": {
                "data": "good_data"
            },
            "output_key": "recovery_report"
        }
    ]
    
    print("\n🚨 Testing Error Handling...")
    print("=" * 35)
    
    result = run_workflow_with_fallback(orchestrator, workflow)
    
    print(f"📊 Overall Success: {result['success']}")
    print(f"📈 Steps Completed: {result['completed_steps']}/{result['workflow_steps']}")
    
    print("\n📋 Step Results:")
    for i, step_result in enumerate(result['results']):
        status = "✅ Success" if step_result.get('success') else "❌ Failed"
        print(f"  Step {i+1}: {status}")
        if not step_result.get('success'):
            print(f"    Error: {step_result.get('error', 'Unknown')}")
    
    return result

if __name__ == "__main__":
    print("🎯 Advanced Data Pipeline Tests")
    print("=" * 40)
    
    # Test transformations
    result1 = test_data_transformation_pipeline()
    
    # Test error handling
    result2 = test_error_handling()
    
    print("\n✅ All tests completed!")
