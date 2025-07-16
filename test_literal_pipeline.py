"""
Test with more explicit instructions to ensure agents follow exact requirements.
"""

from api.orchestrator import TaskOrchestrator

def test_explicit_pipeline():
    """Test with very explicit instructions."""
    
    orchestrator = TaskOrchestrator()
    
    workflow = [
        {
            "agent": "data_analyst",
            "task_data": {
                "task": "IMPORTANT: Do not analyze anything. Simply return this exact text with no additional commentary: 'Revenue data: 1000, 2000, 3000, 4000'"
            },
            "output_key": "raw_text",
            "transform": "extract_numbers"
        },
        {
            "agent": "data_analyst", 
            "task_data": {
                "task": "You received some text with numbers. Calculate the sum of ALL the numbers you find in the input_data."
            },
            "input_mapping": {
                "input_data": "raw_text"
            },
            "output_key": "sum_result"
        }
    ]
    
    print("🎯 Testing Explicit Pipeline...")
    print("=" * 40)
    
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        provider="openai"
    )
    
    print(f"Success: {result['success']}")
    
    # Show what actually happened
    print("\n📝 Step by Step:")
    for i, (key, value) in enumerate(result.get('step_outputs', {}).items()):
        print(f"Step {i+1} ({key}):")
        print(f"  Output: {repr(value[:100])}...")
        print()
    
    return result

def test_content_writer_literal():
    """Test if content writer follows literal instructions better."""
    
    orchestrator = TaskOrchestrator()
    
    workflow = [
        {
            "agent": "content_writer",
            "task_data": {
                "task": "Write exactly this sentence and nothing else: 'The numbers are 100, 200, 300, 400'"
            },
            "output_key": "literal_text",
            "transform": "extract_numbers"
        },
        {
            "agent": "content_writer",
            "task_data": {
                "task": "Write a sentence that includes these numbers"
            },
            "input_mapping": {
                "numbers": "literal_text"
            },
            "output_key": "number_sentence"
        }
    ]
    
    print("\n✍️ Testing Content Writer Literal...")
    print("=" * 40)
    
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        provider="openai"
    )
    
    print(f"Success: {result['success']}")
    
    # Show results
    print("\n📝 Results:")
    for key, value in result.get('step_outputs', {}).items():
        print(f"{key}: {repr(value[:100])}...")
        print()
    
    return result

if __name__ == "__main__":
    test_explicit_pipeline()
    test_content_writer_literal()
