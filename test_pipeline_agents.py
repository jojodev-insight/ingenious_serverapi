"""
Test the new pipeline-optimized agents for reliable data passing.
"""

from api.orchestrator import TaskOrchestrator

def test_simple_data_pipeline():
    """Test basic data flow with new agents."""
    
    orchestrator = TaskOrchestrator()
    
    workflow = [
        {
            "agent": "text_processor",
            "task_data": {
                "operation": "echo",
                "text_input": "Revenue: 1000, 2000, 3000, 4000"
            },
            "output_key": "raw_text",
            "transform": "extract_numbers"
        },
        {
            "agent": "calculator",
            "task_data": {
                "operation": "sum"
            },
            "input_mapping": {
                "numbers": "raw_text"
            },
            "output_key": "total"
        },
        {
            "agent": "formatter",
            "task_data": {
                "format": "sentence"
            },
            "input_mapping": {
                "data": "total"
            },
            "output_key": "result_sentence"
        }
    ]
    
    print("🔄 Testing Simple Data Pipeline...")
    print("=" * 40)
    
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        provider="openai"
    )
    
    print(f"✅ Success: {result['success']}")
    print(f"📊 Steps: {result['completed_steps']}/{result['workflow_steps']}")
    
    # Show each step clearly
    print("\n📋 Pipeline Steps:")
    print("-" * 20)
    for key, value in result.get('step_outputs', {}).items():
        print(f"{key}: {repr(value)}")
    
    return result

def test_number_processing_pipeline():
    """Test number processing and calculations."""
    
    orchestrator = TaskOrchestrator()
    
    workflow = [
        {
            "agent": "text_processor",
            "task_data": {
                "operation": "echo",
                "text_input": "Sales data: Jan=100, Feb=200, Mar=300, Apr=400"
            },
            "output_key": "sales_text",
            "transform": "extract_numbers"
        },
        {
            "agent": "calculator",
            "task_data": {
                "operation": "sum"
            },
            "input_mapping": {
                "expression": "sales_text"
            },
            "output_key": "total_sales"
        },
        {
            "agent": "calculator",
            "task_data": {
                "operation": "average"
            },
            "input_mapping": {
                "expression": "sales_text"
            },
            "output_key": "avg_sales"
        },
        {
            "agent": "formatter",
            "task_data": {
                "format": "report",
                "title": "Sales Summary"
            },
            "input_mapping": {
                "data": "total_sales"
            },
            "output_key": "sales_report"
        }
    ]
    
    print("\n🧮 Testing Number Processing Pipeline...")
    print("=" * 45)
    
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        provider="openai"
    )
    
    print(f"✅ Success: {result['success']}")
    
    # Show transformation flow
    print("\n🔄 Data Transformations:")
    print("-" * 25)
    for step in result.get('data_flow', {}).get('steps', []):
        print(f"Step {step['step']}: {step['agent']}")
        if step['transform']:
            print(f"  🔄 Transform: {step['transform']}")
        if step['input_mapping']:
            print(f"  📥 Input: {step['input_mapping']}")
        print(f"  📤 Output: {step['output_key']}")
        print()
    
    # Show results
    print("📊 Results:")
    for key, value in result.get('step_outputs', {}).items():
        print(f"  {key}: {repr(value[:100])}...")
    
    return result

def test_text_formatting_pipeline():
    """Test text processing and formatting."""
    
    orchestrator = TaskOrchestrator()
    
    workflow = [
        {
            "agent": "text_processor",
            "task_data": {
                "operation": "echo",
                "text_input": "apple,banana,cherry,date"
            },
            "output_key": "fruit_list"
        },
        {
            "agent": "text_processor",
            "task_data": {
                "operation": "uppercase"
            },
            "input_mapping": {
                "text_input": "fruit_list"
            },
            "output_key": "upper_fruits"
        },
        {
            "agent": "formatter",
            "task_data": {
                "format": "json"
            },
            "input_mapping": {
                "data": "upper_fruits"
            },
            "output_key": "fruit_json"
        }
    ]
    
    print("\n📝 Testing Text Formatting Pipeline...")
    print("=" * 40)
    
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        provider="openai"
    )
    
    print(f"✅ Success: {result['success']}")
    
    print("\n📋 Step Outputs:")
    for key, value in result.get('step_outputs', {}).items():
        print(f"{key}: {repr(value)}")
    
    return result

def test_error_recovery_pipeline():
    """Test error handling with new agents."""
    
    orchestrator = TaskOrchestrator()
    
    workflow = [
        {
            "agent": "text_processor",
            "task_data": {
                "operation": "echo",
                "text_input": "Good data: 123, 456"
            },
            "output_key": "good_data"
        },
        {
            "agent": "nonexistent_agent",
            "task_data": {
                "task": "This will fail"
            },
            "output_key": "failed_step",
            "on_error": "continue"
        },
        {
            "agent": "formatter",
            "task_data": {
                "format": "sentence"
            },
            "input_mapping": {
                "data": "good_data"
            },
            "output_key": "recovery_result"
        }
    ]
    
    print("\n🚨 Testing Error Recovery Pipeline...")
    print("=" * 40)
    
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        provider="openai"
    )
    
    print(f"📊 Overall Success: {result['success']}")
    print(f"📈 Steps Completed: {result['completed_steps']}/{result['workflow_steps']}")
    
    print("\n📋 Step Results:")
    for i, step_result in enumerate(result['results']):
        status = "✅ Success" if step_result.get('success') else "❌ Failed"
        print(f"  Step {i+1}: {status}")
    
    print("\n📤 Available Outputs:")
    for key, value in result.get('step_outputs', {}).items():
        if value is not None:
            print(f"  {key}: {repr(str(value)[:50])}...")
        else:
            print(f"  {key}: None")
    
    return result

def test_complex_data_pipeline():
    """Test a complex multi-step pipeline."""
    
    orchestrator = TaskOrchestrator()
    
    workflow = [
        {
            "agent": "text_processor",
            "task_data": {
                "operation": "echo",
                "text_input": "Q1: 1000, Q2: 1500, Q3: 2000, Q4: 2500"
            },
            "output_key": "quarterly_data",
            "transform": "extract_numbers"
        },
        {
            "agent": "calculator",
            "task_data": {
                "operation": "sum"
            },
            "input_mapping": {
                "expression": "quarterly_data"
            },
            "output_key": "annual_total"
        },
        {
            "agent": "calculator",
            "task_data": {
                "operation": "average"
            },
            "input_mapping": {
                "expression": "quarterly_data"
            },
            "output_key": "quarterly_avg"
        },
        {
            "agent": "formatter",
            "task_data": {
                "format": "report",
                "title": "Annual Financial Summary"
            },
            "input_mapping": {
                "data": "annual_total"
            },
            "output_key": "financial_report"
        }
    ]
    
    initial_data = {
        "company": "Test Corp",
        "year": "2025"
    }
    
    print("\n🏢 Testing Complex Data Pipeline...")
    print("=" * 40)
    
    result = orchestrator.run_data_pipeline_workflow(
        workflow=workflow,
        initial_data=initial_data,
        provider="openai"
    )
    
    print(f"✅ Success: {result['success']}")
    print(f"⏱️  Time: {result['execution_time']:.2f}s")
    
    # Show complete data flow
    print("\n📊 Complete Data Flow:")
    print("-" * 25)
    print(f"Input: Q1: 1000, Q2: 1500, Q3: 2000, Q4: 2500")
    for key, value in result.get('step_outputs', {}).items():
        print(f"{key}: {repr(value)}")
    
    print(f"\nShared Data: {result.get('final_data', {})}")
    
    return result

if __name__ == "__main__":
    print("🎯 Pipeline-Optimized Agent Tests")
    print("=" * 50)
    
    # Test all pipelines
    result1 = test_simple_data_pipeline()
    result2 = test_number_processing_pipeline()
    result3 = test_text_formatting_pipeline()
    result4 = test_error_recovery_pipeline()
    result5 = test_complex_data_pipeline()
    
    print("\n✅ All pipeline tests completed!")
    
    # Summary
    all_successful = all([
        result1['success'],
        result2['success'], 
        result3['success'],
        result4['completed_steps'] > 1,  # Should complete at least some steps
        result5['success']
    ])
    
    print(f"\n📈 Overall Test Success: {all_successful}")
    print("=" * 50)
