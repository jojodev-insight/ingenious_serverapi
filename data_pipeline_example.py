"""
Example demonstrating the enhanced data pipeline workflow with data passing between agents.
"""

from api.orchestrator import TaskOrchestrator


def example_data_pipeline():
    """Demonstrate data pipeline workflow with multi-agent data passing."""

    orchestrator = TaskOrchestrator()

    # Example 1: Analysis Pipeline with Data Flow
    analysis_workflow = [
        {
            "agent": "file_data_analyst",
            "task_data": {
                "task": "Load and analyze employee data",
                "files": ["employee_data.xlsx"]
            },
            "output_key": "analysis_results",
            "transform": "strip"
        },
        {
            "agent": "data_analyst",
            "task_data": {
                "task": "Create statistical summary from previous analysis"
            },
            "input_mapping": {
                "previous_analysis": "analysis_results"
            },
            "output_key": "statistical_summary"
        },
        {
            "agent": "content_writer",
            "task_data": {
                "task": "Write executive summary report"
            },
            "input_mapping": {
                "data_analysis": "analysis_results",
                "statistics": "statistical_summary"
            },
            "output_key": "final_report",
            "transform": "uppercase"
        }
    ]

    # Initial data to seed the workflow
    initial_data = {
        "company": "Acme Corp",
        "report_date": "2025-07-15",
        "department": "HR Analytics"
    }

    # Global data mapping between steps
    data_mapping = {
        "report_metadata": "job_id",
        "execution_info": "execution_time"
    }

    print("🚀 Starting Data Pipeline Workflow...")
    print("=" * 60)

    result = orchestrator.run_data_pipeline_workflow(
        workflow=analysis_workflow,
        initial_data=initial_data,
        provider="deepseek",
        data_mapping=data_mapping
    )

    print(f"📊 Workflow Success: {result['success']}")
    print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
    print(f"📈 Steps Completed: {result['completed_steps']}/{result['workflow_steps']}")

    # Display data flow summary
    if 'data_flow' in result:
        print("\n📋 Data Flow Summary:")
        print("-" * 30)
        for step_info in result['data_flow']['steps']:
            print(f"Step {step_info['step']}: {step_info['agent']}")
            print(f"  Input Mapping: {step_info['input_mapping']}")
            print(f"  Output Key: {step_info['output_key']}")
            print(f"  Has Output: {step_info['has_output']}")
            if step_info['transform']:
                print(f"  Transform: {step_info['transform']}")
            print()

    # Display step outputs
    if 'step_outputs' in result:
        print("📤 Step Outputs:")
        print("-" * 20)
        for key, value in result['step_outputs'].items():
            print(f"{key}: {str(value)[:100]}...")
            print()

    # Display final shared data
    if 'final_data' in result:
        print("🎯 Final Shared Data:")
        print("-" * 25)
        for key, value in result['final_data'].items():
            print(f"{key}: {value}")
        print()

    return result


def example_error_handling_pipeline():
    """Demonstrate error handling in data pipeline workflow."""

    orchestrator = TaskOrchestrator()

    # Workflow with potential failure and error handling
    error_handling_workflow = [
        {
            "agent": "data_analyst",
            "task_data": {
                "task": "Analyze valid data"
            },
            "output_key": "step1_output"
        },
        {
            "agent": "nonexistent_agent",  # This will fail
            "task_data": {
                "task": "This will fail"
            },
            "output_key": "step2_output",
            "on_error": "continue"  # Continue despite failure
        },
        {
            "agent": "content_writer",
            "task_data": {
                "task": "Write summary despite previous failure"
            },
            "input_mapping": {
                "data": "step1_output"  # Use data from step 1
            },
            "output_key": "final_output"
        }
    ]

    print("\n🔧 Testing Error Handling Pipeline...")
    print("=" * 45)

    result = orchestrator.run_data_pipeline_workflow(
        workflow=error_handling_workflow,
        provider="deepseek"
    )

    print(f"📊 Workflow Success: {result['success']}")
    print(f"📈 Steps Completed: {result['completed_steps']}/{result['workflow_steps']}")

    # Show which steps succeeded/failed
    for i, step_result in enumerate(result['results']):
        status = "✅ Success" if step_result.get('success') else "❌ Failed"
        print(f"Step {i+1}: {status}")
        if not step_result.get('success'):
            print(f"  Error: {step_result.get('error', 'Unknown error')}")

    return result


def example_transformation_pipeline():
    """Demonstrate data transformations in the pipeline."""

    orchestrator = TaskOrchestrator()

    # Workflow with various data transformations
    transform_workflow = [
        {
            "agent": "data_analyst",
            "task_data": {
                "task": "Generate a list of numbers: 1, 2, 3, 4, 5"
            },
            "output_key": "raw_numbers",
            "transform": "extract_numbers"
        },
        {
            "agent": "content_writer",
            "task_data": {
                "task": "Create formatted text from numbers"
            },
            "input_mapping": {
                "numbers": "raw_numbers"
            },
            "output_key": "formatted_text",
            "transform": "uppercase"
        }
    ]

    print("\n🔄 Testing Data Transformation Pipeline...")
    print("=" * 45)

    result = orchestrator.run_data_pipeline_workflow(
        workflow=transform_workflow,
        provider="deepseek"
    )

    print(f"📊 Workflow Success: {result['success']}")

    # Show transformations applied
    if 'data_flow' in result and 'transformations' in result['data_flow']:
        print("\n🔄 Transformations Applied:")
        for transform in result['data_flow']['transformations']:
            print(f"Step {transform['step']}: {transform['transform']}")

    return result


if __name__ == "__main__":
    print("🎯 Data Pipeline Workflow Examples")
    print("=" * 50)

    # Run examples
    try:
        # Main data pipeline example
        result1 = example_data_pipeline()

        # Error handling example
        result2 = example_error_handling_pipeline()

        # Transformation example
        result3 = example_transformation_pipeline()

        print("\n✅ All examples completed!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
