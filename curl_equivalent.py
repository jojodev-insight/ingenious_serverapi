# """
# Simple Python equivalent of the specific curl command.
# Now with proper file data loading!
# """

# import requests
# import json


# def main():
#     """
#     Direct Python equivalent of:
    
#     curl -X POST 'http://localhost:8000/run-agent' \
#       -H 'Content-Type: application/json' \
#       -d '{
#       "agent_name": "file_data_analyst",
#       "task_data": {
#         "analysis_request": "Examine employee performance, salary distribution, and department insights",
#         "files": [
#           "employee_data.xlsx"
#         ]
#       },
#       "provider": "deepseek",
#       "model_name": "deepseek-chat",
#       "llm_config": {
#         "temperature": 0.2,
#         "max_tokens": 1200
#       }
#     }'
    
#     This script now properly loads file data from the employee_data.xlsx file
#     and includes the actual data content in the analysis prompt.
#     """
    
#     # API endpoint
#     url = "http://localhost:8000/run-agent"
    
#     # Headers (equivalent to -H 'Content-Type: application/json')
#     headers = {
#         "Content-Type": "application/json"
#     }
    
#     # Data payload (equivalent to -d '{...}')
#     # The FileDataAnalyst agent will automatically load and process the employee_data.xlsx file
#     data = {
#         "agent_name": "file_data_analyst",
#         "task_data": {
#             "analysis_request": "Examine employee performance, salary distribution, and department insights",
#             "files": [
#                 "employee_data.xlsx"  # This file will be loaded from the data/ directory
#             ]
#         },
#         "provider": "openai",  # Use DeepSeek with valid API key
#         "model_name": "gpt-4.1",
#         "llm_config": {
#             "temperature": 0.2,
#             "max_tokens": 1200
#         }
#     }
    
#     try:
#         print("🚀 Making API request...")
#         print(f"📄 Processing file: employee_data.xlsx")
#         print(f"🤖 Using agent: file_data_analyst")
#         print(f"🌐 Provider: deepseek")
#         print()
        
#         # Make POST request (equivalent to curl -X POST)
#         response = requests.post(url, headers=headers, json=data)
        
#         # Check for HTTP errors
#         response.raise_for_status()
        
#         # Parse response
#         result = response.json()
        
#         # Print the response
#         print("✅ Response Status:", response.status_code)
#         print("📊 Analysis Results:")
#         print("=" * 60)
        
#         if result.get("success"):
#             print(f"Job ID: {result['job_id']}")
#             print(f"Execution Time: {result['execution_time']:.2f} seconds")
#             print(f"Agent: {result['agent_name']}")
#             print(f"Provider: {result['provider']}")
#             print()
#             print("📋 Analysis Response:")
#             print("-" * 40)
#             print(result['response'])
#         else:
#             print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
#     except requests.exceptions.ConnectionError:
#         print("❌ Error: Could not connect to http://localhost:8000")
#         print("💡 Make sure the FastAPI server is running with:")
#         print("   Set-Location 'c:\\Users\\jojo\\LangChain Test\\autogen_project'")
#         print("   uv run python main.py")
        
#     except requests.exceptions.HTTPError as e:
#         print(f"❌ HTTP Error: {e}")
#         print("Response:", response.text)
        
#     except Exception as e:
#         print(f"❌ Error: {e}")


# if __name__ == "__main__":
#     main()


# Save this as advanced_workflow_example.py and run with:
#   uv run advanced_workflow_example.py
# import requests
# import json

# def run_advanced_workflow():
#     url = "http://localhost:8000/run-workflow"
#     headers = {"Content-Type": "application/json"}
    
#     workflow_data = {
#         "workflow": [
#             {
#                 "agent": "file_data_analyst",
#                 "task_data": {
#                     "analysis_request": "Analyze Q4 financial performance and identify key trends",
#                     "files": ["sales_data.csv", "employee_data.xlsx"],
#                     "output_format": "structured_summary"
#                 }
#             },
#             {
#                 "agent": "data_analyst", 
#                 "task_data": {
#                     "data": "{{previous_result}}",
#                     "context": "Create predictive insights for Q1 planning",
#                     "analysis_type": "forecasting"
#                 }
#             },
#             {
#                 "agent": "content_writer",
#                 "task_data": {
#                     "topic": "Q4 Performance & Q1 Forecast Report",
#                     "audience": "C-suite executives", 
#                     "content_type": "executive briefing",
#                     "data_source": "{{previous_result}}",
#                     "include_charts": True
#                 }
#             }
#         ],
#         "provider": "openai",
#         "model_name": "gpt-4",
#         "llm_config": {
#             "temperature": 0.3,
#             "max_tokens": 2000
#         }
#     }
    
#     print("🚀 Starting advanced workflow...")
#     response = requests.post(url, headers=headers, json=workflow_data)
    
#     if response.status_code == 200:
#         result = response.json()
#         print("✅ Workflow completed successfully!")
#         print(f"📊 Total execution time: {result.get('total_execution_time', 'N/A')} seconds")
#         print("\n📋 Final Report:")
#         print("=" * 60)
#         print(result.get('final_result', 'No result available'))
#     else:
#         print(f"❌ Workflow failed with status: {response.status_code}")
#         print(f"Error: {response.text}")

# if __name__ == "__main__":
#     run_advanced_workflow()

# Save this as programmatic_workflow_example.py and run with:
#   uv run programmatic_workflow_example.py
import asyncio
from typing import Dict, Any
from agents import FileDataAnalyst, ContentWriterAgent, DataAnalystAgent
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
            "agent": "DataAnalystAgent",
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
            "agent": "ContentWriterAgent",
            "task": "Executive Report",
            "result": result,
            "status": "completed"
        })
        
        print(f"✅ Stage 3 completed. Report length: {len(result)} characters")
        return result
    
    def _compile_final_report(self) -> Dict[str, Any]:
        """Compile comprehensive workflow results"""
        # Extract response content safely
        final_deliverable = ""
        financial_insights = ""
        forecast_summary = ""
        
        if self.workflow_results:
            # Get final deliverable
            if len(self.workflow_results) >= 3:
                final_result = self.workflow_results[2]["result"]
                if isinstance(final_result, dict) and "response" in final_result:
                    final_deliverable = final_result["response"]
                else:
                    final_deliverable = str(final_result)
            
            # Get supporting analysis
            if len(self.workflow_results) >= 1:
                first_result = self.workflow_results[0]["result"]
                if isinstance(first_result, dict) and "response" in first_result:
                    financial_insights = first_result["response"][:500] + "..."
                else:
                    financial_insights = str(first_result)[:500] + "..."
            
            if len(self.workflow_results) >= 2:
                second_result = self.workflow_results[1]["result"]
                if isinstance(second_result, dict) and "response" in second_result:
                    forecast_summary = second_result["response"][:500] + "..."
                else:
                    forecast_summary = str(second_result)[:500] + "..."
        
        return {
            "workflow_type": "Advanced Business Intelligence Pipeline",
            "total_stages": len(self.workflow_results),
            "execution_summary": {
                "stage_1": "Financial data analysis completed",
                "stage_2": "Predictive forecasting completed", 
                "stage_3": "Executive report generated"
            },
            "final_deliverable": final_deliverable,
            "supporting_analysis": {
                "financial_insights": financial_insights,
                "forecast_summary": forecast_summary
            },
            "workflow_metadata": {
                "agents_used": ["FileDataAnalyst", "DataAnalystAgent", "ContentWriterAgent"],
                "providers": ["openai", "openai", "openai"],  # All default to OpenAI with DeepSeek fallback
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

    
    return final_report

if __name__ == "__main__":
    frep = asyncio.run(main())
    print("\n" + "=" * 60)
    print(frep)
    print("\n🎉 Advanced Workflow Execution Complete!")