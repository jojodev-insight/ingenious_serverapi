"""
Simple Python equivalent of the specific curl command.
Now with proper file data loading!
"""

import requests
import json


def main():
    """
    Direct Python equivalent of:
    
    curl -X POST 'http://localhost:8000/run-agent' \
      -H 'Content-Type: application/json' \
      -d '{
      "agent_name": "file_data_analyst",
      "task_data": {
        "analysis_request": "Examine employee performance, salary distribution, and department insights",
        "files": [
          "employee_data.xlsx"
        ]
      },
      "provider": "deepseek",
      "model_name": "deepseek-chat",
      "llm_config": {
        "temperature": 0.2,
        "max_tokens": 1200
      }
    }'
    
    This script now properly loads file data from the employee_data.xlsx file
    and includes the actual data content in the analysis prompt.
    """
    
    # API endpoint
    url = "http://localhost:8000/run-agent"
    
    # Headers (equivalent to -H 'Content-Type: application/json')
    headers = {
        "Content-Type": "application/json"
    }
    
    # Data payload (equivalent to -d '{...}')
    # The FileDataAnalyst agent will automatically load and process the employee_data.xlsx file
    data = {
        "agent_name": "file_data_analyst",
        "task_data": {
            "analysis_request": "Examine employee performance, salary distribution, and department insights",
            "files": [
                "employee_data.xlsx"  # This file will be loaded from the data/ directory
            ]
        },
        "provider": "openai",  # Use DeepSeek with valid API key
        "model_name": "gpt-4.1",
        "llm_config": {
            "temperature": 0.2,
            "max_tokens": 1200
        }
    }
    
    try:
        print("🚀 Making API request...")
        print(f"📄 Processing file: employee_data.xlsx")
        print(f"🤖 Using agent: file_data_analyst")
        print(f"🌐 Provider: deepseek")
        print()
        
        # Make POST request (equivalent to curl -X POST)
        response = requests.post(url, headers=headers, json=data)
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Print the response
        print("✅ Response Status:", response.status_code)
        print("📊 Analysis Results:")
        print("=" * 60)
        
        if result.get("success"):
            print(f"Job ID: {result['job_id']}")
            print(f"Execution Time: {result['execution_time']:.2f} seconds")
            print(f"Agent: {result['agent_name']}")
            print(f"Provider: {result['provider']}")
            print()
            print("📋 Analysis Response:")
            print("-" * 40)
            print(result['response'])
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to http://localhost:8000")
        print("💡 Make sure the FastAPI server is running with:")
        print("   Set-Location 'c:\\Users\\jojo\\LangChain Test\\autogen_project'")
        print("   uv run python main.py")
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print("Response:", response.text)
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
