#!/usr/bin/env python3
"""
Demo script for testing the streaming API endpoints.

This script demonstrates how to:
- Call the streaming API endpoints
- Handle Server-Sent Events (SSE)
- Process streaming responses
"""

import requests
import json
import time
from typing import Dict, Any


def test_streaming_agent_api(base_url: str = "http://localhost:8000"):
    """Test the /run-agent-stream endpoint."""
    print("🚀 Testing Streaming Agent API\n")
    
    # Prepare request data
    request_data = {
        "agent_name": "content_writer",
        "task_data": {
            "topic": "The Future of Machine Learning",
            "style": "informative and engaging", 
            "length": "200 words",
            "audience": "developers"
        },
        "provider": "openai"
    }
    
    print("📝 Request data:")
    print(json.dumps(request_data, indent=2))
    print("\n" + "="*60 + "\n")
    
    print("🔄 Streaming response:")
    
    try:
        # Make streaming request
        response = requests.post(
            f"{base_url}/run-agent-stream",
            json=request_data,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return
        
        content_chunks = []
        
        # Process streaming response
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                try:
                    chunk = json.loads(data_str)
                    
                    if chunk.get("type") == "orchestrator_start":
                        print(f"🚀 Job {chunk['job_id']} started")
                        print(f"🤖 Agent: {chunk['agent_name']}")
                        print(f"⚡ Provider: {chunk['provider']}")
                        print(f"🧠 Model: {chunk.get('model_name', 'default')}")
                        print("\n📄 Content:")
                        
                    elif chunk.get("type") == "chunk":
                        print(chunk["chunk"], end="", flush=True)
                        content_chunks.append(chunk["chunk"])
                        
                    elif chunk.get("type") == "complete":
                        print(f"\n\n✅ Agent completed")
                        
                    elif chunk.get("type") == "orchestrator_complete":
                        print(f"🎉 Total execution time: {chunk['execution_time']:.2f}s")
                        
                    elif chunk.get("type") == "stream_complete":
                        print(f"📡 Stream completed")
                        break
                        
                    elif chunk.get("type") == "stream_error":
                        print(f"❌ Stream error: {chunk['error']}")
                        break
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON: {data_str}")
                    
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    print("\n" + "="*60 + "\n")


def test_streaming_workflow_api(base_url: str = "http://localhost:8000"):
    """Test the /run-workflow-stream endpoint."""
    print("🔗 Testing Streaming Workflow API\n")
    
    # Prepare workflow request
    request_data = {
        "workflow": [
            {
                "agent": "summary",
                "task_data": {
                    "text": "Machine learning is transforming industries by enabling computers to learn from data without explicit programming. Deep learning, a subset of ML, uses neural networks to process complex patterns. Applications include image recognition, natural language processing, and predictive analytics. The technology continues to evolve rapidly."
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
    }
    
    print("📊 Workflow request:")
    print(json.dumps(request_data, indent=2))
    print("\n" + "="*60 + "\n")
    
    print("🔄 Streaming workflow:")
    
    try:
        # Make streaming request
        response = requests.post(
            f"{base_url}/run-workflow-stream",
            json=request_data,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return
        
        current_step = 0
        
        # Process streaming response
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                try:
                    chunk = json.loads(data_str)
                    
                    if chunk.get("type") == "workflow_step_start":
                        current_step = chunk["step_number"]
                        print(f"\n📋 Step {current_step}/{chunk['total_steps']}: {chunk['agent_name']}")
                        
                    elif chunk.get("type") == "orchestrator_start":
                        print(f"🚀 Agent started (Job: {chunk['job_id']})")
                        
                    elif chunk.get("type") == "chunk":
                        print(chunk["chunk"], end="", flush=True)
                        
                    elif chunk.get("type") == "complete":
                        print(f"\n✅ Agent completed")
                        
                    elif chunk.get("type") == "workflow_step_complete":
                        print(f"📋 Step {chunk['step_number']} completed")
                        
                    elif chunk.get("type") == "workflow_complete":
                        print(f"\n🎉 Workflow completed!")
                        print(f"📊 Completed {chunk['completed_steps']}/{chunk['total_steps']} steps")
                        break
                        
                    elif chunk.get("type") == "workflow_error":
                        print(f"❌ Workflow error: {chunk['error']}")
                        break
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON: {data_str}")
                    
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    print("\n" + "="*60 + "\n")


def check_api_health(base_url: str = "http://localhost:8000"):
    """Check if the API is running."""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"✅ API is healthy at {base_url}")
            return True
        else:
            print(f"⚠️  API responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach API at {base_url}: {str(e)}")
        return False


def main():
    """Main demonstration function."""
    print("🌟 Streaming API Demo")
    print("=" * 60)
    print()
    
    base_url = "http://localhost:8000"
    
    # Check if API is running
    if not check_api_health(base_url):
        print("\n💡 To start the API, run:")
        print("   uvicorn api.app:app --reload --host 0.0.0.0 --port 8000")
        return
    
    print()
    
    try:
        # Test streaming agent endpoint
        test_streaming_agent_api(base_url)
        
        # Test streaming workflow endpoint
        test_streaming_workflow_api(base_url)
        
        print("🎊 All streaming API tests completed!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
