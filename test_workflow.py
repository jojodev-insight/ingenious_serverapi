import json

import requests


def test_workflow():
    url = "http://localhost:8000/run-workflow"
    headers = {"Content-Type": "application/json"}

    # Simple test workflow
    workflow_data = {
        "workflow": [
            {
                "agent": "file_data_analyst",
                "task_data": {
                    "analysis_request": "Test analysis",
                    "files": ["sales_data.csv"]
                }
            }
        ],
        "provider": "openai"
    }

    print("🧪 Testing workflow endpoint...")
    print(f"📊 Request: {json.dumps(workflow_data, indent=2)}")

    try:
        response = requests.post(url, headers=headers, json=workflow_data, timeout=30)
        print(f"📈 Status Code: {response.status_code}")
        print(f"📋 Response Headers: {dict(response.headers)}")
        print(f"📄 Response Body: {response.text}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
        else:
            print("❌ Failed!")

    except Exception as e:
        print(f"💥 Exception: {e}")

if __name__ == "__main__":
    test_workflow()
