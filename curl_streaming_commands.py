#!/usr/bin/env python3
"""
Curl equivalent commands for testing the streaming API endpoints.

This script shows how to test the streaming endpoints using curl commands.
"""


def print_curl_commands():
    """Print curl commands for testing streaming endpoints."""

    print("🌟 Curl Commands for Streaming API Testing")
    print("=" * 60)
    print()

    print("🚀 1. Test Single Agent Streaming:")
    print("   curl -X POST http://localhost:8000/run-agent-stream \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -H 'Accept: text/event-stream' \\")
    print("        --no-buffer \\")
    print("        -d '{")
    print('          "agent_name": "content_writer",')
    print('          "task_data": {')
    print('            "topic": "AI in Healthcare",')
    print('            "style": "informative",')
    print('            "length": "150 words",')
    print('            "audience": "medical professionals"')
    print('          },')
    print('          "provider": "openai"')
    print("        }'")
    print()

    print("🔗 2. Test Workflow Streaming:")
    print("   curl -X POST http://localhost:8000/run-workflow-stream \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -H 'Accept: text/event-stream' \\")
    print("        --no-buffer \\")
    print("        -d '{")
    print('          "workflow": [')
    print('            {')
    print('              "agent": "summary",')
    print('              "task_data": {')
    print('                "text": "Artificial intelligence is rapidly advancing..."')
    print('              }')
    print('            },')
    print('            {')
    print('              "agent": "formatter",')
    print('              "task_data": {')
    print('                "text": "{{previous_result}}",')
    print('                "format": "numbered list"')
    print('              }')
    print('            }')
    print('          ],')
    print('          "provider": "openai"')
    print("        }'")
    print()

    print("🏥 3. Health Check:")
    print("   curl http://localhost:8000/health")
    print()

    print("📋 4. List Available Endpoints:")
    print("   curl http://localhost:8000/")
    print()

    print("🤖 5. List Available Agents:")
    print("   curl http://localhost:8000/agents")
    print()

    print("💡 Notes:")
    print("   - Use --no-buffer to see streaming output in real-time")
    print("   - The streaming endpoints return Server-Sent Events (SSE)")
    print("   - Each line starts with 'data: ' followed by JSON")
    print("   - Make sure the API is running: uvicorn api.app:app --reload")
    print()


if __name__ == "__main__":
    print_curl_commands()
