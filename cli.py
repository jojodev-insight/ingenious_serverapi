"""CLI interface for testing the Autogen orchestrator."""

import argparse
import json
import sys
from typing import Any

from api.orchestrator import TaskOrchestrator


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Autogen Agent CLI")
    parser.add_argument("--list-agents", action="store_true", help="List available agents")
    parser.add_argument("--agent", help="Agent name to run")
    parser.add_argument("--provider", default=None, help="LLM provider (openai/deepseek)")
    parser.add_argument("--data", help="Data for data analyst agent")
    parser.add_argument("--context", help="Context for the task")
    parser.add_argument("--topic", help="Topic for content writer agent")
    parser.add_argument("--audience", help="Target audience")
    parser.add_argument("--code", help="Code for code reviewer agent")
    parser.add_argument("--json-input", help="JSON file with task data")

    args = parser.parse_args()

    orchestrator = TaskOrchestrator()

    if args.list_agents:
        agents = orchestrator.list_agents()
        print("Available agents:")
        for name, class_name in agents.items():
            print(f"  - {name} ({class_name})")
        return

    if not args.agent:
        print("Error: --agent is required (unless using --list-agents)")
        parser.print_help()
        sys.exit(1)

    # Prepare task data
    if args.json_input:
        try:
            with open(args.json_input) as f:
                task_data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            sys.exit(1)
    else:
        task_data = build_task_data(args)

    print(f"Running agent '{args.agent}' with provider '{args.provider or 'default'}'")
    print("Task data:", json.dumps(task_data, indent=2))
    print("-" * 50)

    # Run the agent
    result = orchestrator.run_agent(args.agent, task_data, args.provider)

    # Print results
    if result["success"]:
        print("✅ Agent execution successful!")
        print(f"Job ID: {result['job_id']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        print(f"Agent: {result.get('agent_name', 'Unknown')}")
        print(f"Provider: {result.get('provider', 'Unknown')}")
        print("\n📄 Response:")
        print(result.get("response", "No response"))
    else:
        print("❌ Agent execution failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def build_task_data(args) -> dict[str, Any]:
    """Build task data from command line arguments."""
    task_data = {}

    # Common fields
    if args.context:
        task_data["context"] = args.context

    # Agent-specific fields
    if args.agent == "data_analyst":
        if args.data:
            task_data["data"] = args.data
        else:
            print("Error: --data is required for data_analyst agent")
            sys.exit(1)

    elif args.agent == "content_writer":
        if args.topic:
            task_data["topic"] = args.topic
        else:
            print("Error: --topic is required for content_writer agent")
            sys.exit(1)

        if args.audience:
            task_data["audience"] = args.audience

    elif args.agent == "code_reviewer":
        if args.code:
            task_data["code"] = args.code
        else:
            print("Error: --code is required for code_reviewer agent")
            sys.exit(1)

    return task_data


if __name__ == "__main__":
    main()
