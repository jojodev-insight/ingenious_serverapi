#!/usr/bin/env python3
"""Test script to verify AutoGen merge into BaseAgent is working correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_merged_autogen_functionality():
    """Test that merged AutoGen functionality works correctly."""
    print("🧪 Testing merged AutoGen functionality...")
    
    try:
        from agents.agent_factory import AgentFactory
        from api.orchestrator import TaskOrchestrator
        
        # Test 1: Agent Factory Integration
        print("  ✅ Testing agent factory integration...")
        agent_types = list(AgentFactory.AGENT_TYPES.keys())
        assert "autogen" in agent_types, "AutoGen agent type not found"
        assert "autogen_workflow" in agent_types, "AutoGen workflow agent type not found"
        print(f"    Available types: {agent_types}")
        
        # Test 2: Agent Creation
        print("  ✅ Testing agent creation...")
        autogen_agent = AgentFactory.create_agent("autogen")
        workflow_agent = AgentFactory.create_agent("autogen_workflow")
        
        assert autogen_agent.agent_type == "autogen", "AutoGen agent type mismatch"
        assert workflow_agent.agent_type == "autogen_workflow", "Workflow agent type mismatch"
        assert autogen_agent.name == "AutoGenAgent", "AutoGen agent name mismatch"
        assert workflow_agent.name == "AutoGenWorkflowAgent", "Workflow agent name mismatch"
        
        print(f"    Created AutoGen agent: {autogen_agent.name} (type: {autogen_agent.agent_type})")
        print(f"    Created workflow agent: {workflow_agent.name} (type: {workflow_agent.agent_type})")
        
        # Test 3: Orchestrator Integration
        print("  ✅ Testing orchestrator integration...")
        orchestrator = TaskOrchestrator()
        available_agents = orchestrator.list_agents()
        
        assert "autogen" in available_agents, "AutoGen not in orchestrator"
        assert "autogen_workflow" in available_agents, "AutoGen workflow not in orchestrator"
        print(f"    Orchestrator agents: {list(available_agents.keys())}")
        
        # Test 4: Task Preparation
        print("  ✅ Testing task preparation...")
        test_task = {"message": "Hello from merged AutoGen!", "agents": []}
        
        autogen_prompt = autogen_agent.prepare_task(test_task)
        workflow_prompt = workflow_agent.prepare_task(test_task)
        
        assert autogen_prompt == "Hello from merged AutoGen!", "AutoGen task preparation failed"
        assert workflow_prompt == "Hello from merged AutoGen!", "Workflow task preparation failed"
        print(f"    Task prepared successfully: '{autogen_prompt[:50]}...'")
        
        # Test 5: Execution Capability (without requiring full LLM call)
        print("  ✅ Testing execution capability...")
        # Just test that the methods exist and can be called without errors
        assert hasattr(autogen_agent, 'execute_sync'), "execute_sync method missing"
        assert hasattr(autogen_agent, 'execute_stream'), "execute_stream method missing"
        assert hasattr(workflow_agent, 'execute_sync'), "workflow execute_sync method missing"
        assert hasattr(workflow_agent, 'execute_stream'), "workflow execute_stream method missing"
        print("    All execution methods available")
        
        print("✅ All merged AutoGen functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Merged AutoGen functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test that existing agent functionality still works."""
    print("\n🧪 Testing backwards compatibility...")
    
    try:
        from agents.agent_factory import AgentFactory
        
        # Test standard agents still work
        standard_types = ["data_analyst", "content_writer", "code_reviewer"]
        
        for agent_type in standard_types:
            agent = AgentFactory.create_agent(agent_type)
            assert agent.agent_type == "standard", f"{agent_type} should have standard agent_type"
            assert hasattr(agent, 'template'), f"{agent_type} should have template"
            assert agent.use_templates is True, f"{agent_type} should use templates"
            print(f"    ✅ {agent_type} agent works correctly")
        
        print("✅ Backwards compatibility maintained!")
        return True
        
    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 AutoGen Merge Verification Test Suite")
    print("=" * 50)
    
    results = []
    results.append(test_merged_autogen_functionality())
    results.append(test_backwards_compatibility())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed < total:
        print(f"❌ Failed: {total - passed}/{total}")
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    else:
        print("🎉 All tests passed! AutoGen merge was successful!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
