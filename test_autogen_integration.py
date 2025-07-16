"""Quick test to verify AutoGen integration works correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_autogen_imports():
    """Test that AutoGen integration is available through BaseAgent."""
    print("🧪 Testing AutoGen integration availability...")
    
    try:
        from agents import BaseAgent
        from agents.agent_factory import AgentFactory
        
        # Test that AutoGen agent types are available
        agent_types = list(AgentFactory.AGENT_TYPES.keys())
        if "autogen" in agent_types and "autogen_workflow" in agent_types:
            print("✅ AutoGen agent types available through AgentFactory")
            return True
        else:
            print(f"❌ AutoGen agent types not found. Available: {agent_types}")
            return False
    except Exception as e:
        print(f"❌ Failed to check AutoGen integration: {e}")
        return False

def test_orchestrator_integration():
    """Test that orchestrator recognizes AutoGen agents."""
    print("\n🧪 Testing orchestrator integration...")
    
    try:
        from api.orchestrator import TaskOrchestrator
        
        orchestrator = TaskOrchestrator()
        agents = orchestrator.list_agents()
        
        if "autogen" in agents and "autogen_workflow" in agents:
            print("✅ AutoGen agents registered in orchestrator")
            print(f"   Available agents: {list(agents.keys())}")
            return True
        else:
            print(f"❌ AutoGen agents not found in orchestrator. Available: {list(agents.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test orchestrator integration: {e}")
        return False

def test_agent_factory():
    """Test that agent factory can create AutoGen agents."""
    print("\n🧪 Testing agent factory...")
    
    try:
        from agents import AgentFactory
        
        # Test if AutoGen agent types are available
        available_types = list(AgentFactory.AGENT_TYPES.keys())
        
        if "autogen" in available_types and "autogen_workflow" in available_types:
            print("✅ AutoGen agents available in factory")
            print(f"   Available types: {available_types}")
            return True
        else:
            print(f"❌ AutoGen agents not found in factory. Available: {available_types}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test agent factory: {e}")
        return False

def test_api_endpoints():
    """Test that API endpoints include AutoGen routes."""
    print("\n🧪 Testing API endpoints...")
    
    try:
        from api.app import app
        
        # Get all routes
        routes = [route.path for route in app.routes]
        autogen_routes = [route for route in routes if "autogen" in route]
        
        expected_routes = [
            "/autogen/conversation",
            "/autogen/workflow", 
            "/autogen/conversation/stream",
            "/autogen/workflow/stream"
        ]
        
        missing_routes = [route for route in expected_routes if route not in routes]
        
        if not missing_routes:
            print("✅ All AutoGen API endpoints available")
            print(f"   AutoGen routes: {autogen_routes}")
            return True
        else:
            print(f"❌ Missing AutoGen routes: {missing_routes}")
            print(f"   Available AutoGen routes: {autogen_routes}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test API endpoints: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AutoGen Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_autogen_imports,
        test_orchestrator_integration,
        test_agent_factory,
        test_api_endpoints
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! AutoGen integration is working correctly.")
        print("✨ You can now use AutoGen agents alongside existing agents.")
        print("\n📖 Next steps:")
        print("   1. Run: uv run autogen_demo.py")
        print("   2. Read: AUTOGEN_INTEGRATION.md")
        print("   3. Try the API endpoints at http://localhost:8000/docs")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
