"""AutoGen demonstration script showing integration capabilities."""

import asyncio
from api.orchestrator import TaskOrchestrator
from core import agent_logger


def demo_basic_autogen_conversation():
    """Demonstrate basic AutoGen conversation functionality."""
    print("=" * 60)
    print("AutoGen Basic Conversation Demo")
    print("=" * 60)
    
    orchestrator = TaskOrchestrator()
    
    # Basic conversation task
    task_data = {
        "message": "Explain the concept of machine learning in simple terms and provide a practical example.",
        "agents": [
            {
                "type": "user_proxy",
                "name": "User",
                "description": "User initiating the conversation"
            },
            {
                "type": "assistant",
                "name": "MLExpert",
                "description": "Machine Learning Expert",
                "system_message": "You are a machine learning expert who explains complex concepts in simple, understandable terms with practical examples."
            }
        ],
        "max_turns": 5
    }
    
    print("Running AutoGen conversation...")
    result = orchestrator.run_agent("autogen", task_data)
    
    if result["success"]:
        print(f"\n✅ Conversation completed successfully!")
        print(f"📊 Total messages: {result.get('total_messages', 'N/A')}")
        print(f"🤖 Agents used: {', '.join(result.get('agents_used', []))}")
        print(f"\n💬 Final response:\n{result['response']}")
        
        # Show conversation history if available
        if 'conversation' in result:
            print(f"\n📝 Conversation History:")
            for i, msg in enumerate(result['conversation'], 1):
                print(f"  {i}. {msg['source']}: {msg['content'][:100]}...")
    else:
        print(f"❌ Conversation failed: {result.get('error', 'Unknown error')}")


def demo_autogen_research_workflow():
    """Demonstrate AutoGen research workflow."""
    print("\n" + "=" * 60)
    print("AutoGen Research Workflow Demo")
    print("=" * 60)
    
    orchestrator = TaskOrchestrator()
    
    # Research workflow task
    task_data = {
        "message": "Research the impact of artificial intelligence on healthcare, analyze the findings, and write a comprehensive report.",
        "workflow_type": "research"
    }
    
    print("Running AutoGen research workflow...")
    result = orchestrator.run_agent("autogen_workflow", task_data)
    
    if result["success"]:
        print(f"\n✅ Research workflow completed successfully!")
        print(f"🔬 Workflow type: {result.get('workflow_type', 'N/A')}")
        print(f"📊 Total messages: {result.get('total_messages', 'N/A')}")
        print(f"🤖 Agents used: {', '.join(result.get('agents_used', []))}")
        print(f"\n📄 Research report:\n{result['response']}")
    else:
        print(f"❌ Research workflow failed: {result.get('error', 'Unknown error')}")


def demo_autogen_code_review_workflow():
    """Demonstrate AutoGen code review workflow."""
    print("\n" + "=" * 60)
    print("AutoGen Code Review Workflow Demo")
    print("=" * 60)
    
    orchestrator = TaskOrchestrator()
    
    # Sample code to review
    sample_code = """
def user_login(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    result = db.execute(query)
    if result:
        session['user_id'] = result[0]['id']
        return True
    return False
"""
    
    # Code review workflow task
    task_data = {
        "message": f"Please review the following Python code for security, performance, and quality issues:\n\n{sample_code}",
        "workflow_type": "code_review"
    }
    
    print("Running AutoGen code review workflow...")
    result = orchestrator.run_agent("autogen_workflow", task_data)
    
    if result["success"]:
        print(f"\n✅ Code review workflow completed successfully!")
        print(f"🔍 Workflow type: {result.get('workflow_type', 'N/A')}")
        print(f"📊 Total messages: {result.get('total_messages', 'N/A')}")
        print(f"🤖 Agents used: {', '.join(result.get('agents_used', []))}")
        print(f"\n📋 Code review report:\n{result['response']}")
    else:
        print(f"❌ Code review workflow failed: {result.get('error', 'Unknown error')}")


async def demo_autogen_streaming():
    """Demonstrate AutoGen streaming functionality."""
    print("\n" + "=" * 60)
    print("AutoGen Streaming Demo")
    print("=" * 60)
    
    orchestrator = TaskOrchestrator()
    
    task_data = {
        "message": "Explain the benefits of renewable energy and create a persuasive argument for solar power adoption.",
        "agents": [
            {
                "type": "user_proxy",
                "name": "Inquirer",
                "description": "Person asking about renewable energy"
            },
            {
                "type": "assistant",
                "name": "EnergyExpert",
                "description": "Renewable Energy Specialist",
                "system_message": "You are a renewable energy expert who provides detailed, factual information about clean energy technologies and their benefits."
            }
        ]
    }
    
    print("Starting AutoGen streaming conversation...")
    
    async for chunk in orchestrator.run_agent_stream("autogen", task_data):
        if chunk["success"]:
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "start":
                print(f"🚀 {chunk.get('message', 'Started')}")
            elif chunk_type == "agents_created":
                print(f"🤖 {chunk.get('message', 'Agents created')}")
                print(f"   Agents: {', '.join(chunk.get('agents', []))}")
            elif chunk_type == "conversation_start":
                print(f"💬 {chunk.get('message', 'Conversation started')}")
            elif chunk_type == "message":
                source = chunk.get('source', 'Unknown')
                content = chunk.get('content', '')[:100]
                print(f"   📝 {source}: {content}...")
            elif chunk_type == "complete":
                print(f"✅ Conversation completed!")
                print(f"📊 Total messages: {chunk.get('total_messages', 'N/A')}")
                print(f"\n🎯 Final response:\n{chunk.get('response', '')}")
        else:
            print(f"❌ Streaming error: {chunk.get('error', 'Unknown error')}")
            break


def demo_traditional_vs_autogen():
    """Compare traditional agent execution with AutoGen."""
    print("\n" + "=" * 60)
    print("Traditional vs AutoGen Comparison Demo")
    print("=" * 60)
    
    orchestrator = TaskOrchestrator()
    
    task_prompt = "Analyze the advantages and disadvantages of remote work and provide recommendations."
    
    # Traditional single agent approach
    print("🔄 Running traditional single agent approach...")
    traditional_task = {
        "prompt": task_prompt,
        "template_name": "content_writer_prompt.txt"
    }
    
    traditional_result = orchestrator.run_agent("content_writer", traditional_task)
    
    # AutoGen multi-agent approach
    print("\n🔄 Running AutoGen multi-agent approach...")
    autogen_task = {
        "message": task_prompt,
        "agents": [
            {
                "type": "user_proxy",
                "name": "BusinessAnalyst",
                "description": "Business analyst requesting the analysis"
            },
            {
                "type": "assistant",
                "name": "HRSpecialist",
                "description": "HR specialist focusing on employee perspectives",
                "system_message": "You are an HR specialist. Focus on employee satisfaction, productivity, and workplace culture aspects of remote work."
            },
            {
                "type": "assistant",
                "name": "BusinessConsultant",
                "description": "Business consultant focusing on organizational impact",
                "system_message": "You are a business consultant. Focus on cost implications, operational efficiency, and business strategy aspects of remote work."
            }
        ]
    }
    
    autogen_result = orchestrator.run_agent("autogen", autogen_task)
    
    # Compare results
    print("\n📊 Comparison Results:")
    print("-" * 40)
    
    if traditional_result["success"]:
        print(f"✅ Traditional approach: Success")
        print(f"   Response length: {len(traditional_result['response'])} characters")
        print(f"   Execution time: {traditional_result.get('execution_time', 'N/A'):.2f}s")
    else:
        print(f"❌ Traditional approach: Failed")
    
    if autogen_result["success"]:
        print(f"✅ AutoGen approach: Success")
        print(f"   Response length: {len(autogen_result['response'])} characters")
        print(f"   Execution time: {autogen_result.get('execution_time', 'N/A'):.2f}s")
        print(f"   Total messages: {autogen_result.get('total_messages', 'N/A')}")
        print(f"   Agents involved: {', '.join(autogen_result.get('agents_used', []))}")
    else:
        print(f"❌ AutoGen approach: Failed")


def main():
    """Run all AutoGen demonstrations."""
    print("🚀 AutoGen Integration Demonstration")
    print("This demo shows how AutoGen has been integrated into the existing agent framework.")
    print("The integration is non-disruptive and works alongside existing agents.\n")
    
    try:
        # Run synchronous demos
        demo_basic_autogen_conversation()
        demo_autogen_research_workflow()
        demo_autogen_code_review_workflow()
        demo_traditional_vs_autogen()
        
        # Run async demo
        print("\n🔄 Running streaming demonstration...")
        asyncio.run(demo_autogen_streaming())
        
        print("\n" + "=" * 60)
        print("✅ All AutoGen demonstrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        agent_logger.error(f"AutoGen demo failed: {str(e)}")


if __name__ == "__main__":
    main()