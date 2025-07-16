#!/usr/bin/env python3
"""
Advanced AutoGen streaming examples with orchestrator and multiple agents.

This script demonstrates advanced AutoGen streaming capabilities including:
- Multi-agent AutoGen conversations
- Research pipelines with AutoGen
- Code analysis with collaborative AutoGen agents
- Interactive brainstorming sessions
"""

import time

from api.orchestrator import TaskOrchestrator


def orchestrator_streaming_demo():
    """Demonstrate orchestrator-level streaming."""
    print("🚀 Orchestrator Streaming Demo\n")

    orchestrator = TaskOrchestrator()

    # Task data for content writing
    task_data = {
        "topic": "The Future of Artificial Intelligence",
        "style": "informative and engaging",
        "length": "300 words",
        "audience": "technology enthusiasts"
    }

    print("📝 Task: Write content about 'The Future of Artificial Intelligence'")
    print("🎯 Style: Informative and engaging (300 words)")
    print("\n" + "="*60 + "\n")

    print("🔄 Streaming from orchestrator...")

    start_time = time.time()
    content_chunks = []

    try:
        for chunk in orchestrator.run_agent_stream(
            agent_name="content_writer",
            task_data=task_data
        ):
            if chunk["success"]:
                if chunk["type"] == "orchestrator_start":
                    print(f"🚀 Orchestrator started job {chunk['job_id']}")
                    print(f"🤖 Agent: {chunk['agent_name']}")
                    print(f"⚡ Provider: {chunk['provider']}")
                    print(f"🧠 Model: {chunk['model_name']}")
                    print("\n📄 Content generation:")

                elif chunk["type"] == "start":
                    print("🔵 Agent started...")

                elif chunk["type"] == "chunk":
                    # Real-time content streaming
                    print(chunk["chunk"], end="", flush=True)
                    content_chunks.append(chunk["chunk"])

                elif chunk["type"] == "complete":
                    print("\n\n✅ Agent completed")

                elif chunk["type"] == "orchestrator_complete":
                    end_time = time.time()
                    print(f"🎉 Orchestrator completed in {chunk['execution_time']:.2f}s")
                    print(f"📊 Total content length: {len(''.join(content_chunks))} characters")

                elif chunk["type"] == "fallback_start":
                    print(f"\n⚠️  {chunk['message']}")
                    print(f"🔄 Switching to: {chunk['provider']}")

            else:
                print(f"❌ Error: {chunk['error']}")
                break

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    print("\n" + "="*60 + "\n")


def multi_agent_streaming_workflow():
    """Demonstrate a multi-agent streaming workflow."""
    print("🔗 Multi-Agent Streaming Workflow\n")

    orchestrator = TaskOrchestrator()

    # Raw data to process
    raw_text = """
    Recent studies have shown that machine learning algorithms are becoming increasingly 
    sophisticated. Deep learning models can now process vast amounts of data and identify 
    complex patterns. Neural networks have evolved to handle multiple types of input data 
    simultaneously. Computer vision systems can recognize objects with high accuracy. 
    Natural language processing has advanced to understand context and nuance. These 
    developments suggest that AI will continue to transform various industries in the 
    coming years. Healthcare, finance, transportation, and entertainment are all being 
    revolutionized by artificial intelligence technologies.
    """

    print("📊 Multi-step workflow: Raw Text → Summary → Enhanced Content")
    print("\n🔤 Raw input text:")
    print(raw_text.strip())
    print("\n" + "="*60 + "\n")

    # Step 1: Summarize the text
    print("📝 Step 1: Creating summary...")

    summary_result = None
    for chunk in orchestrator.run_agent_stream(
        agent_name="summary",
        task_data={"text": raw_text}
    ):
        if chunk["success"]:
            if chunk["type"] == "orchestrator_start":
                print(f"🚀 Summary job {chunk['job_id']} started")

            elif chunk["type"] == "chunk":
                print(chunk["chunk"], end="", flush=True)

            elif chunk["type"] == "complete":
                summary_result = chunk["response"]
                print("\n✅ Summary completed")
                break
        else:
            print(f"❌ Summary failed: {chunk['error']}")
            return

    if not summary_result:
        print("❌ Could not generate summary")
        return

    print("\n" + "-"*40 + "\n")

    # Step 2: Enhance the summary into engaging content
    print("✨ Step 2: Enhancing summary into engaging content...")

    enhancement_task = {
        "text": summary_result,
        "task": "Transform this summary into an engaging, well-structured article with compelling headlines and smooth transitions"
    }

    for chunk in orchestrator.run_agent_stream(
        agent_name="text_processor",
        task_data=enhancement_task
    ):
        if chunk["success"]:
            if chunk["type"] == "orchestrator_start":
                print(f"🚀 Enhancement job {chunk['job_id']} started")

            elif chunk["type"] == "chunk":
                print(chunk["chunk"], end="", flush=True)

            elif chunk["type"] == "complete":
                print("\n✅ Enhancement completed")
                break

            elif chunk["type"] == "orchestrator_complete":
                print(f"🎉 Workflow completed in {chunk['execution_time']:.2f}s")

        else:
            print(f"❌ Enhancement failed: {chunk['error']}")
            break

    print("\n" + "="*60 + "\n")


def streaming_performance_monitor():
    """Monitor streaming performance and provide real-time metrics."""
    print("📊 Streaming Performance Monitor\n")

    orchestrator = TaskOrchestrator()

    # Complex calculation task for performance testing
    calculation_task = {
        "expression": "Calculate the compound interest for an initial investment of $10,000 at 5% annual interest compounded quarterly for 10 years, then analyze the growth pattern",
        "show_work": True
    }

    print("🧮 Task: Complex compound interest calculation with analysis")
    print("\n📈 Performance metrics:")

    start_time = time.time()
    first_chunk_time = None
    chunk_count = 0
    character_count = 0

    try:
        for chunk in orchestrator.run_agent_stream(
            agent_name="calculator",
            task_data=calculation_task
        ):
            if chunk["success"]:
                current_time = time.time() - start_time

                if chunk["type"] == "orchestrator_start":
                    print("⏱️  Job started at t=0.00s")

                elif chunk["type"] == "chunk":
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        print(f"⚡ First chunk received at t={first_chunk_time:.2f}s")

                    chunk_count += 1
                    character_count += len(chunk["chunk"])

                    # Show periodic performance updates
                    if chunk_count % 10 == 0:
                        chars_per_sec = character_count / current_time if current_time > 0 else 0
                        print(f"📊 t={current_time:.2f}s | Chunks: {chunk_count} | Chars: {character_count} | Rate: {chars_per_sec:.1f} chars/s")

                    # Display the actual content
                    print(chunk["chunk"], end="", flush=True)

                elif chunk["type"] == "complete":
                    total_time = current_time
                    chars_per_sec = character_count / total_time if total_time > 0 else 0
                    chunks_per_sec = chunk_count / total_time if total_time > 0 else 0

                    print("\n\n📈 Final Performance Report:")
                    print(f"⏱️  Total time: {total_time:.2f}s")
                    print(f"⚡ Time to first chunk: {first_chunk_time:.2f}s")
                    print(f"📦 Total chunks: {chunk_count}")
                    print(f"📝 Total characters: {character_count}")
                    print(f"🚀 Average rate: {chars_per_sec:.1f} chars/s")
                    print(f"📊 Chunk rate: {chunks_per_sec:.1f} chunks/s")

            else:
                print(f"❌ Error: {chunk['error']}")
                break

    except Exception as e:
        print(f"❌ Monitoring failed: {str(e)}")

    print("\n" + "="*60 + "\n")


async def advanced_autogen_research_pipeline():
    """Advanced research pipeline with AutoGen agents."""
    print("🔬 Advanced AutoGen Research Pipeline")
    print("=" * 50)
    
    orchestrator = TaskOrchestrator()
    
    # Multi-step research workflow
    research_topic = "The impact of quantum computing on cybersecurity"
    
    task_data = {
        "message": f"Conduct comprehensive research on: {research_topic}. Include current state, future implications, and recommendations.",
        "agents": [
            {
                "type": "user_proxy",
                "name": "ResearchDirector",
                "description": "Research director coordinating the investigation"
            },
            {
                "type": "assistant",
                "name": "QuantumExpert",
                "description": "Quantum computing specialist",
                "system_message": "You are a quantum computing expert. Provide detailed technical insights about quantum computing capabilities, current limitations, and future potential in cybersecurity applications."
            },
            {
                "type": "assistant",
                "name": "CyberSecurityAnalyst",
                "description": "Cybersecurity specialist",
                "system_message": "You are a cybersecurity expert. Analyze how quantum computing affects current encryption methods, security protocols, and defensive strategies."
            }
        ],
        "max_turns": 10
    }
    
    print(f"📝 Research Topic: {research_topic}")
    print("🤖 Starting multi-agent research conversation...\n")
    
    async for chunk in orchestrator.run_agent_stream("autogen", task_data):
        if chunk["success"]:
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "start":
                print("🚀 Research initiated")
                
            elif chunk_type == "agents_created":
                agents = chunk.get("agents", [])
                print(f"👥 Research team: {', '.join(agents)}")
                
            elif chunk_type == "message":
                source = chunk.get("source", "Unknown")
                content = chunk.get("content", "")
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"🗣️  {source}: {preview}")
                
            elif chunk_type == "complete":
                print("✅ Research completed!")
                final_response = chunk.get("response", "")
                print(f"📄 Final report length: {len(final_response)} characters")
        else:
            print(f"❌ Error: {chunk.get('error', 'Unknown error')}")
            break


async def collaborative_autogen_code_analysis():
    """Collaborative code analysis with multiple AutoGen agents."""
    print("\n🔍 Collaborative AutoGen Code Analysis")
    print("=" * 50)
    
    orchestrator = TaskOrchestrator()
    
    code_sample = '''
def user_login(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    result = db.execute(query)
    if result:
        session['user_id'] = result[0]['id']
        return True
    return False
'''
    
    task_data = {
        "message": f"Analyze this code for security and quality issues:\n\n{code_sample}",
        "agents": [
            {
                "type": "user_proxy",
                "name": "TechLead",
                "description": "Technical lead requesting code review"
            },
            {
                "type": "assistant",
                "name": "SecurityAuditor",
                "description": "Security specialist",
                "system_message": "You are a cybersecurity expert. Identify security vulnerabilities and provide remediation steps."
            },
            {
                "type": "assistant",
                "name": "QualityReviewer",
                "description": "Code quality specialist",
                "system_message": "You are a senior engineer focused on code quality and best practices."
            }
        ],
        "max_turns": 8
    }
    
    print("👨‍💻 Starting collaborative code analysis...\n")
    
    async for chunk in orchestrator.run_agent_stream("autogen", task_data):
        if chunk["success"]:
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "message":
                source = chunk.get("source", "Unknown")
                content = chunk.get("content", "")
                preview = content[:150] + "..." if len(content) > 150 else content
                print(f"🔍 {source}: {preview}")
                
            elif chunk_type == "complete":
                print("✅ Code analysis completed!")
                response = chunk.get("response", "")
                print(f"📊 Analysis summary: {response[:200]}...")
        else:
            print(f"❌ Analysis error: {chunk.get('error', 'Unknown error')}")
            break


async def interactive_autogen_brainstorming():
    """Interactive brainstorming session with AutoGen agents."""
    print("\n💡 Interactive AutoGen Brainstorming")
    print("=" * 50)
    
    orchestrator = TaskOrchestrator()
    
    topic = "Innovative solutions for sustainable urban transportation"
    
    task_data = {
        "message": f"Brainstorm innovative solutions for: {topic}",
        "agents": [
            {
                "type": "user_proxy",
                "name": "Moderator",
                "description": "Session moderator"
            },
            {
                "type": "assistant",
                "name": "Innovator",
                "description": "Innovation expert",
                "system_message": "You are an innovation expert. Generate creative, cutting-edge solutions."
            },
            {
                "type": "assistant",
                "name": "Analyst",
                "description": "Feasibility analyst",
                "system_message": "You are an analyst. Evaluate the feasibility and impact of proposed solutions."
            }
        ],
        "max_turns": 12
    }
    
    print(f"🎯 Brainstorming: {topic}")
    print("🧠 Starting collaborative session...\n")
    
    idea_count = 0
    
    async for chunk in orchestrator.run_agent_stream("autogen", task_data):
        if chunk["success"]:
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "message":
                source = chunk.get("source", "Unknown")
                content = chunk.get("content", "")
                
                # Count ideas
                if any(keyword in content.lower() for keyword in ["solution", "idea", "propose"]):
                    idea_count += 1
                
                preview = content[:120] + "..." if len(content) > 120 else content
                print(f"💭 {source}: {preview}")
                
            elif chunk_type == "complete":
                print(f"✅ Brainstorming completed! Generated {idea_count} ideas")
                response = chunk.get("response", "")
                print(f"📋 Session summary: {response[:200]}...")
        else:
            print(f"❌ Brainstorming error: {chunk.get('error', 'Unknown error')}")
            break


def main():
    """Main demonstration function."""
    print("🌟 Advanced Agent Streaming Demonstrations")
    print("=" * 60)
    print()

    try:
        # Basic orchestrator streaming
        orchestrator_streaming_demo()

        # Multi-agent workflow
        multi_agent_streaming_workflow()

        # Performance monitoring
        streaming_performance_monitor()

        # AutoGen demonstrations
        print("\n" + "="*60)
        print("🤖 AUTOGEN STREAMING DEMONSTRATIONS")
        print("="*60)
        
        # Run AutoGen demos asynchronously
        import asyncio
        asyncio.run(run_autogen_demos())

        print("🎊 All streaming demonstrations completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


async def run_autogen_demos():
    """Run all AutoGen streaming demonstrations."""
    await advanced_autogen_research_pipeline()
    await collaborative_autogen_code_analysis()
    await interactive_autogen_brainstorming()


if __name__ == "__main__":
    main()
