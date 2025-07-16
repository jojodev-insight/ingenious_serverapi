#!/usr/bin/env python3
"""
Advanced streaming examples with orchestrator and multiple agents.

This script demonstrates advanced streaming capabilities including:
- Orchestrator-level streaming
- Multi-agent streaming workflows
- Real-time progress monitoring
"""

import time
from typing import Dict, Any

from api.orchestrator import TaskOrchestrator
from core import config


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
                    print(f"\n\n✅ Agent completed")
                    
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
                print(f"\n✅ Summary completed")
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
                print(f"\n✅ Enhancement completed")
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
                    print(f"⏱️  Job started at t=0.00s")
                    
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
                    
                    print(f"\n\n📈 Final Performance Report:")
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
        
        print("🎊 All streaming demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
