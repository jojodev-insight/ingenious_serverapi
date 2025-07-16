#!/usr/bin/env python3
"""
Example script demonstrating streaming output from agents.

This script shows how to use the new streaming capabilities
of the agent framework with both Azure OpenAI and standard OpenAI.
"""

import asyncio
import time
from typing import Dict, Any

from agents.summary_agent import SummaryAgent
from agents.text_processor_agent import TextProcessorAgent
from core.config import load_config


def stream_agent_example():
    """Demonstrate streaming agent output."""
    print("🔄 Agent Streaming Output Demo\n")
    
    # Load configuration
    config = load_config()
    
    # Create a summary agent
    summary_agent = SummaryAgent(config)
    
    # Sample task data
    task_data = {
        "text": """
        Artificial Intelligence (AI) has revolutionized many industries over the past decade. 
        From healthcare to finance, from transportation to entertainment, AI technologies 
        are transforming how we work and live. Machine learning algorithms can now analyze 
        vast amounts of data, identify patterns, and make predictions with remarkable accuracy. 
        Natural language processing enables computers to understand and generate human language, 
        while computer vision allows machines to interpret visual information. As AI continues 
        to evolve, we can expect even more innovative applications that will further enhance 
        human capabilities and solve complex global challenges.
        """
    }
    
    print("📝 Input text:")
    print(task_data["text"].strip())
    print("\n" + "="*60 + "\n")
    
    # Example 1: Non-streaming execution
    print("🔵 Non-streaming execution:")
    start_time = time.time()
    
    result = summary_agent.execute(task_data, stream=False)
    
    end_time = time.time()
    
    if result["success"]:
        print(f"✅ Complete response ({end_time - start_time:.2f}s):")
        print(result["response"])
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Streaming execution
    print("🟡 Streaming execution:")
    start_time = time.time()
    
    print("🔄 Streaming response:")
    
    try:
        for chunk in summary_agent.execute_stream(task_data):
            if chunk["success"]:
                if chunk["type"] == "start":
                    print(f"🚀 Started streaming from {chunk['agent_name']} using {chunk['provider']}")
                elif chunk["type"] == "chunk":
                    # Print chunk content without newline to show streaming effect
                    print(chunk["chunk"], end="", flush=True)
                elif chunk["type"] == "complete":
                    end_time = time.time()
                    print(f"\n\n✅ Streaming complete ({end_time - start_time:.2f}s)")
                    print(f"📊 Agent: {chunk['agent_name']}")
                    print(f"🤖 Provider: {chunk['provider']}")
                    print(f"🧠 Model: {chunk['model_name']}")
            else:
                print(f"❌ Streaming error: {chunk['error']}")
                break
                
    except Exception as e:
        print(f"❌ Exception during streaming: {str(e)}")
    
    print("\n" + "="*60 + "\n")


def compare_streaming_modes():
    """Compare streaming vs non-streaming performance."""
    print("⚡ Performance Comparison: Streaming vs Non-Streaming\n")
    
    config = load_config()
    text_processor = TextProcessorAgent(config)
    
    task_data = {
        "text": "The quick brown fox jumps over the lazy dog. " * 50,  # Longer text
        "task": "Rewrite this text to be more engaging and creative."
    }
    
    # Non-streaming
    print("🔵 Non-streaming mode:")
    start_time = time.time()
    result = text_processor.execute(task_data, stream=False)
    non_stream_time = time.time() - start_time
    
    if result["success"]:
        print(f"✅ Response received in {non_stream_time:.2f}s")
        print(f"📏 Response length: {len(result['response'])} characters")
    
    print()
    
    # Streaming
    print("🟡 Streaming mode:")
    start_time = time.time()
    first_chunk_time = None
    total_chunks = 0
    
    try:
        for chunk in text_processor.execute_stream(task_data):
            if chunk["success"]:
                if chunk["type"] == "start":
                    print("🚀 Stream started...")
                elif chunk["type"] == "chunk":
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                    total_chunks += 1
                elif chunk["type"] == "complete":
                    total_time = time.time() - start_time
                    print(f"✅ Stream completed in {total_time:.2f}s")
                    print(f"⚡ First chunk received in {first_chunk_time:.2f}s")
                    print(f"📦 Total chunks: {total_chunks}")
                    print(f"📏 Response length: {len(chunk['response'])} characters")
    
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
    
    print(f"\n📈 Time to first response: {first_chunk_time:.2f}s vs {non_stream_time:.2f}s")
    print(f"🎯 Streaming advantage: {((non_stream_time - first_chunk_time) / non_stream_time * 100):.1f}% faster to first content")


def main():
    """Main demonstration function."""
    print("🤖 Agent Streaming Output Demonstration")
    print("=" * 50)
    print()
    
    try:
        # Basic streaming example
        stream_agent_example()
        
        # Performance comparison
        compare_streaming_modes()
        
        print("🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
