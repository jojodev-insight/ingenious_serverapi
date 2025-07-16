# AutoGen Integration Guide

This document explains how AutoGen has been integrated into the existing agent framework, providing powerful multi-agent conversation capabilities without disrupting the current implementation flow.

## Overview

The AutoGen integration adds two new agent types to the existing framework:

1. **AutoGenAgent** - For flexible multi-agent conversations
2. **AutoGenWorkflowAgent** - For specialized workflows (research, code review, etc.)

These agents work seamlessly with the existing orchestrator and API, maintaining full compatibility with the current system.

## Key Features

### 🤖 Multi-Agent Conversations
- Create conversations between multiple specialized agents
- Each agent can have different roles and system messages
- Configurable conversation flow and termination conditions

### 🔬 Specialized Workflows
- **Research Workflow**: Multi-step research with specialized agents
- **Code Review Workflow**: Collaborative code analysis with security, performance, and quality reviewers
- **Custom Workflows**: Define your own multi-agent workflows

### 📡 Streaming Support
- Real-time streaming of conversations
- See agent interactions as they happen
- Progress monitoring and conversation tracking

### 🔄 Seamless Integration
- Works with existing orchestrator and API
- Compatible with OpenAI and DeepSeek providers
- Automatic fallback support
- Same logging and error handling as other agents

## Usage Examples

### 1. Basic AutoGen Conversation

```python
from api.orchestrator import TaskOrchestrator

orchestrator = TaskOrchestrator()

task_data = {
    "message": "Explain machine learning concepts and provide practical examples",
    "agents": [
        {
            "type": "user_proxy",
            "name": "Student",
            "description": "Student asking about machine learning"
        },
        {
            "type": "assistant",
            "name": "MLExpert",
            "description": "Machine Learning Expert",
            "system_message": "You are a machine learning expert who explains concepts clearly with practical examples."
        }
    ],
    "max_turns": 8
}

result = orchestrator.run_agent("autogen", task_data)
print(result["response"])
```

### 2. Research Workflow

```python
task_data = {
    "message": "Research the impact of AI on healthcare and write a comprehensive report",
    "workflow_type": "research"
}

result = orchestrator.run_agent("autogen_workflow", task_data)
print(result["response"])
```

### 3. Code Review Workflow

```python
code_sample = '''
def user_login(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    # ... rest of code
'''

task_data = {
    "message": f"Review this code for security and quality issues:\n\n{code_sample}",
    "workflow_type": "code_review"
}

result = orchestrator.run_agent("autogen_workflow", task_data)
print(result["response"])
```

### 4. Streaming Conversation

```python
import asyncio

async def stream_conversation():
    task_data = {
        "message": "Discuss renewable energy solutions",
        "agents": [
            {
                "type": "user_proxy",
                "name": "Moderator",
                "description": "Discussion moderator"
            },
            {
                "type": "assistant",
                "name": "EnergyExpert",
                "description": "Renewable energy specialist",
                "system_message": "You are a renewable energy expert."
            },
            {
                "type": "assistant",
                "name": "PolicyAnalyst",
                "description": "Energy policy analyst",
                "system_message": "You analyze energy policies and regulations."
            }
        ]
    }
    
    async for chunk in orchestrator.run_agent_stream("autogen", task_data):
        if chunk["success"]:
            print(f"[{chunk['type']}] {chunk.get('message', '')}")

asyncio.run(stream_conversation())
```

## API Endpoints

### AutoGen Conversation
```bash
POST /autogen/conversation
POST /autogen/conversation/stream
```

**Request Body:**
```json
{
  "message": "Your conversation starter",
  "agents": [
    {
      "type": "user_proxy",
      "name": "User",
      "description": "User description"
    },
    {
      "type": "assistant",
      "name": "Assistant",
      "description": "Assistant description",
      "system_message": "System message for the assistant"
    }
  ],
  "max_turns": 10,
  "provider": "openai",
  "enable_fallback": true
}
```

### AutoGen Workflow
```bash
POST /autogen/workflow
POST /autogen/workflow/stream
```

**Request Body:**
```json
{
  "message": "Your workflow task",
  "workflow_type": "research",  // or "code_review"
  "provider": "openai",
  "enable_fallback": true
}
```

## Agent Configuration

### Agent Types

1. **user_proxy**: Represents a user or coordinator in the conversation
   - Does not require a model
   - Used to initiate conversations and coordinate flow

2. **assistant**: AI assistant with specific expertise
   - Requires model configuration
   - Can have specialized system messages
   - Provides AI-powered responses

### System Messages

System messages define the agent's role and behavior:

```python
{
    "type": "assistant",
    "name": "DataScientist",
    "system_message": "You are a data scientist expert. Provide detailed analysis with statistical insights and practical recommendations."
}
```

### Workflow Types

1. **research**: Multi-agent research workflow
   - Researcher: Gathers information
   - Analyst: Analyzes findings
   - Writer: Creates comprehensive report

2. **code_review**: Collaborative code analysis
   - SecurityReviewer: Security vulnerabilities
   - PerformanceReviewer: Performance optimization
   - QualityReviewer: Code quality and best practices

3. **custom**: User-defined workflow
   - Specify your own agents and flow
   - Custom termination conditions

## Integration Details

### Non-Disruptive Design

The AutoGen integration is designed to be completely non-disruptive:

1. **Existing agents unchanged**: All current agents continue to work exactly as before
2. **Same orchestrator**: Uses the existing TaskOrchestrator without modifications
3. **Same API patterns**: Follows the same request/response patterns as other agents
4. **Same configuration**: Uses the same provider and model configuration system
5. **Same logging**: Integrates with the existing logging system

### Error Handling

AutoGen agents use the same error handling and fallback mechanisms:

- Automatic fallback from OpenAI to DeepSeek
- Comprehensive error logging
- Graceful degradation
- Timeout handling

### Performance

AutoGen agents are optimized for performance:

- Efficient async/await patterns
- Streaming support for long conversations
- Memory-efficient conversation handling
- Connection pooling and reuse

## Testing and Demos

### Running the Demos

1. **Basic Demo**: `python autogen_demo.py`
2. **Advanced Streaming**: `python advanced_streaming_demo.py`

### Demo Features

- Basic conversation examples
- Research workflow demonstration
- Code review workflow showcase
- Streaming conversation examples
- Comparison with traditional single-agent approaches

## Configuration

### Environment Variables

AutoGen uses the same environment variables as other agents:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_API_VERSION=2024-02-15-preview

# DeepSeek Configuration (for fallback)
DEEPSEEK_API_KEY=your_deepseek_key
```

### Model Configuration

```python
model_config = {
    "max_tokens": 4000,
    "temperature": 0.7,
    "top_p": 1.0
}
```

## Best Practices

### 1. Agent Design
- Give agents clear, specific roles
- Use descriptive names and system messages
- Limit the number of agents (3-5 for most workflows)

### 2. Conversation Flow
- Start with clear, specific prompts
- Set appropriate max_turns limits
- Use termination conditions effectively

### 3. Performance
- Use streaming for long conversations
- Monitor token usage
- Implement appropriate timeouts

### 4. Error Handling
- Always enable fallback for production
- Handle async exceptions properly
- Log conversation details for debugging

## Troubleshooting

### Common Issues

1. **"AutoGen provider not supported"**
   - Currently only OpenAI provider is supported for AutoGen
   - Ensure you're using `provider="openai"`

2. **Async/Await Issues**
   - Use `asyncio.run()` for streaming functions
   - Ensure proper async context

3. **Agent Creation Failures**
   - Check agent configuration format
   - Verify system messages are provided for assistant agents

4. **API Key Issues**
   - Ensure OpenAI API key is properly configured
   - Check Azure OpenAI configuration if using Azure

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger("core.logger").setLevel(logging.DEBUG)
```

## Future Enhancements

Planned improvements for the AutoGen integration:

1. **Additional Providers**: Support for Anthropic, Google, and other providers
2. **Advanced Workflows**: More specialized workflow types
3. **Conversation Memory**: Persistent conversation history
4. **Visual Interface**: Web-based conversation viewer
5. **Agent Marketplace**: Pre-built agent configurations

## Contributing

To contribute to the AutoGen integration:

1. Follow the existing code patterns
2. Maintain backward compatibility
3. Add comprehensive tests
4. Update documentation
5. Test with both OpenAI and DeepSeek providers

## Conclusion

The AutoGen integration provides powerful multi-agent conversation capabilities while maintaining full compatibility with the existing system. It's designed to be a natural extension of the current architecture, providing new capabilities without disrupting existing workflows.

For questions or issues, please refer to the troubleshooting section or check the existing agent examples for guidance.
