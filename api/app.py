"""FastAPI application for the Autogen orchestrator."""

import json
from contextlib import asynccontextmanager
from typing import Any, List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.orchestrator import TaskOrchestrator
from core import config, orchestrator_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    orchestrator_logger.info("FastAPI application starting up")

    # Log current configuration
    config_info = config.get_all_config_info()
    orchestrator_logger.info(f"Configuration loaded - Default provider: {config_info['providers']['default_provider']}")

    if config_info['providers']['openai']['is_azure']:
        orchestrator_logger.info(f"Azure OpenAI configured - Endpoint: {config_info['providers']['openai']['api_base']}, Deployment: {config_info['providers']['openai']['azure_deployment']}")
    else:
        orchestrator_logger.info(f"OpenAI configured - Base URL: {config_info['providers']['openai']['api_base']}")

    if config_info['providers']['deepseek']['api_key_set']:
        orchestrator_logger.info(f"DeepSeek configured - Base URL: {config_info['providers']['deepseek']['api_base']}")

    yield
    # Shutdown
    orchestrator_logger.info("FastAPI application shutting down")


# Pydantic models for request/response
class ModelConfigRequest(BaseModel):
    """Model configuration parameters."""
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    temperature: float | None = Field(None, description="Sampling temperature (0.0-2.0)")
    top_p: float | None = Field(None, description="Nucleus sampling parameter (0.0-1.0)")
    frequency_penalty: float | None = Field(None, description="Frequency penalty (-2.0 to 2.0)")
    presence_penalty: float | None = Field(None, description="Presence penalty (-2.0 to 2.0)")


class TaskRequest(BaseModel):
    """Request model for single agent task."""
    agent_name: str = Field(..., description="Name of the agent to run")
    task_data: dict[str, Any] = Field(..., description="Data for the agent task")
    provider: str | None = Field(None, description="LLM provider ('openai', 'deepseek', 'anthropic')")
    model_name: str | None = Field(None, description="Specific model to use")
    llm_config: ModelConfigRequest | None = Field(None, description="Model configuration parameters")
    agent_config: dict[str, Any] | None = Field(None, description="Additional agent configuration")


class EnhancedWorkflowStep(BaseModel):
    """Enhanced model for a single workflow step."""
    agent: str = Field(..., description="Agent name")
    task_data: dict[str, Any] = Field(..., description="Task data for this step")
    provider: str | None = Field(None, description="LLM provider for this step")
    model_name: str | None = Field(None, description="Specific model for this step")
    llm_config: ModelConfigRequest | None = Field(None, description="Model config for this step")
    agent_config: dict[str, Any] | None = Field(None, description="Agent config for this step")


class WorkflowStep(BaseModel):
    """Model for a single workflow step (backward compatibility)."""
    agent: str = Field(..., description="Agent name")
    task_data: dict[str, Any] = Field(..., description="Task data for this step")


class WorkflowRequest(BaseModel):
    """Request model for multi-agent workflow."""
    workflow: list[WorkflowStep] = Field(..., description="List of workflow steps")
    provider: str | None = Field(None, description="LLM provider ('openai', 'deepseek', 'anthropic')")


class EnhancedWorkflowRequest(BaseModel):
    """Enhanced request model for multi-agent workflow."""
    workflow: list[EnhancedWorkflowStep] = Field(..., description="List of enhanced workflow steps")
    default_provider: str | None = Field(None, description="Default LLM provider")
    default_llm_config: ModelConfigRequest | None = Field(None, description="Default model configuration")


class TaskResponse(BaseModel):
    """Response model for task execution."""
    success: bool
    job_id: str
    execution_time: float
    response: str | None = None
    error: str | None = None
    agent_name: str | None = None
    provider: str | None = None
    orchestrator: str


class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    success: bool
    job_id: str
    execution_time: float
    workflow_steps: int
    completed_steps: int
    results: list[dict[str, Any]]
    orchestrator: str
    error: str | None = None

    class Config:
        # Allow arbitrary types to handle complex nested structures
        arbitrary_types_allowed = True


# Create FastAPI app
app = FastAPI(
    title="Autogen Agent Orchestrator",
    description="API for orchestrating Autogen agents with OpenAI and DeepSeek support",
    version="1.0.0",
    lifespan=lifespan
)

# Create orchestrator instance
orchestrator = TaskOrchestrator()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Autogen Agent Orchestrator API",
        "version": "1.0.0",
        "available_endpoints": [
            "/agents - List available agents",
            "/run-agent - Run a single agent task",
            "/run-agent-stream - Run a single agent task with streaming",
            "/run-workflow - Run a multi-agent workflow",
            "/run-workflow-stream - Run a multi-agent workflow with streaming",
            "/run-enhanced-workflow - Run enhanced workflow with per-step config"
        ]
    }


@app.get("/agents")
async def list_agents():
    """List all available agents."""
    agents = orchestrator.list_agents()
    return {
        "agents": agents,
        "count": len(agents),
        "default_provider": config.default_provider
    }


@app.post("/run-agent", response_model=TaskResponse)
async def run_agent(request: TaskRequest):
    """Run a single agent task with enhanced configuration.
    
    Args:
        request: Task request containing agent name, task data, and optional configurations.
        
    Returns:
        Task execution result.
    """
    try:
        orchestrator_logger.info(f"API request to run agent '{request.agent_name}' with provider: {request.provider or config.default_provider}")

        # Log current configuration
        if request.provider == "openai" or (not request.provider and config.default_provider == "openai"):
            if config.is_azure_openai:
                orchestrator_logger.info(f"Using Azure OpenAI endpoint: {config.openai_api_base}, deployment: {config.azure_deployment_name}, API version: {config.azure_api_version}")
            else:
                orchestrator_logger.info(f"Using OpenAI base URL: {config.openai_api_base}")
        elif request.provider == "deepseek" or (not request.provider and config.default_provider == "deepseek"):
            orchestrator_logger.info(f"Using DeepSeek base URL: {config.deepseek_api_base}")

        # Convert model config to dict if provided
        model_config = None
        if request.llm_config:
            model_config = {k: v for k, v in request.llm_config.dict().items() if v is not None}

        result = orchestrator.run_agent(
            agent_name=request.agent_name,
            task_data=request.task_data,
            provider=request.provider,
            model_name=request.model_name,
            model_config=model_config,
            agent_config=request.agent_config
        )

        return TaskResponse(**result)

    except Exception as e:
        orchestrator_logger.error(f"API error running agent '{request.agent_name}'", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-workflow")
async def run_workflow(request: WorkflowRequest):
    """Run a multi-agent workflow.
    
    Args:
        request: Workflow request containing steps and optional provider.
        
    Returns:
        Workflow execution result.
    """
    try:
        orchestrator_logger.info(f"API request to run workflow with {len(request.workflow)} steps using provider: {request.provider or config.default_provider}")

        # Log current configuration
        if request.provider == "openai" or (not request.provider and config.default_provider == "openai"):
            if config.is_azure_openai:
                orchestrator_logger.info(f"Using Azure OpenAI endpoint: {config.openai_api_base}, deployment: {config.azure_deployment_name}, API version: {config.azure_api_version}")
            else:
                orchestrator_logger.info(f"Using OpenAI base URL: {config.openai_api_base}")
        elif request.provider == "deepseek" or (not request.provider and config.default_provider == "deepseek"):
            orchestrator_logger.info(f"Using DeepSeek base URL: {config.deepseek_api_base}")

        workflow_data = [step.dict() for step in request.workflow]
        result = orchestrator.run_multi_agent_workflow(
            workflow=workflow_data,
            provider=request.provider
        )

        # Clean the result for proper JSON serialization
        # Convert complex objects to simple representations
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Convert objects with __dict__ to their dictionary representation
                return clean_for_json(obj.__dict__)
            else:
                # For primitive types, return as-is
                return obj

        cleaned_result = clean_for_json(result)
        return cleaned_result

    except Exception as e:
        orchestrator_logger.error("API error running workflow", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-enhanced-workflow", response_model=WorkflowResponse)
async def run_enhanced_workflow(request: EnhancedWorkflowRequest):
    """Run a multi-agent workflow with enhanced per-step configuration.
    
    Args:
        request: Enhanced workflow request with per-step configurations.
        
    Returns:
        Workflow execution result.
    """
    try:
        orchestrator_logger.info(f"API request to run enhanced workflow with {len(request.workflow)} steps using default provider: {request.default_provider or config.default_provider}")

        # Log current configuration
        default_provider = request.default_provider or config.default_provider
        if default_provider == "openai":
            if config.is_azure_openai:
                orchestrator_logger.info(f"Using Azure OpenAI endpoint: {config.openai_api_base}, deployment: {config.azure_deployment_name}, API version: {config.azure_api_version}")
            else:
                orchestrator_logger.info(f"Using OpenAI base URL: {config.openai_api_base}")
        elif default_provider == "deepseek":
            orchestrator_logger.info(f"Using DeepSeek base URL: {config.deepseek_api_base}")

        # Process enhanced workflow steps
        workflow_data = []
        for step in request.workflow:
            step_data = step.dict()
            # Use step-specific config or fall back to defaults
            if not step_data.get("provider"):
                step_data["provider"] = request.default_provider
            if not step_data.get("llm_config") and request.default_llm_config:
                step_data["model_config"] = {k: v for k, v in request.default_llm_config.dict().items() if v is not None}
            workflow_data.append(step_data)

        # Note: This would require updating run_multi_agent_workflow to support enhanced config
        # For now, calling the basic version
        result = orchestrator.run_multi_agent_workflow(
            workflow=workflow_data,
            provider=request.default_provider
        )

        return WorkflowResponse(**result)

    except Exception as e:
        orchestrator_logger.error("API error running enhanced workflow", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers")
async def list_providers():
    """List all available providers and their models."""
    from agents import ProviderConfig

    try:
        providers = ProviderConfig.list_models()
        return {
            "providers": providers,
            "default_provider": config.default_provider
        }
    except Exception as e:
        orchestrator_logger.error("API error listing providers", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent-types")
async def list_agent_types():
    """List all available agent types."""
    from agents import AgentFactory

    try:
        agent_types = AgentFactory.list_agent_types()
        return {
            "agent_types": agent_types,
            "available_agents": orchestrator.list_agents()
        }
    except Exception as e:
        orchestrator_logger.error("API error listing agent types", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/create-custom-agent")
# async def create_custom_agent(
#     name: str,
#     template_name: str,
#     system_message: str,
#     provider: Optional[str] = None,
#     model_name: Optional[str] = None
# ):
#     """Create and register a custom agent.

#     Args:
#         name: Agent name.
#         template_name: Template file name.
#         system_message: System message for the agent.
#         provider: LLM provider.
#         model_name: Specific model to use.

#     Returns:
#         Agent creation result.
#     """
#     try:
#         from agents import AgentFactory, CustomAgent

#         # Create custom agent class
#         class DynamicCustomAgent(CustomAgent):
#             def __init__(self, provider=None):
#                 super().__init__(
#                     name=name,
#                     template_name=template_name,
#                     system_message=system_message,
#                     provider=provider,
#                     model_name=model_name
#                 )

#         # Register with orchestrator
#         orchestrator.register_agent(name.lower().replace(" ", "_"), DynamicCustomAgent)

#         return {
#             "success": True,
#             "message": f"Custom agent '{name}' created and registered",
#             "agent_name": name.lower().replace(" ", "_")
#         }

#     except Exception as e:
#         orchestrator_logger.error(f"API error creating custom agent '{name}'", {"error": str(e)})
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-agent-stream")
async def run_agent_stream(request: TaskRequest):
    """Run a single agent task with streaming output.
    
    Args:
        request: Task request containing agent name, task data, and optional configurations.
        
    Returns:
        Server-Sent Events stream of task execution.
    """
    try:
        orchestrator_logger.info(f"API streaming request to run agent '{request.agent_name}' with provider: {request.provider or config.default_provider}")

        # Convert model config to dict if provided
        model_config = None
        if request.llm_config:
            model_config = {k: v for k, v in request.llm_config.dict().items() if v is not None}

        def generate_stream():
            """Generate SSE stream from orchestrator."""
            try:
                for chunk in orchestrator.run_agent_stream(
                    agent_name=request.agent_name,
                    task_data=request.task_data,
                    provider=request.provider,
                    model_name=request.model_name,
                    model_config=model_config,
                    agent_config=request.agent_config
                ):
                    # Convert chunk to JSON and send as SSE
                    chunk_json = json.dumps(chunk)
                    yield f"data: {chunk_json}\n\n"

                # Send completion signal
                completion_chunk = {"type": "stream_complete", "success": True}
                yield f"data: {json.dumps(completion_chunk)}\n\n"

            except Exception as e:
                error_chunk = {"type": "stream_error", "success": False, "error": str(e)}
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )

    except Exception as e:
        orchestrator_logger.error(f"API streaming error for agent '{request.agent_name}'", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-workflow-stream")
async def run_workflow_stream(request: WorkflowRequest):
    """Run a multi-agent workflow with streaming output.
    
    Args:
        request: Workflow request containing steps and optional provider.
        
    Returns:
        Server-Sent Events stream of workflow execution.
    """
    try:
        orchestrator_logger.info(f"API streaming request to run workflow with {len(request.workflow)} steps using provider: {request.provider or config.default_provider}")

        def generate_workflow_stream():
            """Generate SSE stream from workflow execution."""
            try:
                workflow_data = [step.dict() for step in request.workflow]

                # For now, execute workflow steps sequentially with streaming
                # This is a simplified version - could be enhanced for true workflow streaming
                step_results = []

                for i, step in enumerate(workflow_data):
                    step_info = {
                        "type": "workflow_step_start",
                        "success": True,
                        "step_number": i + 1,
                        "total_steps": len(workflow_data),
                        "agent_name": step["agent"]
                    }
                    yield f"data: {json.dumps(step_info)}\n\n"

                    # Stream the individual agent execution
                    step_result = None
                    for chunk in orchestrator.run_agent_stream(
                        agent_name=step["agent"],
                        task_data=step["task_data"],
                        provider=request.provider
                    ):
                        # Add step context to chunk
                        chunk["workflow_step"] = i + 1
                        chunk["total_steps"] = len(workflow_data)

                        chunk_json = json.dumps(chunk)
                        yield f"data: {chunk_json}\n\n"

                        if chunk.get("type") == "complete":
                            step_result = chunk.get("response")

                    step_results.append({
                        "step": i + 1,
                        "agent": step["agent"],
                        "result": step_result,
                        "success": step_result is not None
                    })

                    step_complete = {
                        "type": "workflow_step_complete",
                        "success": True,
                        "step_number": i + 1,
                        "total_steps": len(workflow_data)
                    }
                    yield f"data: {json.dumps(step_complete)}\n\n"

                # Send workflow completion
                completion_chunk = {
                    "type": "workflow_complete",
                    "success": True,
                    "completed_steps": len(step_results),
                    "total_steps": len(workflow_data),
                    "results": step_results
                }
                yield f"data: {json.dumps(completion_chunk)}\n\n"

            except Exception as e:
                error_chunk = {"type": "workflow_error", "success": False, "error": str(e)}
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            generate_workflow_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )

    except Exception as e:
        orchestrator_logger.error("API streaming error running workflow", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": orchestrator_logger._get_thread_info()["timestamp"]
    }


@app.get("/config")
async def get_config():
    """Get current configuration information."""
    try:
        return {
            "status": "success",
            "config": config.get_all_config_info()
        }
    except Exception as e:
        orchestrator_logger.error("Error retrieving configuration", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


# AutoGen-specific models
class AutoGenAgentConfig(BaseModel):
    """Configuration for individual AutoGen agents."""
    type: str = Field(..., description="Agent type: 'user_proxy' or 'assistant'")
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    system_message: Optional[str] = Field(None, description="System message for assistant agents")


class AutoGenTaskRequest(BaseModel):
    """Request model for AutoGen conversation task."""
    message: str = Field(..., description="Initial message to start the conversation")
    agents: Optional[List[AutoGenAgentConfig]] = Field(None, description="List of agents for the conversation")
    max_turns: Optional[int] = Field(10, description="Maximum number of conversation turns")
    provider: Optional[str] = Field(None, description="LLM provider to use (openai, deepseek)")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    model_config: Optional[ModelConfigRequest] = Field(None, description="Model configuration")
    enable_fallback: bool = Field(True, description="Enable fallback to DeepSeek if OpenAI fails")


class AutoGenWorkflowRequest(BaseModel):
    """Request model for AutoGen workflow task."""
    message: str = Field(..., description="Initial message for the workflow")
    workflow_type: str = Field("research", description="Type of workflow: 'research', 'code_review', or 'custom'")
    workflow_config: Optional[Dict[str, Any]] = Field(None, description="Custom workflow configuration")
    provider: Optional[str] = Field(None, description="LLM provider to use")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    model_config: Optional[ModelConfigRequest] = Field(None, description="Model configuration")
    enable_fallback: bool = Field(True, description="Enable fallback to DeepSeek if OpenAI fails")


class AutoGenConversationResponse(BaseModel):
    """Response model for AutoGen conversations."""
    success: bool
    response: str
    conversation: Optional[List[Dict[str, Any]]] = None
    total_messages: Optional[int] = None
    agents_used: Optional[List[str]] = None
    workflow_type: Optional[str] = None
    job_id: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


# AutoGen API endpoints
@app.post("/autogen/conversation", response_model=AutoGenConversationResponse)
async def run_autogen_conversation(request: AutoGenTaskRequest):
    """Run an AutoGen multi-agent conversation.
    
    This endpoint creates a conversation between multiple AutoGen agents.
    You can specify custom agents or use default configurations.
    """
    orchestrator = TaskOrchestrator()
    
    try:
        # Convert request to task data
        task_data = {
            "message": request.message,
            "max_turns": request.max_turns
        }
        
        # Add agents configuration if provided
        if request.agents:
            task_data["agents"] = [agent.model_dump() for agent in request.agents]
        
        # Convert model config if provided
        model_config = None
        if request.model_config:
            model_config = request.model_config.model_dump(exclude_none=True)
        
        orchestrator_logger.info(f"Running AutoGen conversation with {len(request.agents or [])} agents")
        
        result = orchestrator.run_agent(
            agent_name="autogen",
            task_data=task_data,
            provider=request.provider,
            model_name=request.model_name,
            model_config=model_config,
            enable_fallback=request.enable_fallback
        )
        
        return AutoGenConversationResponse(**result)
        
    except Exception as e:
        orchestrator_logger.error("AutoGen conversation failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/autogen/workflow", response_model=AutoGenConversationResponse)
async def run_autogen_workflow(request: AutoGenWorkflowRequest):
    """Run an AutoGen workflow with specialized agents.
    
    This endpoint runs predefined workflows (research, code_review) or custom workflows
    using AutoGen's multi-agent conversation capabilities.
    """
    orchestrator = TaskOrchestrator()
    
    try:
        # Convert request to task data
        task_data = {
            "message": request.message,
            "workflow_type": request.workflow_type
        }
        
        # Add custom workflow config if provided
        if request.workflow_config:
            task_data["workflow_config"] = request.workflow_config
        
        # Convert model config if provided
        model_config = None
        if request.model_config:
            model_config = request.model_config.model_dump(exclude_none=True)
        
        orchestrator_logger.info(f"Running AutoGen {request.workflow_type} workflow")
        
        result = orchestrator.run_agent(
            agent_name="autogen_workflow",
            task_data=task_data,
            provider=request.provider,
            model_name=request.model_name,
            model_config=model_config,
            enable_fallback=request.enable_fallback
        )
        
        return AutoGenConversationResponse(**result)
        
    except Exception as e:
        orchestrator_logger.error("AutoGen workflow failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/autogen/conversation/stream")
async def stream_autogen_conversation(request: AutoGenTaskRequest):
    """Stream an AutoGen multi-agent conversation.
    
    This endpoint provides real-time streaming of AutoGen conversations,
    allowing you to see the conversation develop step by step.
    """
    orchestrator = TaskOrchestrator()
    
    async def generate_stream():
        try:
            # Convert request to task data
            task_data = {
                "message": request.message,
                "max_turns": request.max_turns
            }
            
            # Add agents configuration if provided
            if request.agents:
                task_data["agents"] = [agent.model_dump() for agent in request.agents]
            
            # Convert model config if provided
            model_config = None
            if request.model_config:
                model_config = request.model_config.model_dump(exclude_none=True)
            
            orchestrator_logger.info(f"Streaming AutoGen conversation with {len(request.agents or [])} agents")
            
            async for chunk in orchestrator.run_agent_stream(
                agent_name="autogen",
                task_data=task_data,
                provider=request.provider,
                model_name=request.model_name,
                model_config=model_config,
                enable_fallback=request.enable_fallback
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            error_chunk = {
                "success": False,
                "error": str(e),
                "type": "error"
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/autogen/workflow/stream")
async def stream_autogen_workflow(request: AutoGenWorkflowRequest):
    """Stream an AutoGen workflow execution.
    
    This endpoint provides real-time streaming of AutoGen workflow execution,
    showing the progress of multi-agent collaboration in specialized workflows.
    """
    orchestrator = TaskOrchestrator()
    
    async def generate_stream():
        try:
            # Convert request to task data
            task_data = {
                "message": request.message,
                "workflow_type": request.workflow_type
            }
            
            # Add custom workflow config if provided
            if request.workflow_config:
                task_data["workflow_config"] = request.workflow_config
            
            # Convert model config if provided
            model_config = None
            if request.model_config:
                model_config = request.model_config.model_dump(exclude_none=True)
            
            orchestrator_logger.info(f"Streaming AutoGen {request.workflow_type} workflow")
            
            async for chunk in orchestrator.run_agent_stream(
                agent_name="autogen_workflow",
                task_data=task_data,
                provider=request.provider,
                model_name=request.model_name,
                model_config=model_config,
                enable_fallback=request.enable_fallback
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            error_chunk = {
                "success": False,
                "error": str(e),
                "type": "error"
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
