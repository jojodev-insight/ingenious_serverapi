"""FastAPI application for the Autogen orchestrator."""

from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
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
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature (0.0-2.0)")
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter (0.0-1.0)")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty (-2.0 to 2.0)")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty (-2.0 to 2.0)")


class TaskRequest(BaseModel):
    """Request model for single agent task."""
    agent_name: str = Field(..., description="Name of the agent to run")
    task_data: Dict[str, Any] = Field(..., description="Data for the agent task")
    provider: Optional[str] = Field(None, description="LLM provider ('openai', 'deepseek', 'anthropic')")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    llm_config: Optional[ModelConfigRequest] = Field(None, description="Model configuration parameters")
    agent_config: Optional[Dict[str, Any]] = Field(None, description="Additional agent configuration")


class EnhancedWorkflowStep(BaseModel):
    """Enhanced model for a single workflow step."""
    agent: str = Field(..., description="Agent name")
    task_data: Dict[str, Any] = Field(..., description="Task data for this step")
    provider: Optional[str] = Field(None, description="LLM provider for this step")
    model_name: Optional[str] = Field(None, description="Specific model for this step")
    llm_config: Optional[ModelConfigRequest] = Field(None, description="Model config for this step")
    agent_config: Optional[Dict[str, Any]] = Field(None, description="Agent config for this step")


class WorkflowStep(BaseModel):
    """Model for a single workflow step (backward compatibility)."""
    agent: str = Field(..., description="Agent name")
    task_data: Dict[str, Any] = Field(..., description="Task data for this step")


class WorkflowRequest(BaseModel):
    """Request model for multi-agent workflow."""
    workflow: List[WorkflowStep] = Field(..., description="List of workflow steps")
    provider: Optional[str] = Field(None, description="LLM provider ('openai', 'deepseek', 'anthropic')")


class EnhancedWorkflowRequest(BaseModel):
    """Enhanced request model for multi-agent workflow."""
    workflow: List[EnhancedWorkflowStep] = Field(..., description="List of enhanced workflow steps")
    default_provider: Optional[str] = Field(None, description="Default LLM provider")
    default_llm_config: Optional[ModelConfigRequest] = Field(None, description="Default model configuration")


class TaskResponse(BaseModel):
    """Response model for task execution."""
    success: bool
    job_id: str
    execution_time: float
    response: Optional[str] = None
    error: Optional[str] = None
    agent_name: Optional[str] = None
    provider: Optional[str] = None
    orchestrator: str


class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    success: bool
    job_id: str
    execution_time: float
    workflow_steps: int
    completed_steps: int
    results: List[Dict[str, Any]]
    orchestrator: str
    error: Optional[str] = None


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
            "/run-workflow - Run a multi-agent workflow"
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


@app.post("/run-workflow", response_model=WorkflowResponse)
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
        
        return WorkflowResponse(**result)
        
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
