"""Test orchestrator module."""

import pytest
from unittest.mock import Mock, patch
from api.orchestrator import TaskOrchestrator


class TestTaskOrchestrator:
    """Test task orchestrator functionality."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = TaskOrchestrator()
        agents = orchestrator.list_agents()
        
        assert "data_analyst" in agents
        assert "content_writer" in agents
        assert "code_reviewer" in agents
    
    def test_register_agent(self):
        """Test agent registration."""
        orchestrator = TaskOrchestrator()
        
        class TestAgent:
            pass
        
        orchestrator.register_agent("test_agent", TestAgent)
        agents = orchestrator.list_agents()
        
        assert "test_agent" in agents
        assert agents["test_agent"] == "TestAgent"
    
    @patch('api.orchestrator.AgentFactory')
    def test_run_agent_success(self, mock_factory):
        """Test successful agent execution."""
        # Mock agent instance and execution
        mock_agent = Mock()
        mock_agent.execute.return_value = {
            "success": True,
            "response": "Test response",
            "agent_name": "DataAnalyst",
            "provider": "openai"
        }
        mock_agent.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4"
        }
        mock_factory.create_agent.return_value = mock_agent
        
        orchestrator = TaskOrchestrator()
        result = orchestrator.run_agent(
            "data_analyst",
            {"data": "test data"},
            "openai",
            enable_fallback=False
        )
        
        assert result["success"] is True
        assert "job_id" in result
        assert "execution_time" in result
        assert result["response"] == "Test response"
        
        # Verify agent was created with correct parameters
        mock_factory.create_agent.assert_called_once()
        call_args = mock_factory.create_agent.call_args
        assert call_args[1]['agent_type'] == "data_analyst"
        assert call_args[1]['provider'] == "openai"
        mock_agent.execute.assert_called_once_with({"data": "test data"})
    
    def test_run_agent_invalid_name(self):
        """Test running agent with invalid name."""
        orchestrator = TaskOrchestrator()
        result = orchestrator.run_agent(
            "invalid_agent",
            {"data": "test data"},
            enable_fallback=False
        )
        
        assert result["success"] is False
        assert "Unknown agent" in result["error"]
    
    @patch('api.orchestrator.AgentFactory')
    def test_multi_agent_workflow_success(self, mock_factory):
        """Test successful multi-agent workflow."""
        # Mock agents
        mock_analyst = Mock()
        mock_analyst.execute.return_value = {
            "success": True,
            "response": "Analysis complete",
            "agent_name": "DataAnalyst"
        }
        mock_analyst.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4"
        }
        
        mock_writer = Mock()
        mock_writer.execute.return_value = {
            "success": True,
            "response": "Content written",
            "agent_name": "ContentWriter"
        }
        mock_writer.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4"
        }
        
        # Configure factory to return appropriate agents
        def side_effect(**kwargs):
            if kwargs.get('agent_type') == 'data_analyst':
                return mock_analyst
            elif kwargs.get('agent_type') == 'content_writer':
                return mock_writer
            
        mock_factory.create_agent.side_effect = side_effect
        
        orchestrator = TaskOrchestrator()
        workflow = [
            {"agent": "data_analyst", "task_data": {"data": "test data"}},
            {"agent": "content_writer", "task_data": {"topic": "test topic"}}
        ]
        
        result = orchestrator.run_multi_agent_workflow(workflow, enable_fallback=False)
        
        assert result["success"] is True
        assert result["workflow_steps"] == 2
        assert result["completed_steps"] == 2
        assert len(result["results"]) == 2
    
    @patch('api.orchestrator.AgentFactory')
    def test_multi_agent_workflow_failure(self, mock_factory):
        """Test multi-agent workflow with failure."""
        # Mock agent that fails
        mock_agent = Mock()
        mock_agent.execute.return_value = {
            "success": False,
            "error": "Agent failed",
            "agent_name": "DataAnalyst"
        }
        mock_agent.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4"
        }
        mock_factory.create_agent.return_value = mock_agent
        
        orchestrator = TaskOrchestrator()
        workflow = [
            {"agent": "data_analyst", "task_data": {"data": "test data"}},
            {"agent": "data_analyst", "task_data": {"data": "more data"}}
        ]
        
        result = orchestrator.run_multi_agent_workflow(workflow, enable_fallback=False)
        
        assert result["success"] is False
        assert result["workflow_steps"] == 2
        assert result["completed_steps"] == 1  # Should stop after first failure
