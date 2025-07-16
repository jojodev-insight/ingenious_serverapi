"""Test configuration for pytest."""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "data": "Sample data for analysis",
        "context": "Testing context"
    }


@pytest.fixture
def sample_workflow():
    """Sample workflow for testing."""
    return [
        {
            "agent": "data_analyst",
            "task_data": {"data": "test data", "context": "analysis context"}
        },
        {
            "agent": "content_writer", 
            "task_data": {"topic": "test results", "audience": "technical"}
        }
    ]
