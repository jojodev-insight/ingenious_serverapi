"""Test logger module."""

import tempfile
import threading
from pathlib import Path

from core.logger import ThreadAwareLogger


class TestThreadAwareLogger:
    """Test thread-aware logger functionality."""

    def test_logger_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ThreadAwareLogger("test_logger", tmp_dir)

            assert logger.name == "test_logger"
            assert logger.log_dir == Path(tmp_dir)
            assert len(logger.logger.handlers) >= 2  # File and console handlers

            # Clean up handlers to avoid file lock issues
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)

    def test_log_file_creation(self):
        """Test log file is created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ThreadAwareLogger("test_logger", tmp_dir)
            logger.info("Test message")

            # Force flush all handlers
            for handler in logger.logger.handlers:
                handler.flush()

            log_file = Path(tmp_dir) / "test_logger.log"
            assert log_file.exists()

            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content
                assert "Thread-" in content

            # Clean up handlers to avoid file lock issues
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)

    def test_thread_info_included(self):
        """Test thread information is included in logs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ThreadAwareLogger("test_logger", tmp_dir)

            def log_from_thread():
                logger.info("Message from thread")

            thread = threading.Thread(target=log_from_thread, name="TestThread")
            thread.start()
            thread.join()

            # Force flush all handlers
            for handler in logger.logger.handlers:
                handler.flush()

            log_file = Path(tmp_dir) / "test_logger.log"
            with open(log_file) as f:
                content = f.read()
                assert "Message from thread" in content
                assert "thread_name" in content

            # Clean up handlers to avoid file lock issues
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)

    def test_agent_lifecycle_logging(self):
        """Test agent lifecycle logging methods."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ThreadAwareLogger("test_logger", tmp_dir)

            logger.log_agent_start("TestAgent", "test task")
            logger.log_agent_end("TestAgent", "test task", True)

            # Force flush all handlers
            for handler in logger.logger.handlers:
                handler.flush()

            log_file = Path(tmp_dir) / "test_logger.log"
            with open(log_file) as f:
                content = f.read()
                assert "Agent 'TestAgent' starting task" in content
                assert "Agent 'TestAgent' finished task" in content
                assert "SUCCESS" in content

            # Clean up handlers to avoid file lock issues
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)

    def test_orchestrator_lifecycle_logging(self):
        """Test orchestrator lifecycle logging methods."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ThreadAwareLogger("test_logger", tmp_dir)

            logger.log_orchestrator_start("job123", "TestAgent")
            logger.log_orchestrator_end("job123", True, 1.5)

            # Force flush all handlers
            for handler in logger.logger.handlers:
                handler.flush()

            log_file = Path(tmp_dir) / "test_logger.log"
            with open(log_file) as f:
                content = f.read()
                assert "Orchestrator starting job job123" in content
                assert "Orchestrator job job123 completed" in content
                assert "SUCCESS" in content
                assert "Duration: 1.50s" in content

            # Clean up handlers to avoid file lock issues
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
