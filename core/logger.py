"""Logging utilities for the Autogen project."""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from core.config import config


class ThreadAwareLogger:
    """Thread-aware logger for tracking agent and orchestrator execution."""
    
    def __init__(self, name: str, log_dir: Optional[str] = None) -> None:
        """Initialize the logger.
        
        Args:
            name: Logger name.
            log_dir: Directory for log files. If None, uses config default.
        """
        self.name = name
        self.log_dir = Path(log_dir or config.log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Set up file and console handlers."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [Thread-%(thread)d] - %(message)s'
        )
        
        # File handler
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _get_thread_info(self) -> Dict[str, Any]:
        """Get current thread information."""
        thread = threading.current_thread()
        return {
            "thread_id": thread.ident,
            "thread_name": thread.name,
            "timestamp": datetime.now().isoformat()
        }
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log info message with thread context."""
        thread_info = self._get_thread_info()
        if extra_data:
            message = f"{message} | Extra: {extra_data}"
        message = f"{message} | Thread Info: {thread_info}"
        self.logger.info(message)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with thread context."""
        thread_info = self._get_thread_info()
        if extra_data:
            message = f"{message} | Extra: {extra_data}"
        message = f"{message} | Thread Info: {thread_info}"
        self.logger.error(message)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message with thread context."""
        thread_info = self._get_thread_info()
        if extra_data:
            message = f"{message} | Extra: {extra_data}"
        message = f"{message} | Thread Info: {thread_info}"
        self.logger.warning(message)
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message with thread context."""
        thread_info = self._get_thread_info()
        if extra_data:
            message = f"{message} | Extra: {extra_data}"
        message = f"{message} | Thread Info: {thread_info}"
        self.logger.debug(message)
    
    def log_agent_start(self, agent_name: str, task: str) -> None:
        """Log agent execution start."""
        self.info(f"Agent '{agent_name}' starting task: {task}")
    
    def log_agent_end(self, agent_name: str, task: str, success: bool) -> None:
        """Log agent execution end."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Agent '{agent_name}' finished task: {task} | Status: {status}")
    
    def log_orchestrator_start(self, job_id: str, agent_name: str) -> None:
        """Log orchestrator job start."""
        self.info(f"Orchestrator starting job {job_id} with agent '{agent_name}'")
    
    def log_orchestrator_end(self, job_id: str, success: bool, duration: float) -> None:
        """Log orchestrator job end."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Orchestrator job {job_id} completed | Status: {status} | Duration: {duration:.2f}s")


# Global logger instances
orchestrator_logger = ThreadAwareLogger("orchestrator")
agent_logger = ThreadAwareLogger("agent")
