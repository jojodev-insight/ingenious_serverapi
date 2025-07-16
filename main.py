"""Main entry point for the Autogen project."""

import uvicorn
from api.app import app
from core import config, orchestrator_logger


def main():
    """Run the FastAPI application."""
    orchestrator_logger.info(f"Starting Autogen Orchestrator API on {config.api_host}:{config.api_port}")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    main()
