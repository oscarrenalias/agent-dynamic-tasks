"""
Agent Dynamic Tasks Library

A collection of utilities and modules for the agent dynamic tasks project.
"""

__version__ = "0.1.0"

# Import main components for easy access
from .logging import logger, LoggingMixin, AgentLogger

__all__ = ["logger", "LoggingMixin", "AgentLogger"]