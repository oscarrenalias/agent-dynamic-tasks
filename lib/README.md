# Agent Dynamic Tasks Library

This directory contains the core library modules for the agent dynamic tasks project.

## Modules

### `logging.py`
Enhanced logging system with colors and agent-specific functionality.

**Main Components:**
- `AgentLogger`: Main logger class with colored output and timing
- `LoggingMixin`: Mixin class to add logging capabilities to other classes
- `logger`: Global logger instance ready to use

**Features:**
- Colored terminal output with different colors for different log categories
- Step tracking with timing information
- Execution summaries
- Tool usage logging
- Real-time elapsed time display

**Usage:**
```python
from lib.logging import logger, LoggingMixin

# Direct logging
logger.info("Task started", "COORDINATOR")
logger.success("Task completed", "EXECUTOR")
logger.error("Task failed", "ERROR")

# Step tracking
logger.step_start(1, 5, "Analyze user requirements")
logger.step_complete(1, 5, 2.5)  # duration in seconds

# Using as a mixin
class MyAgent(LoggingMixin):
    def __init__(self):
        super().__init__()
        self.logger.info("Agent initialized", "AGENT")
```

## Installation and Imports

The library is designed to be imported from the main project directory:

```python
# Import specific components
from lib.logging import logger, LoggingMixin, AgentLogger

# Or import from the main lib module
from lib import logger, LoggingMixin
```

## Module Structure

```
lib/
├── __init__.py          # Main module exports
└── logging.py           # Enhanced logging system
```