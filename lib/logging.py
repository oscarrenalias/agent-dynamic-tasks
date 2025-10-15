"""
Enhanced logging system with colors using Python's standard logging framework.
"""
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform colored terminal text
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and enhanced formatting."""
    
    # Color mappings for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN + Style.DIM,
        'INFO': Fore.BLUE,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'SUCCESS': Fore.GREEN,
        'STEP_START': Fore.CYAN + Style.BRIGHT,
        'STEP_COMPLETE': Fore.GREEN + Style.BRIGHT,
        'STEP_ERROR': Fore.RED + Style.BRIGHT,
        'INPUT': Fore.MAGENTA,
        'OUTPUT': Fore.GREEN,
        'COORDINATOR': Fore.BLUE + Style.BRIGHT,
        'EXECUTOR': Fore.YELLOW + Style.BRIGHT,
        'SUBAGENT': Fore.CYAN,
        'CONSOLIDATION': Fore.MAGENTA + Style.BRIGHT,
        'TOOL_USAGE': Fore.CYAN + Style.DIM,
    }
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def format(self, record):
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Get color for the log level or category
        category = getattr(record, 'category', record.levelname)
        color = self.COLORS.get(category, self.COLORS.get(record.levelname, ''))
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Build the log message
        parts = [
            f"{Fore.CYAN + Style.DIM}[{timestamp}]",
            f"{Fore.YELLOW + Style.DIM}[+{elapsed:.2f}s]",
            f"{color}[{category}]",
            f"{Style.RESET_ALL}{record.getMessage()}"
        ]
        
        return " ".join(parts)

class AgentLogger:
    """Wrapper around Python's logging with agent-specific methods."""
    
    def __init__(self, name: str = "AgentCoordinator", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create console handler with custom formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _log_with_category(self, level: int, message: str, category: str):
        """Log a message with a custom category."""
        record = self.logger.makeRecord(
            self.logger.name, level, __file__, 0, message, (), None
        )
        record.category = category
        self.logger.handle(record)
    
    def info(self, message: str, category: str = "INFO"):
        """Log info message."""
        self._log_with_category(logging.INFO, message, category)
    
    def success(self, message: str, category: str = "SUCCESS"):
        """Log success message."""
        self._log_with_category(logging.INFO, message, category)
    
    def warning(self, message: str, category: str = "WARNING"):
        """Log warning message."""
        self._log_with_category(logging.WARNING, message, category)
    
    def error(self, message: str, category: str = "ERROR"):
        """Log error message."""
        self._log_with_category(logging.ERROR, message, category)
    
    def debug(self, message: str, category: str = "DEBUG"):
        """Log debug message."""
        self._log_with_category(logging.DEBUG, message, category)
    
    def step_start(self, step_num: int, total_steps: int, description: str):
        """Log the start of a step execution."""
        message = f"🚀 Starting Step {step_num}/{total_steps}: {description}"
        self._log_with_category(logging.INFO, message, "STEP_START")
    
    def step_input(self, step_num: int, input_data: str):
        """Log step input data."""
        # Truncate long inputs for readability
        display_input = input_data[:200] + "..." if len(input_data) > 200 else input_data
        message = f"[Step {step_num}] Input: {display_input}"
        self._log_with_category(logging.INFO, message, "INPUT")
    
    def step_output(self, step_num: int, output_data: str):
        """Log step output data."""
        # Truncate long outputs for readability
        display_output = output_data[:200] + "..." if len(output_data) > 200 else output_data
        message = f"[Step {step_num}] Output: {display_output}"
        self._log_with_category(logging.INFO, message, "OUTPUT")
    
    def step_complete(self, step_num: int, total_steps: int, duration: float):
        """Log step completion."""
        message = f"✅ Step {step_num}/{total_steps} completed in {duration:.2f}s"
        self._log_with_category(logging.INFO, message, "STEP_COMPLETE")
    
    def tool_usage(self, step_num: int, tool_name: str, query: str):
        """Log tool usage during step execution."""
        message = f"🔍 [Step {step_num}] Using {tool_name}: {query[:100]}{'...' if len(query) > 100 else ''}"
        self._log_with_category(logging.INFO, message, "TOOL_USAGE")
    
    def step_error(self, step_num: int, total_steps: int, error: str, duration: float):
        """Log step error."""
        message = f"❌ Step {step_num}/{total_steps} failed after {duration:.2f}s: {error}"
        self._log_with_category(logging.ERROR, message, "STEP_ERROR")
    
    def breakdown_complete(self, steps: list):
        """Log task breakdown completion."""
        self.info("📋 Task Breakdown:", "COORDINATOR")
        for i, step in enumerate(steps, 1):
            self.info(f"  {i}. {step}", "COORDINATOR")
    
    def final_result(self, result: str):
        """Log final consolidated result."""
        self.print_separator()
        self.success("🎯 FINAL RESULT:", "SUCCESS")
        # Print result in chunks to avoid overwhelming the log
        lines = result.split('\n')
        for line in lines:
            if line.strip():
                self.info(f"  {line}", "SUCCESS")
        self.print_separator()
    
    def execution_summary(self, total_time: float, successful_steps: int, total_steps: int, errors: int):
        """Log execution summary."""
        success_rate = (successful_steps/total_steps*100) if total_steps > 0 else 0
        
        self.info("📊 EXECUTION SUMMARY:", "SUCCESS" if errors == 0 else "WARNING")
        self.info(f"  • Total Time: {total_time:.2f}s", "SUCCESS" if errors == 0 else "WARNING")
        self.info(f"  • Successful Steps: {successful_steps}/{total_steps}", "SUCCESS" if errors == 0 else "WARNING")
        self.info(f"  • Errors: {errors}", "SUCCESS" if errors == 0 else "WARNING")
        self.info(f"  • Success Rate: {success_rate:.1f}%", "SUCCESS" if errors == 0 else "WARNING")
    
    def print_separator(self, title: str = ""):
        """Print a visual separator."""
        separator = "=" * 80
        if title:
            self.info(f"{separator}", "INFO")
            self.info(f"  {title.upper()}", "INFO")
            self.info(f"{separator}", "INFO")
        else:
            self.info(separator, "INFO")

# Global logger instance
logger = AgentLogger()

class LoggingMixin:
    """Mixin to add logging capabilities to agent classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
    
    def log_step_start(self, step_num: int, total_steps: int, description: str):
        """Log step start with timing."""
        self.step_start_time = time.time()
        self.logger.step_start(step_num, total_steps, description)
    
    def log_step_complete(self, step_num: int, total_steps: int):
        """Log step completion with duration."""
        duration = time.time() - getattr(self, 'step_start_time', 0)
        self.logger.step_complete(step_num, total_steps, duration)
    
    def log_step_error(self, step_num: int, total_steps: int, error: str):
        """Log step error with duration."""
        duration = time.time() - getattr(self, 'step_start_time', 0)
        self.logger.step_error(step_num, total_steps, error, duration)