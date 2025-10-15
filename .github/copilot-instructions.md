# Copilot Instructions for agent-dynamic-tasks

## Project Overview

This is a Python project that implements an agent coordination system with dynamic task breakdown and execution using LangChain and LangGraph. The system breaks down complex prompts into actionable subtasks, executes them with LLMs, includes web search capabilities, and aggregates results.

## Development Environment & Package Management

### Package Manager: uv (NOT pip)
This project uses **uv** as the package manager, not pip. Always use uv commands:

#### ❌ Avoid These Commands:
- `pip install <package>` - Use `uv add` instead
- `python <script>` - Use `uv run python <script>` instead
- `pip freeze` - Dependencies are managed in `pyproject.toml`

### Python Version
- Requires Python 3.13+
- Environment should be managed through uv

## Project Structure

```
├── main.py                 # Main LangGraph coordinator with CLI interface
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock               # Lock file for exact dependency versions
├── README.md             # Project documentation
├── examples/
│   └── prompt.txt        # Example input file
└── lib/
    ├── __init__.py
    ├── logging.py        # Enhanced logging utilities
    └── README.md
```

## Key Dependencies

- **langchain**: Core LangChain framework
- **langchain-openai**: OpenAI integration
- **langchain-tavily**: Tavily search integration for web search
- **langgraph**: Graph-based workflow coordination
- **openai**: OpenAI API client
- **python-dotenv**: Environment variable management
- **colorama & rich**: Enhanced terminal output

## Running the Application

### Main Entry Points:
1. **LangGraph Coordinator (primary)**: `uv run python main.py`
2. **Command-line interface**: Supports input/output files

### Example Usage:
```bash
# Run with input file
echo "Analyze the latest AI trends and provide a summary" > prompt.txt
uv run python main.py --input prompt.txt --output results.txt

# Run interactively
uv run python main.py
```

### Workflow Pattern:
1. Prompt analysis and task breakdown
2. Sequential execution of subtasks
3. Optional web search when agents determine it's needed
4. Result aggregation and final report generation
5. Error handling and logging throughout

## Testing & Development

### Running Tests:
```bash
uv run python -m pytest  # If tests exist
```

### Development Workflow:
1. Add dependencies: `uv add <package>`
2. Run code: `uv run python main.py`
3. Check logs for execution details
4. Use input/output files for testing different scenarios

## Common Pitfalls to Avoid

1. **Don't use pip** - Always use uv for package management
2. **Don't run python directly** - Use `uv run python` to ensure proper environment
3. **API keys required** - The application won't work without proper .env configuration
4. **Web search is automatic** - Agents decide when to use Tavily search based on task needs
5. **File-based I/O** - The CLI interface supports file input/output for automation

## Logging & Debugging

- Enhanced logging is available through `lib.logging`
- Real-time step tracking shows progress
- Execution summaries provide timing information
- Errors are captured and reported in final output

When debugging or adding features, always use the enhanced logging system and remember that this is a multi-agent system where coordination and state management are key.