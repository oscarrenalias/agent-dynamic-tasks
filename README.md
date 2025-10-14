# agent-dynamic-tasks

A Python project for dynamic multi-step coordination of LLM tasks using LangChain and LangGraph.

## Features

- Breaks down complex prompts into actionable subtasks
- Executes each subtask with an LLM (OpenAI)
- Aggregates results and errors into a final report
- Supports both standalone and LangGraph workflow modes
- **LangGraph coordinator supports command-line interface with input/output files**
- Loads the OpenAI API key from a `.env` file
- Enhanced logging with real-time step tracking and execution summaries

## Getting Started

1. **Install dependencies**  
   Recommended: Use [uv](https://github.com/astral-sh/uv) or pip  
   ```
   uv pip install -r requirements.txt
   ```
   Or use `pyproject.toml`:
   ```
   uv pip install -e .
   ```

2. **Configure API Key**  
   Copy `.env.example` to `.env` and fill in your OpenAI API key:
   ```
   cp .env.example .env
   ```

3. **Run the agents**  
   
   **Standalone Agent:**
   ```bash
   uv run python agent_coordinator.py
   ```
   
   **LangGraph Coordinator (with command-line interface):**
   ```bash
   # Create an input file with your task instructions
   echo "Your task instructions here" > prompt.txt
   
   # Run with output to stdout
   uv run python langgraph_coordinator.py -i prompt.txt
   
   # Run with output to file
   uv run python langgraph_coordinator.py -i prompt.txt -o results.md
   
   # View help and usage examples
   uv run python langgraph_coordinator.py --help
   ```

## Files

- `agent_coordinator.py` — Standalone agent for dynamic task execution
- `langgraph_coordinator.py` — LangGraph workflow version with command-line interface
- `enhanced_logging.py` — Enhanced logging system with step tracking and colored output
- `example_prompt.txt` — Example task file for testing the LangGraph coordinator
- `.env.example` — Example environment file for OpenAI API key
- `pyproject.toml` — Project dependencies

## Command-Line Options (LangGraph Coordinator)

The `langgraph_coordinator.py` script supports the following command-line options:

- `-i, --input` (required): Input file containing task instructions
- `-o, --output` (optional): Output file to write results to (defaults to stdout if not specified)
- `-h, --help`: Show help message with usage examples

## Examples

**Create a task file:**
```bash
echo "Write a comprehensive analysis of renewable energy trends in 2024" > energy_analysis.txt
```

**Run analysis and save to markdown:**
```bash
uv run python langgraph_coordinator.py -i energy_analysis.txt -o energy_report.md
```

**View results in terminal:**
```bash
uv run python langgraph_coordinator.py -i energy_analysis.txt
```

## License

MIT
