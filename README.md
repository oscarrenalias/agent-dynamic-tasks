# agent-dynamic-tasks

A Python project for dynamic multi-step coordination of LLM tasks using LangChain and LangGraph.

## Features

- Breaks down complex prompts into actionable subtasks
- Executes each subtask with an LLM (OpenAI)
- Aggregates results and errors into a final report
- Supports both standalone and LangGraph workflow modes
- Loads the OpenAI API key from a `.env` file

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
   ```
   python agent_coordinator.py
   python langgraph_coordinator.py
   ```

## Files

- `agent_coordinator.py` — Standalone agent for dynamic task execution
- `langgraph_coordinator.py` — LangGraph workflow version
- `.env.example` — Example environment file for OpenAI API key
- `pyproject.toml` — Project dependencies

## License

MIT
