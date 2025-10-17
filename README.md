# agent-dynamic-tasks

A tool based on a coordinator agent that with dynamic multi-step tasks using LangChain and LangGraph. When provided with a task in a prompt, the coordinator agent will determine what kind of steps are required to achieve it, and will coordinate the execution of each step with task-specific agents that will carry out the tasks. The coordinator is also responsible for managing inputs and outputs via the context, and passing the context between task agents.

## Features

- Breaks down complex prompts into actionable subtasks, based on the agent's understanding of the task and its own analysis of the steps required to achieve the requested goal
- Executes each subtask with an LLM (OpenAI)
- **Web search capabilities** via Tavily Search API for real-time information access
- Agents automatically decide when to use web search based on task requirements
- Aggregates results and errors into a final report
- Supports both standalone and LangGraph workflow modes
- **LangGraph coordinator supports command-line interface with input/output files**
- Loads API keys from a `.env` file
- Enhanced logging with real-time step tracking and execution summaries

## Getting Started

1. **Install dependencies**  
   Use [uv](https://github.com/astral-sh/uv)
   ```
   uv sync
   ```

2. **Configure API Keys**  
   Copy `.env.example` to `.env` and fill in your API keys:
   ```
   cp .env.example .env
   ```
   
   Required API keys:
   - **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
   - **Tavily API Key**: Get from [Tavily](https://app.tavily.com/sign-in) (1,000 free searches/month)

3. **Run the agents**  
   
   ```bash
   # Create an input file with your task instructions
   echo "Your task instructions here" > prompt.txt
   
   # Run with output to stdout
   uv run python main.py -i prompt.txt
   
   # Run with output to file
   uv run python main.py -i prompt.txt -o results.md
   ```

## Files

- `main.py` — LangGraph workflow version with command-line interface
- `lib/` — Python library modules
  - `lib/logging.py` — Enhanced logging system with step tracking and colored output
- `examples/` — Example task files for testing
- `.env.example` — Example environment file for API keys
- `pyproject.toml` — Project dependencies

## Web Search Integration

The agents have **automatic web search capabilities** powered by Tavily Search API:

- **Intelligent Usage**: Agents automatically decide when to search for current information
- **Research Tasks**: Perfect for tasks requiring recent data, statistics, news, or fact-checking
- **Seamless Integration**: Search results are used internally by each step's agent
- **No Manual Control**: The LLM determines when web search is beneficial

### Example Search-Enhanced Tasks:
```bash
echo "Analyze the latest developments in renewable energy policy in 2024" > research_task.txt
uv run python main.py -i research_task.txt -o energy_report.md
```

```bash
echo "Write a report on recent AI breakthrough announcements from major tech companies" > ai_research.txt
uv run python main.py -i ai_research.txt
```

## Command-Line Options (LangGraph Coordinator)

The `main.py` script supports the following command-line options:

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
uv run python main.py -i energy_analysis.txt -o energy_report.md
```

**View results in terminal:**
```bash
uv run python main.py -i energy_analysis.txt
```

**Test web search capabilities:**
```bash
uv run python main.py -i search_test_task.txt -o ai_analysis_report.md
```

## License

MIT
