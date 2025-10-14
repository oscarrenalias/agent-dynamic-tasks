import os
import sys
import time
import argparse
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict
from dotenv import load_dotenv
from enhanced_logging import logger, LoggingMixin

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

class CoordinatorState(TypedDict):
    prompt: str
    steps: List[str]
    results: Dict[int, str]
    errors: Dict[int, str]
    final_output: str
    execution_start_time: float
    step_timings: Dict[int, float]

def create_search_tool():
    """Create and return a Tavily search tool for web search capabilities."""
    if not tavily_api_key or tavily_api_key == "your-tavily-api-key-here":
        logger.warning("Tavily API key not configured. Web search will be unavailable.", "COORDINATOR")
        return None
    
    try:
        search_tool = TavilySearch(
            max_results=5,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        logger.info("Web search tool (Tavily) initialized successfully", "COORDINATOR")
        return search_tool
    except Exception as e:
        logger.error(f"Failed to initialize Tavily search tool: {str(e)}", "COORDINATOR")
        return None

def break_down_task(state: CoordinatorState) -> CoordinatorState:
    logger.info("Starting task breakdown", "COORDINATOR")
    logger.info(f"Original request: {state['prompt'][:100]}{'...' if len(state['prompt']) > 100 else ''}")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    breakdown_prompt = PromptTemplate(
        input_variables=["task"],
        template=(
            "You are a task breakdown specialist. Analyze the following user request and break it down into a numbered list of clear, actionable steps.\n\n"
            "Each step should:\n"
            "- Be specific and actionable\n"
            "- Stay focused on the main topic and domain\n"
            "- Build logically toward the final deliverable\n"
            "- Be scoped appropriately for an AI agent to execute\n\n"
            "User request: {task}\n\n"
            "Numbered list of actionable steps:"
        )
    )
    
    try:
        steps_raw = llm.invoke(breakdown_prompt.format(task=state["prompt"])).content
        steps = [line.split(". ", 1)[-1].strip()
                 for line in steps_raw.split("\n") if line.strip() and line[0].isdigit()]
        state["steps"] = steps
        
        logger.breakdown_complete(steps)
        logger.success(f"Task broken down into {len(steps)} steps", "COORDINATOR")
        
    except Exception as e:
        logger.error(f"Failed to break down task: {str(e)}", "COORDINATOR")
        state["steps"] = []
    
    return state

def execute_step(state: CoordinatorState, step_idx: int) -> CoordinatorState:
    step_start_time = time.time()
    total_steps = len(state["steps"])
    step = state["steps"][step_idx]
    
    logger.step_start(step_idx + 1, total_steps, step)
    
    # Create LLM instance
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    
    # Add search tool capabilities if available
    search_tool = create_search_tool()
    tools = []
    if search_tool:
        tools = [search_tool]
        # Bind the search tool to the LLM
        llm = llm.bind_tools(tools)
        logger.info(f"[Step {step_idx + 1}] Web search tool available to agent", "EXECUTOR")
    else:
        logger.info(f"[Step {step_idx + 1}] No web search tool available", "EXECUTOR")
    
    # Build context from previous step results
    previous_results = ""
    if step_idx > 0:
        prev_results = []
        for i in range(step_idx):
            if i in state["results"]:
                prev_results.append(f"Step {i+1} ({state['steps'][i]}): {state['results'][i]}")
        if prev_results:
            previous_results = f"\n\nPrevious step results that may be relevant:\n" + "\n".join(prev_results)
    
    try:
        # Add tool information to the prompt
        tool_info = ""
        if search_tool:
            tool_info = (
                "AVAILABLE TOOLS: You have access to web search capabilities. Use the search tool when you need current information, "
                "recent data, facts, statistics, or any information that requires up-to-date knowledge. The search tool can help you "
                "find reliable, current information to complete your task effectively.\n\n"
            )
        
        # Create the prompt content
        prompt_content = (
            f"ORIGINAL USER REQUEST: {state['prompt']}\n\n"
            f"YOUR ROLE: You are executing step {step_idx + 1} of {total_steps} in a multi-step process to fulfill the above request.\n"
            f"CURRENT STEP: {step}\n\n"
            f"CONTEXT: This step contributes to achieving the original user's goal. Stay focused on the topic and domain of the original request. "
            f"Provide information that is directly relevant and useful for the final deliverable.\n\n"
            f"{tool_info}"
            f"SPECIFIC INSTRUCTION: {step}\n"
            f"{previous_results}\n\n"
            f"Execute this instruction while keeping the original request and your role in mind. Provide a focused, relevant response:"
        )
        
        logger.step_input(step_idx + 1, prompt_content)
        
        # Create messages for the conversation
        messages = [HumanMessage(content=prompt_content)]
        
        # Get initial response from LLM
        response = llm.invoke(messages)
        messages.append(response)
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls and tools:
            logger.info(f"[Step {step_idx + 1}] LLM is calling {len(response.tool_calls)} tool(s)", "EXECUTOR")
            
            # Create tool node to execute tools
            tool_node = ToolNode(tools)
            
            # Execute tools
            tool_response = tool_node.invoke({"messages": messages})
            
            # Add tool messages to conversation
            messages.extend(tool_response["messages"])
            
            # Get final response from LLM after tool execution
            final_response = llm.invoke(messages)
            result = final_response.content
        else:
            # No tool calls, use the initial response
            result = response.content
        
        state["results"][step_idx] = result
        
        logger.step_output(step_idx + 1, result)
        
        duration = time.time() - step_start_time
        state["step_timings"][step_idx] = duration
        logger.step_complete(step_idx + 1, total_steps, duration)
        
    except Exception as e:
        error_msg = str(e)
        state["errors"][step_idx] = error_msg
        
        duration = time.time() - step_start_time
        state["step_timings"][step_idx] = duration
        logger.step_error(step_idx + 1, total_steps, error_msg, duration)
    
    return state

def consolidate_results(state: CoordinatorState) -> CoordinatorState:
    logger.info("Starting result consolidation", "COORDINATOR")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    results_str = "\n".join(
        [f"Step {i+1}: {state['steps'][i]}\nResult: {state['results'].get(i, '')}" for i in range(len(state['steps']))]
    )
    errors_str = "\n".join(
        [f"Step {i+1}: {state['steps'][i]}\nError: {state['errors'][i]}" for i in state['errors']]
    )
    synth_prompt = PromptTemplate(
        input_variables=["user_request", "step_results", "step_errors"],
        template=(
            "ORIGINAL USER REQUEST: {user_request}\n\n"
            "COMPLETED STEPS AND RESULTS:\n{step_results}\n\n"
            "ERRORS (if any):\n{step_errors}\n\n"
            "YOUR TASK: Synthesize the above step results into a comprehensive final output that directly addresses the user's original request. "
            "Ensure the final output:\n"
            "- Directly fulfills what the user asked for\n"
            "- Integrates all relevant information from the step results\n"
            "- Maintains focus on the original topic and requirements\n"
            "- Presents information in the format requested (if specified)\n"
            "- Provides actionable insights or conclusions where appropriate\n\n"
            "FINAL OUTPUT:"
        )
    )
    
    try:
        consolidation_input = synth_prompt.format(
            user_request=state["prompt"],
            step_results=results_str if results_str else "No results.",
            step_errors=errors_str if errors_str else "No errors."
        )
        
        logger.info("Sending consolidation request to LLM", "CONSOLIDATION")
        
        final_output = llm.invoke(consolidation_input).content
        state["final_output"] = final_output
        
        logger.success("Result consolidation completed", "COORDINATOR")
        logger.final_result(final_output)
        
        # Log execution summary
        total_time = time.time() - state.get("execution_start_time", 0)
        successful_steps = len(state["results"])
        total_steps = len(state["steps"])
        errors = len(state["errors"])
        
        logger.execution_summary(total_time, successful_steps, total_steps, errors)
        
    except Exception as e:
        logger.error(f"Failed to consolidate results: {str(e)}", "COORDINATOR")
        state["final_output"] = "Error occurred during result consolidation."
    
    return state

def execute_all_steps(state: CoordinatorState) -> CoordinatorState:
    logger.print_separator("Step Execution Phase")
    logger.info(f"Executing {len(state['steps'])} steps", "EXECUTOR")
    
    for i in range(len(state["steps"])):
        state = execute_step(state, i)
    
    logger.print_separator("Step Execution Complete")
    return state

def build_coordinator_graph():
    graph = StateGraph(CoordinatorState)
    graph.add_node("breakdown", break_down_task)
    graph.add_node("execute", execute_all_steps)
    graph.add_node("consolidate", consolidate_results)

    graph.add_edge(START, "breakdown")
    graph.add_edge("breakdown", "execute")
    graph.add_edge("execute", "consolidate")
    graph.add_edge("consolidate", END)
    
    return graph.compile()

def run_coordinator(prompt: str):
    logger.print_separator("Agent Coordination Starting")
    logger.info(f"Initializing coordinator for task", "MAIN")
    
    graph = build_coordinator_graph()
    initial_state: CoordinatorState = {
        "prompt": prompt,
        "steps": [],
        "results": {},
        "errors": {},
        "final_output": "",
        "execution_start_time": time.time(),
        "step_timings": {}
    }
    
    logger.info("Starting LangGraph execution", "MAIN")
    result = graph.invoke(initial_state)
    
    logger.print_separator("Agent Coordination Complete")
    return result

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LangGraph Task Coordinator - Break down and execute complex tasks using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i prompt.txt
  %(prog)s -i instructions.txt -o results.md
  %(prog)s --input task.txt --output report.txt
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        type=str,
        help='Input file containing the task instructions (required)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file to write results to (optional, defaults to stdout)'
    )
    
    return parser.parse_args()

def read_input_file(file_path: str) -> str:
    """Read and return the contents of the input file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if not content:
                logger.error(f"Input file '{file_path}' is empty", "MAIN")
                sys.exit(1)
            return content
    except FileNotFoundError:
        logger.error(f"Input file '{file_path}' not found", "MAIN")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading input file '{file_path}': {str(e)}", "MAIN")
        sys.exit(1)

def write_output(content: str, output_file: str = None):
    """Write content to output file or stdout."""
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(content)
            logger.success(f"Results written to '{output_file}'", "MAIN")
        except Exception as e:
            logger.error(f"Error writing to output file '{output_file}': {str(e)}", "MAIN")
            sys.exit(1)
    else:
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(content)
        print("="*80)

if __name__ == "__main__":
    args = parse_arguments()
    
    # Read input from file
    logger.info(f"Reading instructions from '{args.input}'", "MAIN")
    prompt = read_input_file(args.input)
    logger.info(f"Loaded task: {prompt[:100]}{'...' if len(prompt) > 100 else ''}", "MAIN")
    
    # Execute the coordination
    state = run_coordinator(prompt)
    
    # Write output
    final_output = state.get("final_output", "No output generated")
    write_output(final_output, args.output)
    
    # The final output and detailed results are already logged by the enhanced logger
    # No need for additional print statements as everything is logged in real-time