import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class CoordinatorState(TypedDict):
    prompt: str
    steps: List[str]
    results: Dict[int, str]
    errors: Dict[int, str]
    final_output: str

def break_down_task(state: CoordinatorState) -> CoordinatorState:
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
    steps_raw = llm.invoke(breakdown_prompt.format(task=state["prompt"])).content
    steps = [line.split(". ", 1)[-1].strip()
             for line in steps_raw.split("\n") if line.strip() and line[0].isdigit()]
    state["steps"] = steps
    return state

def execute_step(state: CoordinatorState, step_idx: int) -> CoordinatorState:
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    step = state["steps"][step_idx]
    
    # Build context from previous step results
    previous_results = ""
    if step_idx > 0:
        prev_results = []
        for i in range(step_idx):
            if i in state["results"]:
                prev_results.append(f"Step {i+1} ({state['steps'][i]}): {state['results'][i]}")
        if prev_results:
            previous_results = f"\n\nPrevious step results that may be relevant:\n" + "\n".join(prev_results)
    
    prompt = PromptTemplate(
        input_variables=["original_task", "current_step", "step_number", "total_steps", "instruction", "previous_results"],
        template=(
            "ORIGINAL USER REQUEST: {original_task}\n\n"
            "YOUR ROLE: You are executing step {step_number} of {total_steps} in a multi-step process to fulfill the above request.\n"
            "CURRENT STEP: {current_step}\n\n"
            "CONTEXT: This step contributes to achieving the original user's goal. Stay focused on the topic and domain of the original request. "
            "Provide information that is directly relevant and useful for the final deliverable.\n\n"
            "SPECIFIC INSTRUCTION: {instruction}\n"
            "{previous_results}\n\n"
            "Execute this instruction while keeping the original request and your role in mind. Provide a focused, relevant response:"
        )
    )
    try:
        result = llm.invoke(prompt.format(
            original_task=state["prompt"],
            current_step=step,
            step_number=step_idx + 1,
            total_steps=len(state["steps"]),
            instruction=step,
            previous_results=previous_results
        )).content
        state["results"][step_idx] = result
    except Exception as e:
        state["errors"][step_idx] = str(e)
    return state

def consolidate_results(state: CoordinatorState) -> CoordinatorState:
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
    state["final_output"] = llm.invoke(
        synth_prompt.format(
            user_request=state["prompt"],
            step_results=results_str if results_str else "No results.",
            step_errors=errors_str if errors_str else "No errors."
        )
    ).content
    return state

def execute_all_steps(state: CoordinatorState) -> CoordinatorState:
    for i in range(len(state["steps"])):
        state = execute_step(state, i)
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
    graph = build_coordinator_graph()
    initial_state: CoordinatorState = {
        "prompt": prompt,
        "steps": [],
        "results": {},
        "errors": {},
        "final_output": ""
    }
    result = graph.invoke(initial_state)
    return result

if __name__ == "__main__":
    prompt = "Write a report about the increase of measles cases in the US. Identify sources, pull the necessary data, and generate a report in markdown with the key outcomes and your analysis."
    state = run_coordinator(prompt)
    print("Final Output:\n", state["final_output"])
    print("\nSubtasks:")
    for i, step in enumerate(state["steps"]):
        print(f"Step {i+1}: {step}")
        print("Result:", state["results"].get(i, ""))
        if i in state["errors"]:
            print("Error:", state["errors"][i])
        print("---")