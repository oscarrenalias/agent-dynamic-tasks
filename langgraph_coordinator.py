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
            "You are a helpful assistant. Break down the following user request into a numbered list of actionable steps. "
            "Only output the list.\nUser request: {task}\nList of steps:"
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
    prompt = PromptTemplate(
        input_variables=["instruction"],
        template="Follow this instruction and provide a concise answer:\nInstruction: {instruction}"
    )
    try:
        result = llm.invoke(prompt.format(instruction=step)).content
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
            "The user request was: {user_request}\n\n"
            "The following steps were executed, with their results:\n{step_results}\n\n"
            "Errors (if any):\n{step_errors}\n\n"
            "Based on the above, produce a final, consolidated answer, report, or output that best fulfills the user's original request."
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