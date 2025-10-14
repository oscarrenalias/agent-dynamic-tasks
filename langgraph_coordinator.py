import os
from typing import Dict, Any, List
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, State, END

# ----- 1. Define the State -----
class CoordinatorState(State):
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.steps: List[str] = []
        self.results: Dict[int, str] = {}
        self.errors: Dict[int, str] = {}
        self.final_output: str = ""

# ----- 2. Node: Break down the task -----
def break_down_task(state: CoordinatorState) -> CoordinatorState:
    llm = OpenAI(model="gpt-3.5-turbo")
    breakdown_prompt = PromptTemplate(
        input_variables=["task"],
        template=(
            "You are a helpful assistant. Break down the following user request into a numbered list of actionable steps. "
            "Only output the list.\nUser request: {task}\nList of steps:"
        )
    )
    steps_raw = llm.invoke(breakdown_prompt.format(task=state.prompt))
    steps = [line.split(". ", 1)[-1].strip()
             for line in steps_raw.split("\n") if line.strip() and line[0].isdigit()]
    state.steps = steps
    return state

# ----- 3. Node: Execute each subtask -----
def execute_step(state: CoordinatorState, step_idx: int) -> CoordinatorState:
    llm = OpenAI(model="gpt-3.5-turbo")
    step = state.steps[step_idx]
    prompt = PromptTemplate(
        input_variables=["instruction"],
        template="Follow this instruction and provide a concise answer:\nInstruction: {instruction}"
    )
    try:
        result = llm.invoke(prompt.format(instruction=step))
        state.results[step_idx] = result
    except Exception as e:
        state.errors[step_idx] = str(e)
    return state

# ----- 4. Node: Consolidate results -----
def consolidate_results(state: CoordinatorState) -> CoordinatorState:
    llm = OpenAI(model="gpt-3.5-turbo")
    results_str = "\n".join(
        [f"Step {i+1}: {state.steps[i]}\nResult: {state.results.get(i, '')}" for i in range(len(state.steps))]
    )
    errors_str = "\n".join(
        [f"Step {i+1}: {state.steps[i]}\nError: {state.errors[i]}" for i in state.errors]
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
    state.final_output = llm.invoke(
        synth_prompt.format(
            user_request=state.prompt,
            step_results=results_str if results_str else "No results.",
            step_errors=errors_str if errors_str else "No errors."
        )
    )
    return state

# ----- 5. Build the LangGraph -----
def build_coordinator_graph(max_steps: int = 10):
    graph = StateGraph(CoordinatorState)
    graph.add_node("breakdown", break_down_task)
    # Dynamically add step nodes and edges for up to max_steps
    for i in range(max_steps):
        graph.add_node(f"step_{i+1}", lambda state, i=i: execute_step(state, i))
    graph.add_node("consolidate", consolidate_results)

    # Edges: breakdown -> step_1, step_i -> step_{i+1}, last step -> consolidate
    graph.add_edge("breakdown", "step_1")
    for i in range(1, max_steps):
        graph.add_edge(f"step_{i}", f"step_{i+1}")
    graph.add_edge(f"step_{max_steps}", "consolidate")
    graph.set_entry_point("breakdown")
    graph.set_exit_point("consolidate")
    return graph

# ----- 6. Runner -----
def run_coordinator(prompt: str, max_steps: int = 10):
    graph = build_coordinator_graph(max_steps)
    # Initial state
    state = CoordinatorState(prompt)
    # First run: breakdown
    state = graph.run_node("breakdown", state)
    # Only run as many steps as are present
    steps_to_run = min(len(state.steps), max_steps)
    for i in range(steps_to_run):
        state = graph.run_node(f"step_{i+1}", state)
    state = graph.run_node("consolidate", state)
    return state

# ----- 7. Example Usage -----
if __name__ == "__main__":
    prompt = "Write a report about the increase of measles cases in the US. Identify sources, pull the necessary data, and generate a report in markdown with the key outcomes and your analysis."
    state = run_coordinator(prompt, max_steps=10)
    print("Final Output:\n", state.final_output)
    print("\nSubtasks:")
    for i, step in enumerate(state.steps):
        print(f"Step {i+1}: {step}")
        print("Result:", state.results.get(i, ""))
        if i in state.errors:
            print("Error:", state.errors[i])
        print("---")