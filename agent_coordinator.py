import time
import os
from typing import List, Dict, Any, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import concurrent.futures
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Ledger abstraction ---
class AbstractLedger:
    def add_task(self, task_id: str, description: str): raise NotImplementedError
    def update_task(self, task_id: str, status: str, result: Optional[str]=None, error: Optional[str]=None): raise NotImplementedError
    def summary(self) -> Dict[str, Any]: raise NotImplementedError

class InMemoryLedger(AbstractLedger):
    def __init__(self):
        self.tasks = {}
    def add_task(self, task_id, description):
        self.tasks[task_id] = {
            "description": description,
            "status": "pending",
            "result": None,
            "error": None,
            "start_time": time.time(),
            "end_time": None
        }
    def update_task(self, task_id, status=None, result=None, error=None):
        task = self.tasks[task_id]
        if status:
            task["status"] = status
            if status == "completed" or status == "error":
                task["end_time"] = time.time()
        if result:
            task["result"] = result
        if error:
            task["error"] = error
    def summary(self):
        return self.tasks

def run_with_timeout(func: Callable, args: tuple, timeout: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("SubAgent execution timed out.")

class SubAgent:
    def __init__(self, llm, ledger: AbstractLedger, max_steps=3, timeout=30):
        self.llm = llm
        self.ledger = ledger
        self.max_steps = max_steps
        self.timeout = timeout

    def execute(self, task_id: str, description: str) -> Optional[str]:
        prompt = PromptTemplate(
            input_variables=["instruction"],
            template="Follow this instruction step by step and provide a concise answer:\nInstruction: {instruction}"
        )
        for step in range(self.max_steps):
            try:
                result = run_with_timeout(
                    self.llm.invoke,
                    (prompt.format(instruction=description),),
                    self.timeout
                ).content
                self.ledger.update_task(task_id, status="completed", result=result)
                return result
            except TimeoutError as e:
                self.ledger.update_task(task_id, status="error", error=str(e))
                return None
            except Exception as e:
                self.ledger.update_task(task_id, status="error", error=f"Error: {str(e)}")
                return None
        self.ledger.update_task(task_id, status="error", error="Max steps exceeded.")
        return None

class CoordinatorAgent:
    def __init__(self, llm, ledger: Optional[AbstractLedger]=None, subagent_args: Optional[dict]=None):
        self.llm = llm
        self.ledger = ledger or InMemoryLedger()
        self.subagent_args = subagent_args or {}

    def break_down_task(self, prompt: str) -> List[str]:
        breakdown_prompt = PromptTemplate(
            input_variables=["task"],
            template=(
                "You are a helpful assistant. Break down the following user request into a numbered list of self-contained steps or activities, "
                "making each step actionable and clear. Only output the list.\nUser request: {task}\nList of steps:"
            )
        )
        steps_raw = self.llm.invoke(breakdown_prompt.format(task=prompt)).content
        steps = [line.split(". ", 1)[-1].strip() for line in steps_raw.split("\n") if line.strip() and line[0].isdigit()]
        return steps

    def consolidate_results(self, original_prompt: str, steps: List[str], ledger: Dict[str, Any]) -> str:
        results = []
        errors = []
        for task_id, info in ledger.items():
            if info['status'] == 'completed' and info['result']:
                results.append(f"Step: {info['description']}\nResult: {info['result']}")
            elif info['status'] == 'error':
                errors.append(f"Step: {info['description']}\nError: {info['error']}")
        synth_prompt = PromptTemplate(
            input_variables=["user_request", "step_results", "step_errors"],
            template=(
                "The user request was: {user_request}\n\n"
                "The following steps were executed, with their results:\n{step_results}\n\n"
                "Errors (if any):\n{step_errors}\n\n"
                "Based on the above, produce a final, consolidated answer, report, or output that best fulfills the user's original request."
            )
        )
        step_results_str = "\n---\n".join(results) if results else "No results."
        step_errors_str = "\n---\n".join(errors) if errors else "No errors."
        final_output = self.llm.invoke(
            synth_prompt.format(
                user_request=original_prompt,
                step_results=step_results_str,
                step_errors=step_errors_str
            )
        ).content
        return final_output

    def run(self, prompt: str):
        steps = self.break_down_task(prompt)
        for i, step in enumerate(steps):
            task_id = f"task_{i+1}"
            self.ledger.add_task(task_id, step)
            subagent = SubAgent(self.llm, self.ledger, **self.subagent_args)
            subagent.execute(task_id, step)
        ledger_summary = self.ledger.summary()
        final_output = self.consolidate_results(prompt, steps, ledger_summary)
        return {
            "steps": steps,
            "ledger": ledger_summary,
            "final_output": final_output,
        }

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    coordinator = CoordinatorAgent(llm, subagent_args={"max_steps": 3, "timeout": 20})
    prompt = "Write a report about the increase of measles cases in the US. Identify sources, pull the necessary data, and generate a report in markdown with the key outcomes and your analysis"
    result = coordinator.run(prompt)
    print("Final Output:\n", result["final_output"])
    print("\nSteps Identified:")
    for step in result["steps"]:
        print("-", step)
    print("\nLedger Summary:")
    for tid, info in result["ledger"].items():
        print(f"{tid}: {info}")