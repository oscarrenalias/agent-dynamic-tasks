import time
import os
from typing import List, Dict, Any, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import concurrent.futures
from dotenv import load_dotenv
from lib.logging import logger, LoggingMixin

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

class SubAgent(LoggingMixin):
    def __init__(self, llm, ledger: AbstractLedger, max_steps=3, timeout=30):
        super().__init__()
        self.llm = llm
        self.ledger = ledger
        self.max_steps = max_steps
        self.timeout = timeout

    def execute(self, task_id: str, description: str, step_num: int = 1, total_steps: int = 1) -> Optional[str]:
        self.log_step_start(step_num, total_steps, description)
        
        prompt = PromptTemplate(
            input_variables=["instruction"],
            template="Follow this instruction step by step and provide a concise answer:\nInstruction: {instruction}"
        )
        
        for attempt in range(self.max_steps):
            try:
                formatted_prompt = prompt.format(instruction=description)
                self.logger.step_input(step_num, formatted_prompt)
                
                self.logger.info(f"[{task_id}] Attempt {attempt + 1}/{self.max_steps} with timeout {self.timeout}s", "SUBAGENT")
                
                result = run_with_timeout(
                    self.llm.invoke,
                    (formatted_prompt,),
                    self.timeout
                ).content
                
                self.logger.step_output(step_num, result)
                self.ledger.update_task(task_id, status="completed", result=result)
                self.log_step_complete(step_num, total_steps)
                
                return result
                
            except TimeoutError as e:
                error_msg = f"Timeout after {self.timeout}s on attempt {attempt + 1}"
                self.logger.warning(f"[{task_id}] {error_msg}", "SUBAGENT")
                if attempt == self.max_steps - 1:  # Last attempt
                    self.ledger.update_task(task_id, status="error", error=str(e))
                    self.log_step_error(step_num, total_steps, str(e))
                    return None
                    
            except Exception as e:
                error_msg = f"Error on attempt {attempt + 1}: {str(e)}"
                self.logger.error(f"[{task_id}] {error_msg}", "SUBAGENT")
                if attempt == self.max_steps - 1:  # Last attempt
                    self.ledger.update_task(task_id, status="error", error=f"Error: {str(e)}")
                    self.log_step_error(step_num, total_steps, str(e))
                    return None
        
        # If we get here, all attempts failed
        error_msg = "Max attempts exceeded."
        self.ledger.update_task(task_id, status="error", error=error_msg)
        self.log_step_error(step_num, total_steps, error_msg)
        return None

class CoordinatorAgent(LoggingMixin):
    def __init__(self, llm, ledger: Optional[AbstractLedger]=None, subagent_args: Optional[dict]=None):
        super().__init__()
        self.llm = llm
        self.ledger = ledger or InMemoryLedger()
        self.subagent_args = subagent_args or {}

    def break_down_task(self, prompt: str) -> List[str]:
        self.logger.info("Starting task breakdown", "COORDINATOR")
        self.logger.info(f"Original request: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        breakdown_prompt = PromptTemplate(
            input_variables=["task"],
            template=(
                "You are a helpful assistant. Break down the following user request into a numbered list of self-contained steps or activities, "
                "making each step actionable and clear. Only output the list.\nUser request: {task}\nList of steps:"
            )
        )
        
        try:
            steps_raw = self.llm.invoke(breakdown_prompt.format(task=prompt)).content
            steps = [line.split(". ", 1)[-1].strip() for line in steps_raw.split("\n") if line.strip() and line[0].isdigit()]
            
            self.logger.breakdown_complete(steps)
            self.logger.success(f"Task broken down into {len(steps)} steps", "COORDINATOR")
            
            return steps
        except Exception as e:
            self.logger.error(f"Failed to break down task: {str(e)}", "COORDINATOR")
            return []

    def consolidate_results(self, original_prompt: str, steps: List[str], ledger: Dict[str, Any]) -> str:
        self.logger.info("Starting result consolidation", "COORDINATOR")
        
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
        
        try:
            step_results_str = "\n---\n".join(results) if results else "No results."
            step_errors_str = "\n---\n".join(errors) if errors else "No errors."
            
            self.logger.info("Sending consolidation request to LLM", "CONSOLIDATION")
            
            final_output = self.llm.invoke(
                synth_prompt.format(
                    user_request=original_prompt,
                    step_results=step_results_str,
                    step_errors=step_errors_str
                )
            ).content
            
            self.logger.success("Result consolidation completed", "COORDINATOR")
            self.logger.final_result(final_output)
            
            return final_output
        except Exception as e:
            self.logger.error(f"Failed to consolidate results: {str(e)}", "COORDINATOR")
            return "Error occurred during result consolidation."

    def run(self, prompt: str):
        start_time = time.time()
        self.logger.print_separator("Agent Coordination Starting")
        self.logger.info(f"Initializing coordinator for task", "MAIN")
        
        steps = self.break_down_task(prompt)
        if not steps:
            self.logger.error("No steps generated, aborting execution", "MAIN")
            return {"steps": [], "ledger": {}, "final_output": "Failed to break down task."}
        
        self.logger.print_separator("Step Execution Phase")
        self.logger.info(f"Executing {len(steps)} steps", "EXECUTOR")
        
        # Execute steps with proper logging
        for i, step in enumerate(steps):
            task_id = f"task_{i+1}"
            self.ledger.add_task(task_id, step)
            subagent = SubAgent(self.llm, self.ledger, **self.subagent_args)
            subagent.execute(task_id, step, step_num=i+1, total_steps=len(steps))
        
        self.logger.print_separator("Step Execution Complete")
        
        ledger_summary = self.ledger.summary()
        final_output = self.consolidate_results(prompt, steps, ledger_summary)
        
        # Log execution summary
        total_time = time.time() - start_time
        successful_steps = sum(1 for info in ledger_summary.values() if info['status'] == 'completed')
        errors = sum(1 for info in ledger_summary.values() if info['status'] == 'error')
        
        self.logger.execution_summary(total_time, successful_steps, len(steps), errors)
        self.logger.print_separator("Agent Coordination Complete")
        
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
    
    # The final output and detailed results are already logged by the enhanced logger
    # No need for additional print statements as everything is logged in real-time