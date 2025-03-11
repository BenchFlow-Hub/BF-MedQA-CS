from benchflow import BaseBench
from benchflow.schemas import BenchArgs
from benchflow.schemas import BenchmarkResult
from typing import Dict, Any
import json
import os

class MedQABench(BaseBench):
    def get_args(self, task_id: str) -> BenchArgs:
        arugments = {
            "required":["OPENAI_API_KEY"],
            "optional":{
                "CASE_ID": "1-44",
            }
        }
        return BenchArgs(arugments)

    def get_image_name(self) -> str:
        return "kirk2000/benchflow:medqa-cs-v1"

    def get_results_dir_in_container(self) -> str:
        return "/app/result"

    def get_log_files_dir_in_container(self) -> str:
        return "/app/output"

    def get_result(self, task_id: str) -> BenchmarkResult:
        """
        You should return the results in this function.
        
        Return a BenchmarkResult containing the benchmark results.

        The BenchmarkResult model has the following fields:
            - is_resolved (bool): Indicates whether the task is resolved.
            - message (dict): Contains additional information to be displayed to the agent user.
            - log (str): Contains the log output (e.g., trace, trajectory, etc).
            - metrics (dict): A dictionary of various metrics, where each metric can be of different types (e.g., bool, int, float, or str).
            - other (dict): Any extra fields or metadata relevant to the benchmark result.
        
        Please refer to the example in the definition of BenchmarkResult for the expected format.
        """
        result_path = f"{self.results_dir}/result_summary.json"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
        else:
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=False,
                log={"error": "Result file not found"},
                metrics={},
                other={}
            )
        log_path = f"{self.log_files_dir}/med-exam.json"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
        else:
            return BenchmarkResult(
                task_id=task_id,
                is_resolved=False,
                log={"error": "Log file not found"},
                metrics={},
                other={}
            )

        return BenchmarkResult(
            task_id=task_id,
            is_resolved=True,
            log=log[0],
            metrics=result,
            other={}
        )

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        return {
            "task_ids": ["qa", "physical_exam", "closure", "diagnosis"],
            "error_message": None
        }
