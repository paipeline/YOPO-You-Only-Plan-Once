"""
GAIA Benchmark Implementation for YOPO System

This module provides the GAIABenchmark class for evaluating YOPO operators
on the GAIA (General AI Assistants) benchmark dataset.

GAIA is a benchmark that evaluates AI systems' ability to:
- Follow specific output formatting rules
- Provide step-by-step reasoning
- Give precise final answers in a standardized format
- Handle complex multi-step queries requiring web search and tool usage
"""

import os
import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from datasets import DatasetDict, Dataset

from base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)


class GAIABenchmark(BaseBenchmark):
    """
    GAIA Benchmark implementation for YOPO system.
    
    This benchmark evaluates YOPO operators on the GAIA dataset,
    which tests AI systems' ability to handle complex reasoning tasks
    with specific output formatting requirements.
    
    Key Features:
    - GAIA-compliant question processing
    - Automatic answer extraction using "FINAL ANSWER:" pattern
    - Support for attached files and context
    - Level-based evaluation (Level 1, 2, 3)
    - Submission file generation for official evaluation
    """

    def __init__(self, 
                 configs: Optional[List[str]] = None):
        """
        Initialize GAIA Benchmark.
        
        Args:
            attached_files_dir: Directory containing attached files for GAIA questions
            configs: List of dataset configurations to load (e.g., ['level1', 'level2'])
        """
        super().__init__(
            name="GAIA",
            # local_path="your_dir_to_gaia",
            local_path="/Users/jakcieshi/Desktop/Home/Projects/FreeLan/2023",
            configs=configs or ["1", "2", "3"]
        )
        
    def _extract_ground_truth_answer(self, text: str) -> Optional[str]:
        """
        Extract final answer from model response using GAIA format.
        
        Args:
            response: Model response text
            
        Returns:
            Extracted final answer or None if not found
        """
        return text

    def _load_data_from_local(self):
        dataset_dict = DatasetDict()

        for split_name in os.listdir(self.local_path):
            if split_name.startswith("."):
                continue
            queries, answers = [], []
            file_names = []
            metadata_path = os.path.join(self.local_path, split_name, "metadata.jsonl")
            with open(metadata_path, 'r') as f:
                for line in f.readlines():
                    item = json.loads(line)
                    if str(item["Level"]) not in self.configs:
                        continue
                    queries.append(item["Question"])
                    answers.append(str(item["Final answer"]) if item["Final answer"] != "?" else None)
                    if item["file_name"] != "":
                        file_names.append(os.path.join(self.local_path, split_name, item["file_name"]))
                    else: file_names.append(None)
            
            datas = {
                "query": queries,
                "answer": answers,
                "file_name": file_names
            }

            if split_name == "test":
                dataset_dict["test"] = Dataset.from_dict(datas)
            else:
                dataset_dict["train"] = Dataset.from_dict(datas)

        return {"default": dataset_dict}
                
    def _format_dataset(self) -> Dict[str, Dataset]:
        result: Dict[str, Dataset] = defaultdict()
        queries: Dict[str, List[str]] = defaultdict(list)
        answers: Dict[str, List[str]] = defaultdict(list)
        file_names: Dict[str, List[str]] = defaultdict(list)

        for config, dataset in self.datasets.items():
            for split_name in dataset.keys():
                for data in dataset[split_name]:
                    query: str = data["query"]
                    answer: Optional[str] = self._extract_ground_truth_answer(data["answer"])
                    if data["file_name"]:
                        query += f"\n\nTo solve the task above, you will have to use these attached files: {data['file_name']}"
                    if split_name == "test":
                        queries[split_name].append(query)
                        answers[split_name].append(answer)
                        file_names[split_name].append(data["file_name"])
                    else:
                        queries["train"].append(query)
                        answers["train"].append(answer)
                        file_names["train"].append(data["file_name"])

        for key in queries.keys():
            result[key] = Dataset.from_dict({
                "query": queries[key],
                "answer": answers[key],
                "file_name": file_names[key]
            })

        return result

    def get_csv_columns(self) -> List[str]:
        return ["query", "answer", "agent_answer", "file_name", "score"]

    async def _run_single_evaluation(self, data_dict: Dict[str, Any], agent):
        # FIXME: parse response
        query: str = data_dict["query"]
        answer: str = data_dict["answer"]

        response = await agent.ainvoke(query)
        score = 0.
        if response == answer:
            score = 1.0
        
        return query, answer, response, data_dict["file_name"], score
