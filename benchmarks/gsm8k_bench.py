"""
GSM8K Benchmark Implementation for YOPO System

This module provides the GSM8KBenchmark class for evaluating YOPO operators
on the GSM8K (Grade School Math 8K) benchmark dataset.

GSM8K is a benchmark that evaluates AI systems' ability to:
- Solve grade school level math word problems
- Provide step-by-step reasoning
- Give precise numerical answers
- Handle arithmetic and basic algebraic reasoning
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from datasets import Dataset

from base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)
 

class GSM8KBenchmark(BaseBenchmark):
    """
    GSM8K Benchmark implementation for YOPO system.
    
    This benchmark evaluates YOPO operators on the GSM8K dataset,
    which tests AI systems' ability to solve grade school math problems
    requiring multi-step reasoning.
    
    Key Features:
    - Math word problem processing
    - Automatic answer extraction from step-by-step solutions
    - Numerical answer comparison
    - Support for both training and test splits
    """
    
    def __init__(self, configs: Optional[List[str]] = None):
        """
        Initialize GSM8K Benchmark.
        
        Args:
            configs: List of dataset configurations to load (e.g., ["main", "socratic"])
        """
        super().__init__(
            name="GSM8K",
            huggingface_path="openai/gsm8k",
            configs=configs or ["main", "socratic"]
        )
                    
    def _extract_ground_truth_answer(self, text: str) -> Optional[str]:
        """
        Extract numerical answer from text.
        
        Args:
            text: Text containing the answer
            
        Returns:
            Numerical answer or None if not found
        """
        # First try to find answer marked with ####
        answer_pattern = re.compile(r'####\s*([0-9,]+(?:\.[0-9]+)?)')
        match = answer_pattern.search(text)
        if match:
            try:
                # Remove commas and convert to float
                answer_str = match.group(1).replace(',', '')
                return answer_str
            except ValueError:
                pass
        
        # Fallback: look for numbers at the end of the text
        numbers = re.findall(r'([0-9,]+(?:\.[0-9]+)?)', text)
        if numbers:
            try:
                # Take the last number found
                answer_str = numbers[-1].replace(',', '')
                return answer_str
            except ValueError:
                pass
        
        return None
    
    def format_dataset(self) -> Dict[str, Dataset]:
        result: Dict[str, Dataset] = defaultdict()
        queries: Dict[str, List[str]] = defaultdict(list)
        answers: Dict[str, List[str]] = defaultdict(list)

        for config, dataset in self.datasets.items():
            for split_name in dataset.keys():
                for data in dataset[split_name]:
                    question: str = data["question"]
                    answer: Optional[str] = self._extract_ground_truth_answer(data["answer"])
                    if split_name == "test":
                        queries[split_name].append(question)
                        answers[split_name].append(answer)
                    else:
                        queries["train"].append(question)
                        answers["train"].append(answer)

        for key in queries.keys():
            result[key] = Dataset.from_dict({
                "query": queries[key],
                "answer": answers[key]
            })

        return result
    
        

        