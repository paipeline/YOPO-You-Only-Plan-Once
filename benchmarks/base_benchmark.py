"""
Base Benchmark Class for YOPO System

This module provides the abstract base class for all benchmark implementations.
It handles dataset loading using the datasets library and defines the interface
for data processing that must be implemented by concrete benchmark classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import asyncio
import logging

import pandas as pd
from pydantic import BaseModel, Field
from datasets import Dataset, DatasetDict, load_dataset


logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmark implementations in YOPO.
    
    This class provides common functionality for loading datasets and defines
    the interface that all benchmark classes must implement for data processing.
    
    Key Features:
    - Dataset loading from various sources (HuggingFace Hub, local files, etc.)
    - Flexible data format support
    - Abstract data processing interface
    - Consistent logging and error handling
    - Memory-efficient data iteration
    """
    
    def __init__(self, 
                 name: str,
                 huggingface_path: Optional[str] = None,
                 local_path: Optional[Union[List[str], str]] = None,
                 configs: Optional[List[str]] = None):
        """
        Initialize the base benchmark.
        
        Args:
            name: Name of the benchmark
            huggingface_path: Optional path to HuggingFace dataset
            local_path: Optional path to local dataset files
            configs: Optional list of dataset configurations to load
        """
        self.name = name

        self.huggingface_path = huggingface_path
        self.local_path = local_path
        self.configs = configs

        self.datasets: Dict[str, DatasetDict] = self._load_dataset()
        self.formatted_datasets: Dict[str, Dataset] = self._format_dataset()

        logger.info(f"Initialized {self.name} benchmark")

    def _load_dataset(self) -> Dict[str, DatasetDict]:
        """
        Load dataset from HuggingFace Hub.
        
        This method handles loading datasets with different configurations.
        If configs are specified, it loads each configuration separately.
        Otherwise, it loads the default configuration.
        
        Returns:
            Dict[str, DatasetDict]: Dictionary mapping config names to DatasetDict objects
        """

        if self.local_path:
            try:
                return self._load_data_from_local()
            except Exception as e:
                raise e
        if self.huggingface_path:
            try:
                datasets: Dict[str, DatasetDict] = {}
                if self.configs:
                    for config in self.configs:
                        dataset: DatasetDict = load_dataset(self.huggingface_path, config)
                        datasets[config] = dataset
                else:
                    dataset: DatasetDict = load_dataset(self.huggingface_path)
                    datasets["default"] = dataset
                return datasets
            except Exception as e:
                raise e
        
        raise ValueError("Neither huggingface_path nor local_path specified")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        info = {
            "name": self.name,
            "dataset": self.datasets
        }
        
        return info
    
    def _load_data_from_local(self):
        """
        Load data from local file system.
        
        This method must be implemented by all concrete benchmark classes
        to define how to load benchmark data from local files when
        local_path is specified instead of huggingface_path.
        
        Returns:
            The loaded data in the appropriate format for the benchmark
        """
        raise NotImplementedError("_load_data_from_local must be implemented by subclasses")

    @abstractmethod
    def _extract_ground_truth_answer(self, text: str):
        """
        Extract ground truth answer from the given text.
        
        This method must be implemented by all concrete benchmark classes
        to define how to extract the correct/expected answer from the
        benchmark dataset's answer field or text.
        
        Args:
            text: The text containing the ground truth answer
            
        Returns:
            The extracted ground truth answer (format depends on benchmark)
        """

    @abstractmethod
    def _format_dataset(self) -> Dict[str, Dataset]:
        """
        Format and return the dataset for benchmark evaluation.
        
        This method must be implemented by all concrete benchmark classes
        to define how the loaded dataset should be formatted and structured
        for evaluation by YOPO operators.
        
        Returns:
            Dict[str, Dataset]: Dictionary mapping split names to Dataset objects
                containing formatted benchmark data ready for evaluation
        """
    
    @abstractmethod
    async def _run_single_evaluation(self, data_dict: Dict[str, Any], agent):
        """
        Run a single evaluation on a data sample with an agent.
        
        This method must be implemented by all concrete benchmark classes
        to define how to evaluate a single data sample from the benchmark
        using the provided agent. It should handle the interaction between
        the agent and the benchmark data, execute the evaluation, and
        return the results.
        
        Args:
            data_dict: Dictionary containing a single data sample from the benchmark
                      with keys like 'query', 'answer', and other benchmark-specific fields
            agent: The agent/model to be evaluated on this data sample
            
        Returns:
            The evaluation result for this single data sample (format depends on benchmark)
        """
    
    @abstractmethod
    def get_csv_columns(self) -> List[str]:
        """
        Get the column names for CSV output format.
        
        This method must be implemented by all concrete benchmark classes
        to define the column headers for CSV output when saving evaluation
        results. The columns should include at minimum the query, expected
        answer, agent response, and evaluation score.
        
        Returns:
            List[str]: List of column names for CSV output format
        """
    
    def fetch_datasets_for_training(self) -> Tuple[Union[Dataset, None], Union[Dataset, None]]:
        """
        Fetch training and test datasets for benchmark evaluation.
        
        This method retrieves the formatted training and test datasets
        from the benchmark's formatted_datasets dictionary. It looks for
        datasets with split names "test" and "training" and returns them
        as a tuple.
        
        Returns:
            Tuple[Dataset, Dataset]: A tuple containing (training_dataset, test_dataset)
                - training_dataset: Dataset object for training split, or None if not found
                - test_dataset: Dataset object for test split, or None if not found
        """
        training_dataset, test_dataset = None, None
        for split_name in self.formatted_datasets:
            if split_name == "test":
                test_dataset: Dataset = self.formatted_datasets[split_name]
            elif split_name == "train":
                training_dataset: Dataset = self.formatted_datasets[split_name]
        
        return training_dataset, test_dataset

    async def run_benchmark(self, dataset: Dataset, nums: Optional[int] = None, max_concurrency: int = 10):
        """
        """
        nums: int = nums if nums else len(dataset)
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def run_single_with_semaphore(data_dict: Dict[str, Any], agent, index: int):
            async with semaphore:
                return await self._run_single_evaluation(data_dict, agent)
        
        # Create tasks for parallel execution
        tasks: List[asyncio.Task] = []
        for i in range(nums):
            data_dict = dataset[i]
            task = asyncio.create_task(run_single_with_semaphore(data_dict, None, i))
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and calculate metrics
        csv_columns = self.get_csv_columns()
        csv_data = []
        total_score = 0.0
        valid_results = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                continue
                
            # Create CSV row from result tuple
            csv_row = {}
            for j, column in enumerate(csv_columns):
                if j < len(result):
                    csv_row[column] = result[j]
                else:
                    csv_row[column] = None
            
            csv_data.append(csv_row)
            
            # Accumulate score (assuming last column is score)
            if len(result) > 0 and isinstance(result[-1], (int, float)):
                total_score += result[-1]
                valid_results += 1
        
        # Save to CSV
        if csv_data:
            
            df = pd.DataFrame(csv_data)
            csv_filename = f"{self.name}_benchmark_results.csv"
            df.to_csv(csv_filename, index=False)
            logger.info(f"Results saved to {csv_filename}")
        
        # Calculate average score
        average_score = total_score / valid_results if valid_results > 0 else 0.0
        logger.info(f"Average score: {average_score:.4f} ({valid_results}/{len(results)} valid results)")
        
        return results, average_score


