"""
Base Benchmark Class for YOPO System

This module provides the abstract base class for all benchmark implementations.
It handles dataset loading using the datasets library and defines the interface
for data processing that must be implemented by concrete benchmark classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Iterator
from pathlib import Path
import logging

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
            description: Description of the benchmark
            cache_dir: Directory to cache downloaded datasets
            **kwargs: Additional configuration parameters
        """
        self.name = name

        self.huggingface_path = huggingface_path
        self.local_path = local_path
        self.configs = configs

        self.datasets: Dict[str, DatasetDict] = self._load_dataset()
    
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
    
    @abstractmethod
    def _load_data_from_local(self):
        """
        Load data from local file system.
        
        This method must be implemented by all concrete benchmark classes
        to define how to load benchmark data from local files when
        local_path is specified instead of huggingface_path.
        
        Returns:
            The loaded data in the appropriate format for the benchmark
        """

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
    def format_dataset(self) -> Dict[str, Dataset]:
        """
        Process a single data example or batch of examples.
        
        This method must be implemented by all concrete benchmark classes
        to define how the loaded data should be processed, evaluated, or
        transformed for the specific benchmark requirements.
        
        Args:
            data: Single example (dict) or batch of examples (list of dicts)
            **kwargs: Additional processing parameters
        
        Returns:
            Processed result (format depends on concrete implementation)
        """
    