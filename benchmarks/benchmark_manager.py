"""
Benchmark Manager for YOPO System

This module provides the BenchmarkManager class for managing and executing
different benchmarks in the YOPO system.
"""

from typing import Dict, Tuple, Union, Optional
from datasets import Dataset

from base_benchmark import BaseBenchmark
from gsm8k_bench import GSM8KBenchmark
from gaia_bench import GAIABenchmark


########################################################################
######################## HELPER FUNCTIONS ##############################
########################################################################

def get_benchmark(dataset_name: str) -> Dict[str, Dataset]:
    """
    Get formatted dataset for the specified benchmark.
    
    Args:
        dataset_name: Name of the dataset benchmark to format.
                     Supported values: 'gsm8k', 'gaia'
    
    Returns:
        Dict[str, Dataset]: Formatted dataset with splits as keys and Dataset objects as values
        
    Raises:
        ValueError: If dataset_name is not supported
        Exception: If there's an error during dataset formatting
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == 'gsm8k':
        return GSM8KBenchmark()
    elif dataset_name == 'gaia':
        return GAIABenchmark()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: 'gsm8k', 'gaia'")

def get_training_datasets(benchmark: BaseBenchmark) -> Tuple[Union[Dataset, None], Union[Dataset, None]]:
    return benchmark.fetch_datasets_for_training()
