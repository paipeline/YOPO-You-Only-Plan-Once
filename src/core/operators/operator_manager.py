
import os
from typing import Dict, Any, Optional, List, Type, get_type_hints
import logging
import inspect
import importlib
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.operators.base_op import BaseOperator

class OperatorInfo(BaseModel):
    """Information about an operator including its metadata and argument schema."""
    name: str = Field(description="Operator name")
    description: str = Field(description="Operator description") 
    class_name: str = Field(description="Operator class name")
    module_path: str = Field(description="Module path where operator is defined")
    args_schema: Dict[str, Any] = Field(description="Arguments schema for arun method")

########################################################################
######################## MAIN CLASSES ##################################
########################################################################

class OperatorManager:
    """
    Operator Zoo - Centralized registry for all operators in the YOPO system.
    
    This class provides a unified interface to discover, inspect, and instantiate
    all available operators in the system.
    """
    
    def __init__(self, auto_discover: bool = True):
        """
        Initialize the operator zoo.
        
        Args:
            auto_discover: Whether to automatically discover operators on initialization
        """
        self._operators: Dict[str, Type[BaseOperator]] = {}
        self._operator_info: Dict[str, OperatorInfo] = {}
        self._operator_instance: Dict[str, BaseOperator] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.prompts_map = self._load_prompts_map()

        if auto_discover:
            self._discover_operators()

    def _load_prompts_map(self) -> Dict[str, str]:
        """
        Load all prompt files from the prompts directory and create a mapping.
        
        Returns:
            Dict mapping operator names to their prompt file paths.
            Key format: first_two_parts_of_filename (e.g., "deep_research" from "deep_research_op.yaml")
            Value: full file path
        """
        prompts_map = {}
        
        # Get the prompts directory relative to the project root
        current_dir = Path(__file__).parent
        prompts_dir = current_dir / "prompts"
        
        if not prompts_dir.exists():
            self.logger.warning(f"Prompts directory not found: {prompts_dir}")
            return prompts_map
        
        # Find all YAML files in prompts directory
        for file_path in prompts_dir.glob("*.yaml"):
            try:
                # Split filename by underscore and take all parts except the last one
                filename_parts = file_path.stem.split("_")
                if len(filename_parts) >= 2:
                    key = "_".join(filename_parts[:-1])
                    prompts_map[key] = str(file_path)
                    self.logger.debug(f"Loaded prompt mapping: {key} -> {file_path}")
                else:
                    self.logger.warning(f"Prompt file {file_path.name} doesn't follow expected naming convention")
                    
            except Exception as e:
                self.logger.error(f"Failed to process prompt file {file_path}: {e}")
        
        return prompts_map
    
    def _discover_operators(self) -> None:
        """
        Automatically discover all available operator classes in the operators module.
        """
        operators_dir = Path(__file__).parent
        
        # Get all Python files in the operators directory
        for file_path in operators_dir.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name in ["base_op.py", "utils.py"]:
                continue
                
            module_name = file_path.stem
            try:
                # Import the module
                module = importlib.import_module(f".{module_name}", package="src.core.operators")
                
                # Find all classes that inherit from BaseOperator
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # if name == "MCPOperator":
                    #     breakpoint()
                    if (issubclass(obj, BaseOperator) and 
                        obj is not BaseOperator):
                        _key = "_".join(file_path.name.split('_')[:-1])
                        # Register the operator
                        self._register_operator(obj, module_name, prompt_path=self.prompts_map[_key])
                        
                        self.logger.info(f"Discovered operator: {_key} ({obj.__name__})")
                        
            except Exception as e:
                self.logger.warning(f"Failed to import operator module {module_name}: {e}")
    
    def _register_operator(self, operator_class: Type[BaseOperator], module_name: str, prompt_path: Optional[str] = None) -> None:
        """Register an operator with its metadata."""
        try:
            # Extract operator info
            instance = operator_class(prompt_path)
            name = instance.name
            description = instance.description
            self._operator_instance[instance.name] = instance
            # Store operator class and info
            self._operators[name] = operator_class

            # Get arun method argument schema
            arun_schema = self._extract_arun_schema(operator_class)
            self._operator_info[name] = OperatorInfo(
                name=name,
                description=description,
                class_name=operator_class.__name__,
                module_path=f"src.core.operators.{module_name}",
                args_schema=arun_schema
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register operator {name}: {e}")

    def _extract_arun_schema(self, operator_class: Type[BaseOperator]) -> Dict[str, Any]:
        """Extract argument schema for the arun method."""
        try:
            arun_method = getattr(operator_class, 'arun')
            sig = inspect.signature(arun_method)
            type_hints = get_type_hints(arun_method)
            
            schema = {}
            for param_name, param in sig.parameters.items():
                if param_name in ['self']:
                    continue
                
                param_info = {
                    "type": str(type_hints.get(param_name, param.annotation)),
                    "required": param.default == inspect.Parameter.empty,
                    "default": param.default if param.default != inspect.Parameter.empty else None
                }
                
                
                schema[param_name] = param_info
            
            return schema
            
        except Exception as e:
            self.logger.warning(f"Failed to extract arun schema for {operator_class.__name__}: {e}")
            return {}
    

    # Public Interface Methods
    
    def get_all_operator_infos(self) -> List[OperatorInfo]:
        """
        Interface 1: Get all operators with their metadata and argument schemas.
        
        Returns:
            List[OperatorInfo]: List of all available operators with their complete information
        """
        return list(self._operator_info.values())
    
    def get_operator_instance(self, operator_name: str) -> BaseOperator:
        """
        Interface 2: Get an operator instance by name.
        
        Args:
            operator_name: Name of the operator to create
            **kwargs: Arguments to pass to the operator constructor
            
        Returns:
            BaseOperator: Instance of the requested operator
            
        Raises:
            ValueError: If operator_name is not found
            RuntimeError: If operator dependencies are not available
        """
        if operator_name not in self._operator_instance:
            available = list(self._operator_instance.keys())
            raise ValueError(f"Operator '{operator_name}' not found. Available operators: {available}")
        
        return self._operator_instance[operator_name]
    
    # Additional utility methods
    
    def list_operator_names(self) -> List[str]:
        """Get list of all available operator names."""
        return list(self._operator_instance.keys())
    
    def get_operator_info(self, operator_name: str) -> OperatorInfo:
        """Get detailed information about a specific operator."""
        if operator_name not in self._operator_info:
            available = list(self._operator_info.keys())
            raise ValueError(f"Operator '{operator_name}' not found. Available operators: {available}")
        
        return self._operator_info[operator_name]

# Global operator zoo instance
_global_operator_manager: Optional[OperatorManager] = None


def get_operator_manager() -> OperatorManager:
    """
    Get the global operator zoo instance.
    
    Returns:
        OperatorZoo: Global operator zoo instance
    """
    global _global_operator_manager
    if _global_operator_manager is None:
        _global_operator_manager = OperatorManager()
    return _global_operator_manager


# Convenience functions for the two main interfaces
########################################################################
######################## MAIN ENTRY FUNCTION ###########################
########################################################################

def get_all_operators() -> List[OperatorInfo]:
    """
    Convenience function for Interface 1: Get all operators with metadata.
    
    Returns:
        List[OperatorInfo]: List of all available operators with their complete information
    """
    return get_operator_manager().get_all_operator_infos()


def get_operator_instance(operator_name: str, **kwargs) -> BaseOperator:
    """
    Convenience function for Interface 2: Get operator instance by name.
    
    Args:
        operator_name: Name of the operator to create
        **kwargs: Arguments to pass to the operator constructor
        
    Returns:
        BaseOperator: Instance of the requested operator
    """
    return get_operator_manager().get_operator_instance(operator_name, **kwargs)




if __name__ == "__main__":
    # Test the FileProcessingOperator
    import asyncio
    from src.core.operators.file_processing_op import FileProcessingOperator
    from src.core.operators.browser_use_op import BrowserUseOperator
    from src.tool.tool_manager import get_tools

    async def test_file_processing_operator():
        """Simple test for FileProcessingOperator."""
        print("Testing FileProcessingOperator...")
        
        file_op = FileProcessingOperator(prompt_path="./prompts/file_processing_prompts.yaml")
        
        result = await file_op.arun(query="what is the file about?", sources=["/Users/jakcieshi/Desktop/Home/Projects/FreeLan/2023/test/943255a6-8c56-4cf8-9faf-c74743960097.csv"])
        print(f"Result: {result.base_result}")
    
    async def test_browser_use_operator():
        """Simple test for BrowserUseOperator."""
        print("Testing BrowserUseOperator...")
        
        browser_op = BrowserUseOperator(prompt_path="./prompts/browser_use_prompts.yaml")
        
        result = await browser_op.arun(query="Go to google.com and search for 'OpenAI'. Finally give me one of the founder of this company.")
        print(f"Result: {result.base_result}")
    
    # Run the test
    # asyncio.run(test_browser_use_operator())


    # l = get_all_operators()
    # for info in l:
    #     print(info.model_dump())
    op = get_operator_instance(operator_name="DeepResearchOperator")
    async def func(op: BaseOperator):
        # result = await op.arun(query="The wether in Beijing", tools=get_tools(tool_base_names=["firecrawl_search"]))
        result = await op.arun(query="The founder of OpenAI. Just give me the name.")
        print(result)
    
    asyncio.run(func(op))
    