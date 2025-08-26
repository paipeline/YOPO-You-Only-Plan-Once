"""
File Processing Agent for YOPO System

A specialized agent designed for processing, analyzing, and extracting information from various file types.
This agent implements advanced file handling capabilities including text extraction, content analysis,
structured data processing, and intelligent summarization for comprehensive file understanding.
"""

import json
import os
import asyncio
from typing import Dict, Any, Optional, List, TypedDict, Annotated
import yaml
import base64

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.core.operators.llm.llm_wrapped_agent import create_wrapped_agent
from src.core.operators.llm.llm_provider import llm_instance, LLMProvider
from src.core.operators.utils import MarkitdownConverter
from src.core.operators.base_op import BaseOperator, OperatorResult


class FileProcessingState(TypedDict):
    """State object for the File Processing workflow"""
    # Input
    original_query: str
    sources: List[str]
    context: Optional[Dict[str, Any]] = None

    # File processing results
    extracted_content: List[dict]
    # LLM analysis results
    analysis_result: str
    error_msg: Optional[str] = None
    
    # Messages for LLM interaction
    messages: Annotated[List[BaseMessage], add_messages]


def prepare_content_for_analysis(content: str) -> str:
    """Prepare extracted content for LLM analysis."""
    max_length = 10000  # Limit content length for LLM
    
    if len(content) > max_length:
        content = content[:max_length] + f"\n\n[Content truncated for analysis - original length: {len(content)} characters]"
    
    return content

def process_image_file(source: str):
    ext = os.path.splitext(source)[-1].lower()
    with open(source, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    return{
            "type": "image",
            "source_type": "base64",
            "data": image_data,
            "mime_type": f"image/{ext.strip('.')}"
        }

def process_file(source: str, converter: MarkitdownConverter):
    content = []
    ext = os.path.splitext(source)[-1].lower()

    if ext in ['.png', '.jpg', '.jpeg']:
        content.append(process_image_file)
    else:
        try:
            extracted_text = converter.convert(source).text_content
        except Exception as e:
            extracted_text = f"Failed to extract content from {source}. Error: {e}"

        content.append(
            {
                "type": "text",
                "text": " - Attached file content: \n\n" + extracted_text,
            }
        )
    
    return content


# Node Functions - Independent of the operator class
async def file_processing_node(state: FileProcessingState, converter: MarkitdownConverter) -> FileProcessingState:
    """
    Node 1: File reading and processing using PIL, markitdown, etc.
    This node doesn't use LLM - just processes files directly.
    """
    print("🔄 Starting file processing node...")
    sources: List[str] = state["sources"]
    print(f"📁 Processing {len(sources)} source(s): {sources}")
    
    file_contents = []
    try:
        for i, source in enumerate(sources):
            print(f"📄 Processing file {i+1}/{len(sources)}: {source}")
            processed_content = process_file(source, converter)
            file_contents.extend(processed_content)
            print(f"✅ Successfully processed {source}, extracted {len(processed_content)} content item(s)")
        
        print(f"🎯 Total extracted content items: {len(file_contents)}")
        state.update({
            "extracted_content": file_contents
        })
        print("✅ File processing node completed successfully")

    except Exception as e:
        print(f"❌ Error in file processing node: {str(e)}")
        state.update({
            "extracted_content": [],
            "error_msg": str(e)
        })
    
    return state

async def llm_analysis_node(state: FileProcessingState, llm_provider: LLMProvider) -> FileProcessingState:
    """
    Node 2: LLM-based understanding and analysis of the processed file content.
    """
    print("🤖 Starting LLM analysis node...")
    
    original_query = state["original_query"]
    extracted_content = state["extracted_content"]
    context: Optional[Dict[str, Any]] = state["context"]
    messages = state.get("messages", [])

    user_query = f"Query: {original_query}" if context is None else f"Context: {json.dumps(context, indent=2)}\n\n Query: {original_query}"

    try:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_query}] + extracted_content
        })
        
        print("🚀 Sending request to LLM provider...")
        # Get LLM response
        response = llm_provider.generate(messages)
        print(f"✅ Received LLM response (length: {len(response)} chars)")
        
        state.update({
            "analysis_result": response,
            "messages": messages + [AIMessage(content=response)]
        })
        print("✅ LLM analysis node completed successfully")
        
    except Exception as e:
        print(f"❌ Error in LLM analysis node: {str(e)}")
        error_response = f"Analysis failed: {str(e)}"
        state.update({
            "analysis_result": error_response,
            "messages": [HumanMessage(content=f"Error: {str(e)}")]
        })
    
    return state

class FileProcessingOperator(BaseOperator):
    """
    File Processing Operator for YOPO System
    
    A specialized operator designed for processing and analyzing various file types including documents,
    images, PDFs, spreadsheets, and other common file formats. This operator combines file content
    extraction with LLM-based analysis to provide intelligent insights and answers about file contents.
    
    The operator uses a two-stage workflow:
    1. File Processing: Extract and convert file content to a standardized format
    2. LLM Analysis: Analyze the extracted content using language models to answer queries
    
    Supported file types include but are not limited to:
    - Documents: PDF, DOCX, TXT, MD
    - Images: JPG, PNG, GIF, BMP, TIFF
    - Spreadsheets: XLSX, CSV
    - Presentations: PPTX
    - Audio/Video: Basic metadata extraction
    
    Key Features:
    - Multi-format file processing and content extraction
    - Intelligent content analysis using LLM capabilities
    - Structured workflow with error handling
    - Support for batch file processing
    - Context-aware query answering about file contents
    """
    
    def __init__(self, 
                 prompt_path: str,
                 **kwargs):
        # Load all prompts from YAML
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            system_prompt = config.get('system_prompt', None)
            description = config.get('description', None)

        # Initialize base agent
        name = "FileProcessingOperator"
        super().__init__(name=name, description=description, model_name=os.getenv("MULTIMODEL_MODEL_NAME", "gpt-4o"), system_prompt=system_prompt, **kwargs)

        self.converter: MarkitdownConverter = MarkitdownConverter()

    def _build_workflow(self):
        """Build the LangGraph workflow for file processing."""
        workflow = StateGraph(FileProcessingState)
        
        # Create wrapper functions that pass the required parameters
        async def _file_processing_wrapper(state: FileProcessingState) -> FileProcessingState:
            return await file_processing_node(state, self.converter)
        
        async def _llm_analysis_wrapper(state: FileProcessingState) -> FileProcessingState:
            return await llm_analysis_node(state, self.llm_provider)
        
        # Add nodes
        workflow.add_node("file_processor", _file_processing_wrapper)
        workflow.add_node("llm_analyzer", _llm_analysis_wrapper)
        
        # Define the workflow edges
        workflow.add_edge(START, "file_processor")
        workflow.add_edge("file_processor", "llm_analyzer")
        workflow.add_edge("llm_analyzer", END)
        
        return workflow.compile()

    async def arun(self, query: str, context: Optional[Dict[str, Any]] = None, sources: Optional[List[str]] = []) -> OperatorResult:
        """
        Run the file processing workflow.
        
        Args:
            query: User query about the file
            context: Additional context containing file information
            
        Returns:
            OperatorResult with processed file analysis
        """
        try:
            # Initialize state
            initial_state = FileProcessingState(
                original_query=query,
                sources=sources,
                context=context,
                extracted_content=[],
                analysis_result="",
                error_msg=None,
                messages=[]
            )
            
            # Run the workflow
            final_state = await self.worklfow.ainvoke(initial_state)
            
            # Prepare result
            result = OperatorResult(
                base_result=final_state.get("analysis_result")
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"File processing failed: {str(e)}")
            return OperatorResult(
                base_result=f"File processing failed: {str(e)}",
                metadata={"error": str(e)}
            )

        
