"""
FireCrawl Tool for YOPO System

This module provides FireCrawl web scraping and crawling capabilities
using LangChain tools and the FirecrawlApp library.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp

from src.tool.base_tool import YOPOBaseTool, BaseToolConfig

load_dotenv()


class SearchResult(BaseModel):
    """
    Result model for search operations.
    
    This model represents the output of search tools, containing
    search metadata and a list of found items.
    """
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    description: str = Field(description="Description of the search result")
    error_msg: Optional[str] = Field(default=None, description="Error message if search failed")

class SearchDetailResult(BaseModel):
    """
    Result model for detailed content fetching operations.
    
    This model represents the output of content fetching tools,
    containing the full content and metadata of a specific URL.
    """
    url: str = Field(description="The URL that was fetched")
    content: str = Field(description="Main content of the page in markdown format")


class FireCrawlTool(YOPOBaseTool):
    """
    FireCrawl tool implementation for web scraping and crawling.
    
    This tool provides web scraping capabilities using the FireCrawl service,
    wrapped in LangChain tools for easy integration with agents.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize FireCrawl tool.
        
        Args:
            api_key: FireCrawl API key (will use FIRECRAWL_API_KEY env var if not provided)
            base_url: FireCrawl base URL
            **kwargs: Additional configuration parameters
        """
        config = BaseToolConfig(
            name="firecrawl_search",
            description="Web scraping and crawling tool using FireCrawl service",
            **kwargs
        )
        self.api_key = os.getenv("FIRECRAWL_API_KEY", None)
        self.max_length = 4096
        super().__init__(config)
    
    def get_langchain_tools(self) -> List[BaseTool]:
        """Create LangChain tools for FireCrawl functionality."""
        
        @tool
        def search(query: str, 
                  max_items: int = 10,
                  filter_year: Optional[int] = None) -> List[SearchResult]:
            """
            Search the web using FireCrawl search capabilities.
            
            Args:
                query: Search query string
                max_items: Maximum number of items to return (default: 10)
                filter_year: Optional year filter to limit results to specific year
                
            Returns:
                SearchResult: Structured search results with metadata
            """
            start_time = time.time()
            
            try:
                # Initialize FireCrawl app
                init_params = {}
                init_params["api_key"] = self.api_key
                
                app = FirecrawlApp(**init_params)
                
                tbs = None
                if filter_year is not None:
                    tbs = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

                # Perform search
                result = app.search(
                    query=query,
                    limit=max_items,
                    tbs=tbs
                )
                search_time = time.time() - start_time
                
                search_results = []
                for item in result.web:
                    title = item.title
                    url = item.url
                    description = item.description
                    search_results.append(SearchResult(
                        title=title,
                        url=url,
                        description=description
                    ))
                
                return search_results
                    
            except Exception as e:
                search_time = time.time() - start_time
                return [SearchResult(
                    title="Search Failed",
                    url="",
                    description="",
                    error_msg=str(e)
                )]
        
        @tool
        def fetch(url: str) -> Dict[str, Any]:
            """
            Fetch detailed content from a specific URL using FireCrawl.
            
            Args:
                url: The URL to fetch content from
                
            Returns:
                SearchDetailResult: Detailed content and metadata from the URL
            """
            
            start_time = time.time()
            
            try:
                # Initialize FireCrawl app
                init_params = {}
                init_params["api_key"] = self.api_key
                
                app = FirecrawlApp(**init_params)
                
                # Perform scraping
                result = app.scrape(url)
                fetch_time = time.time() - start_time
                
                # TODO: more advanced processing
                markdown_content = result.markdown
                if len(markdown_content) > self.max_length:
                    markdown_content = markdown_content[:self.max_length] + "..."
                
                return SearchDetailResult(
                    url=url,
                    content=markdown_content
                    )
            except Exception as e:
                fetch_time = time.time() - start_time
                return SearchDetailResult(
                    url=url,
                    content=""
                )
        
        return [search, fetch]