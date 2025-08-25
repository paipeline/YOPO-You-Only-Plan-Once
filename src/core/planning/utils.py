from typing import List
from src.core.agents.llm.llm_provider import llm_instance
import os

system_prompt = "You are a tag identifier agent. You are given a query and you need to identify the tags that are most relevant to the query."

# get tags from query
def get_tags_from_query(query: str) -> List[str]:
    """
    Get tags from query using analysis-focused system prompt.
    """
    llm = llm_instance(os.getenv("DEV_MODEL_NAME"), system_prompt=system_prompt)
    response = llm.generate([{"role": "user", "content": query}])
    return response
