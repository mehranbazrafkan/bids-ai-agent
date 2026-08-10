from planner import Planner
from tool_registry import ToolRegistry
from llm_client import LLMClient
from retriever import Retriever
from utils import *

system_prompt = """You are a helpful local AI assistant specializing in managing MRI datasets according to the BIDS (Brain Imaging Data Structure) specification.
Use the provided documentation and context to explain errors/warnings and help the user manage their dataset.
When fixing issues, ensure you explain what action was taken.
"""

class Agent:
    def __init__(self):
        self.llm = LLMClient()
        self.retriever = Retriever()
        self.tools = ToolRegistry()
        
        self.planner = Planner(
            llm=self.llm, 
            retriever=self.retriever,
            tools=self.tools)
        
    def run(self, user_input: str, context: dict) -> str:
        """
        Main Entry point for the agent
        """
        response = self.planner.execute(system_prompt=system_prompt, user_input=user_input, context=context)
        return response