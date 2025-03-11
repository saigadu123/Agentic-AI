from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

agent = Agent(
    model = OpenAIChat(id="gpt-4o"),
    description = "You are an assistant please reply based on the user queries",
    tools = [DuckDuckGoTools()],
    markdown = True
)

agent.print_response("Tell me the recent sport news that you know",stream=True)