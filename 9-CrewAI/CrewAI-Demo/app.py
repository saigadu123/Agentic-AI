from crewai import Crew,Agent,Task,LLM
from crewai_tools import SerperDevTool 
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model = "gemini/gemini-2.0-flash",
    temperature = 0.7
)

topic = "AI in HealthCare"

# search tool
search_tool = SerperDevTool(n=2)

# Agent-1
senior_research_analyst = Agent(
    role = "Senior Research Analyst",
    goal = f"Research,Analyze and synthesize comprehensive information on {topic} from reliable web sources",
    backstory= "You are an expert research analyst with advanced web research skills.",
    verbose = True,
    allow_delegation=False,
    tools = [search_tool],
    llm = llm
)

content_writer = Agent(
    role = "content writer",
    goal = "Transform research findings into engaging blog posts while maintaining the accuracy",
    backstory="You are a skilled content writer specialized in creating."
                "engaging,accessible content from technical research",
    verbose=True,
    allow_delegation=False,
    llm = llm
)

#Research Task
research_task = Task(
    description = (
        """ 
        1. Conduct comprehensive research on {topic} including:
            - Research developments and news
            - key industry trends and innovations
            - Expert opinion and analysis
            - statistical data and market insights

        2. Evaluate source credibility and fact-check all information
        3. organize findings into a structured research brief
        4. Include all relevant citiations and sources
        """
    ),
    agent = senior_research_analyst,
    expected_output="""A detailed research report contains
                        - Executive  summary of key findings
                        - comprehensive analysis of current trends and developments
                    """
)

content_writer_task = Task(
    description = """ 
                Using the research brief provided create an engaging blog post that.
                1. Transforms technical information into accessible content
                2. Maintains all factual accuracy and citiations from research.
                """,
    agent = content_writer,
    expected_output = """ 
            A polished blog post in markdown format that
            - Engages readers while maintaining accuracy
            - contains properly structured sections
            - Includes inline citiations hyperlinked to the original source url
            """
)

crew =  Crew(
    agents = [senior_research_analyst,content_writer],
    tasks = [research_task,content_writer_task],
    verbose=True
)

crew.kickoff()