from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
import openai
from dotenv import load_dotenv
load_dotenv()

from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for financial information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    verbose=True,
    instructions=["Hey always include the sources"],
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    verbose=True,
    instructions=["Use tables to display the data"],
    markdown=True,
    show_tools_calls=True,
)

multi_ai_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team = [web_search_agent, finance_agent],
    instructions=["Always include the sources", "Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,
)

app=Playground(agents=[multi_ai_agent]).get_app()

if(__name__=="__main__"):
    serve_playground_app("playground:app", reload=True)