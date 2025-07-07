from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.google import GeminiEmbedder
from phi.model.groq import Groq
from phi.model.google import Gemini
import os
from dotenv import load_dotenv
load_dotenv()

from phi.agent import AgentKnowledge
from phi.embedder.fastembed import FastEmbedEmbedder
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.embedder.google import GeminiEmbedder
import google.generativeai as genai

my_api_key = "AIzaSyBZDOH1shRYFMdzmqaKAF7mtNHNtSW9RpM"
genai.configure(api_key=my_api_key)

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes", db_url=db_url, search_type=SearchType.hybrid, embedder=GeminiEmbedder()),
)
# Load the knowledge base: Comment out after first run
knowledge_base.load(recreate=True, upsert=True)

agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    knowledge=knowledge_base,
    # Add a tool to read chat history.
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True,
    # debug_mode=True,
)
agent.print_response("Hey what is Pad Thai Goong Sod", stream=True)
agent.print_response("What was my last question?", stream=True)