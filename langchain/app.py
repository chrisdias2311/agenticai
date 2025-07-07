# Resources
# https://python.langchain.com/docs/integrations/chat/google_generative_ai/

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "default"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
) 
output_parser = StrOutputParser()

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         ("human", "{input}"),
#     ]
# )
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

# Streamlit frame work 
st.title("LangChain with Google Generative AI")
input_text = st.text_input("Search the topic you want")

chain = prompt | llm | output_parser
if input_text:
    st.write(chain.invoke({'question': input_text}))