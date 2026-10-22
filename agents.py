from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tools import websearch, webscrape

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

def build_search_agent():
    return create_agent(
        model = llm,
        tools = [websearch]
    )

def build_scrape_agent():
    return create_agent(
        model = llm,
        tools = [webscrape]
    )

write_search_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, well-structured, and insightful report."),
    ("human", """Write a detailed report on the topic below
    topic: {topic}
    research gathered: {research}
    Structure the report as:
    -Introduction
    -Key Findings (minimum 3 well explained points)
    -Conclusion
    -References (all urls found in research)
    Be detailed, factual, and profressional.
    """)
])

writer_chain = write_search_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()