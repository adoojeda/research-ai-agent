import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str = Field(description="The main subject of the research")
    summary: str = Field(description="A comprehensive summary of the findings")
    sources: list[str] = Field(description="List of URLs or references used")
    tools_used: list[str] = Field(description="List of tools the agent decided to use")
    
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0) 
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional research assistant. Use the available tools to find accurate information. "
            "After gathering information, you MUST provide your final answer strictly following this format: \n{format_instructions}"
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True 
)

def run_research():
    query = input("What can I help you research? ")
    if not query.strip():
        print("Please provide a valid query.")
        return

    try:
        raw_response = agent_executor.invoke({"query": query})
        output_text = raw_response.get("output")

        structured_response = parser.parse(output_text)
        
        print("\n--- RESEARCH RESULTS ---")
        print(f"TOPIC: {structured_response.topic}")
        print(f"SUMMARY: {structured_response.summary}")
        print(f"SOURCES: {', '.join(structured_response.sources)}")
        
    except Exception as e:
        print(f"\n[Error] Could not parse structured response.")
        print(f"Raw Output: {raw_response.get('output') if 'raw_response' in locals() else 'No output'}")
        print(f"Details: {e}")

if __name__ == "__main__":
    run_research()