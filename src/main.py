from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import (
    search_tool,
    wiki_tool,
    save_tool,
    save_to_txt,
    get_wikipedia_summary,
    get_topic_output_filename,
    wikipedia_error_message,
)

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str = Field(description="The main subject of the research")
    summary: str = Field(description="A comprehensive summary of the findings")
    sources: list[str] = Field(description="List of URLs or references used")
    tools_used: list[str] = Field(description="List of tools the agent decided to use")
    
_CACHED_AGENT_EXECUTOR: Optional[AgentExecutor] = None
_CACHED_PARSER: Optional[PydanticOutputParser] = None

def _get_agent_executor_and_parser() -> tuple[AgentExecutor, PydanticOutputParser]:
    global _CACHED_AGENT_EXECUTOR, _CACHED_PARSER
    if _CACHED_AGENT_EXECUTOR is not None and _CACHED_PARSER is not None:
        return _CACHED_AGENT_EXECUTOR, _CACHED_PARSER

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional research assistant. Use the available tools to find accurate information. "
                "After gathering information, you MUST provide your final answer strictly following this format: \n{format_instructions}",
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
        handle_parsing_errors=True,
    )

    _CACHED_AGENT_EXECUTOR = agent_executor
    _CACHED_PARSER = parser
    return agent_executor, parser

def _is_anthropic_credit_error(err: Exception) -> bool:
    message = str(err).lower()
    return (
        "anthropic api" in message
        and "credit" in message
        and ("too low" in message or "insufficient" in message)
    )

def _prompt_disambiguation_choice(query: str, options: list[tuple[str, str]]) -> Optional[str]:
    if not options:
        return None

    print("\nWikipedia returned multiple results for your query. Please select the most relevant one:")
    for idx, (title, url) in enumerate(options, start=1):
        suffix = f" ({url})" if url else ""
        print(f"{idx}. {title}{suffix}")
    while True:
        choice = input("Number (Enter to cancel): ").strip()
        if choice == "":
            return None
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(options):
                return options[selected - 1][0]
        print(f"Invalid choice. Please enter a number between 1 and {len(options)}, or press Enter to cancel.")

def _fallback_wikipedia(query: str) -> ResearchResponse:
    current_query = query
    while True:
        result = get_wikipedia_summary(current_query, options_limit=5)
        if not result["ok"]:
            return ResearchResponse(
                topic=current_query,
                summary=wikipedia_error_message(result),
                sources=[],
                tools_used=["wikipedia"],
            )

        if result["is_disambiguation"]:
            chosen = _prompt_disambiguation_choice(current_query, result["options"])
            if chosen is None:
                new_query = input(
                    "You chose to cancel the disambiguation selection. You can enter a new query to try again, or press Enter to exit: "
                ).strip()
                if not new_query:
                    return ResearchResponse(
                        topic=current_query,
                        summary="Consulta ambigua en Wikipedia. No se seleccionó ninguna opción.",
                        sources=[result["url"]] if result["url"] else [],
                        tools_used=["wikipedia"],
                    )
                current_query = new_query
                continue

            current_query = chosen
            continue

        return ResearchResponse(
            topic=result["title"] or current_query,
            summary=result["summary"],
            sources=[result["url"]] if result["url"] else [],
            tools_used=["wikipedia"],
        )

def _format_for_save(response: ResearchResponse, *, original_query: str, mode: str) -> str:
    sources = "\n".join(f"- {s}" for s in response.sources) if response.sources else "- (none)"
    tools_used = ", ".join(response.tools_used) if response.tools_used else "(none)"
    return (
        f"MODE: {mode}\n"
        f"ORIGINAL_QUERY: {original_query}\n\n"
        f"TOPIC: {response.topic}\n\n"
        f"SUMMARY:\n{response.summary}\n\n"
        f"SOURCES:\n{sources}\n\n"
        f"TOOLS_USED: {tools_used}\n"
    )

def run_research():
    query = input("What can I help you research? ")
    if not query.strip():
        print("Please provide a valid query.")
        return

    try:
        agent_executor, parser = _get_agent_executor_and_parser()
        raw_response = agent_executor.invoke({"query": query})
        output_text = raw_response.get("output")

        structured_response = parser.parse(output_text)
        
        print("\n--- RESEARCH RESULTS ---")
        print(f"TOPIC: {structured_response.topic}")
        print(f"SUMMARY: {structured_response.summary}")
        print(f"SOURCES: {', '.join(structured_response.sources)}")
        save_to_txt(
            _format_for_save(structured_response, original_query=query, mode="agent"),
            filename=get_topic_output_filename(structured_response.topic),
        )
        
    except Exception as e:
        if _is_anthropic_credit_error(e):
            structured_response = _fallback_wikipedia(query)
            print("\n--- RESEARCH RESULTS (FALLBACK: WIKIPEDIA) ---")
            print(f"TOPIC: {structured_response.topic}")
            print(f"SUMMARY: {structured_response.summary}")
            print(f"SOURCES: {', '.join(structured_response.sources)}")
            save_to_txt(
                _format_for_save(structured_response, original_query=query, mode="fallback"),
                filename=get_topic_output_filename(structured_response.topic),
            )
            return

        print(f"\n[Error] Could not parse structured response.")
        print(f"Raw Output: {raw_response.get('output') if 'raw_response' in locals() else 'No output'}")
        print(f"Details: {e}")

if __name__ == "__main__":
    run_research()
