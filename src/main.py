import os
import json
import socket
import urllib.parse
import urllib.request
import urllib.error
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, save_to_txt

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

def _is_anthropic_credit_error(err: Exception) -> bool:
    message = str(err).lower()
    return (
        "anthropic api" in message
        and "credit" in message
        and ("too low" in message or "insufficient" in message)
    )

def _is_offline_error(err: Exception) -> bool:
    if isinstance(err, (socket.gaierror, socket.timeout, TimeoutError)):
        return True

    if isinstance(err, urllib.error.URLError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, (socket.gaierror, socket.timeout, TimeoutError)):
            return True
        if isinstance(reason, OSError) and getattr(reason, "errno", None) in {8, -2}:
            return True

    message = str(err).lower()
    return any(
        phrase in message
        for phrase in (
            "nodename nor servname provided",
            "name or service not known",
            "temporary failure in name resolution",
            "no address associated with hostname",
            "max retries exceeded",
            "failed to establish a new connection",
        )
    )

def _prompt_disambiguation_choice(query: str, options: list[tuple[str, str]]) -> Optional[str]:
    if not options:
        return None

    print("\nWikipedia encontró una búsqueda ambigua. Elige una opción:")
    for idx, (title, url) in enumerate(options, start=1):
        suffix = f" ({url})" if url else ""
        print(f"{idx}. {title}{suffix}")

    while True:
        choice = input("Número (Enter para cancelar): ").strip()
        if choice == "":
            return None
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(options):
                return options[selected - 1][0]
        print(f"Opción inválida. Escribe un número entre 1 y {len(options)} o Enter para cancelar.")

def _fallback_wikipedia(query: str) -> ResearchResponse:
    try:
        title = query
        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title, safe="")
        )

        def fetch_json(url: str) -> dict:
            req = urllib.request.Request(url, headers={"User-Agent": "research-ai-agent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = resp.read().decode("utf-8")
            return json.loads(payload)

        data = fetch_json(summary_url)
        extract_lower = (data.get("extract") or "").lower()
        is_disambiguation = data.get("type") == "disambiguation" or "may refer to" in extract_lower
        if is_disambiguation:
            search_url = (
                "https://en.wikipedia.org/w/api.php?action=opensearch&limit=5&namespace=0&format=json&search="
                + urllib.parse.quote(query)
            )
            search_data = fetch_json(search_url)
            titles = (search_data[1] if len(search_data) > 1 else []) or []
            urls = (search_data[3] if len(search_data) > 3 else []) or []
            options = list(zip(titles, urls))

            chosen = _prompt_disambiguation_choice(query, options)
            if chosen is None:
                disambig_url = (
                    (data.get("content_urls") or {})
                    .get("desktop", {})
                    .get("page", "")
                )
                return ResearchResponse(
                    topic=query,
                    summary="Consulta ambigua en Wikipedia. No se seleccionó ninguna opción.",
                    sources=[disambig_url] if disambig_url else [],
                    tools_used=["wikipedia"],
                )

            data = fetch_json(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + urllib.parse.quote(chosen, safe="")
            )

        title = data.get("title") or query
        summary = data.get("extract") or "No summary available."
        url = (
            (data.get("content_urls") or {})
            .get("desktop", {})
            .get("page", "")
        )
        return ResearchResponse(topic=title, summary=summary, sources=[url] if url else [], tools_used=["wikipedia"])
    except Exception as e:
        if _is_offline_error(e):
            return ResearchResponse(
                topic=query,
                summary=(
                    "No connection to the internet. Unable to fetch information from Wikipedia. Please check your connection and try again."
                ),
                sources=[],
                tools_used=["wikipedia"],
            )
        return ResearchResponse(
            topic=query,
            summary="No connection to the internet. Unable to fetch information from Wikipedia. Please check your connection and try again.",
            sources=[],
            tools_used=["wikipedia"],
        )

def _format_for_save(response: ResearchResponse) -> str:
    sources = "\n".join(f"- {s}" for s in response.sources) if response.sources else "- (none)"
    tools_used = ", ".join(response.tools_used) if response.tools_used else "(none)"
    return (
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
        raw_response = agent_executor.invoke({"query": query})
        output_text = raw_response.get("output")

        structured_response = parser.parse(output_text)
        
        print("\n--- RESEARCH RESULTS ---")
        print(f"TOPIC: {structured_response.topic}")
        print(f"SUMMARY: {structured_response.summary}")
        print(f"SOURCES: {', '.join(structured_response.sources)}")
        
    except Exception as e:
        if _is_anthropic_credit_error(e):
            structured_response = _fallback_wikipedia(query)
            print("\n--- RESEARCH RESULTS (FALLBACK: WIKIPEDIA) ---")
            print(f"TOPIC: {structured_response.topic}")
            print(f"SUMMARY: {structured_response.summary}")
            print(f"SOURCES: {', '.join(structured_response.sources)}")
            save_to_txt(_format_for_save(structured_response))
            return

        print(f"\n[Error] Could not parse structured response.")
        print(f"Raw Output: {raw_response.get('output') if 'raw_response' in locals() else 'No output'}")
        print(f"Details: {e}")

if __name__ == "__main__":
    run_research()
