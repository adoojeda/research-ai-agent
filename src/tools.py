import os
from datetime import datetime
from langchain.tools import Tool 
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun 
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper

def save_to_txt(data: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, "research_output.txt")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"Data successfully saved to {filename}"
    except Exception as e:
        return f"Error saving data: {e}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves the final research summary to a local file. Input should be the text to save."
)

try:
    ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    search_tool = DuckDuckGoSearchRun(api_wrapper=ddg_wrapper)
    search_tool.name = "search"
except ImportError:
    def _missing_ddgs(_: str) -> str:
        return "DuckDuckGo search is unavailable. Install dependency: `pip install -U ddgs`."

    search_tool = Tool(
        name="search",
        func=_missing_ddgs,
        description="Search the web using DuckDuckGo (requires the `ddgs` package).",
    )

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
wiki_tool.name = "wikipedia"
