import os
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from langchain.tools import Tool 
from langchain_community.tools import DuckDuckGoSearchRun 
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

def _safe_filename(name: str, *, default: str = "research_output") -> str:
    name = (name or "").strip()
    if not name:
        return default

    name = name.lower()
    name = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name).strip("_")
    name = re.sub(r"_+", "_", name)
    return name or default

def get_topic_output_filename(topic: str) -> str:
    return f"{_safe_filename(topic, default='topic')}.txt"

def save_to_txt(data: str, filename: Optional[str] = None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(data_dir, exist_ok=True)
    if filename:
        target_name = os.path.basename(filename)
        target_name = _safe_filename(os.path.splitext(target_name)[0]) + os.path.splitext(target_name)[1]
    else:
        target_name = "research_output.txt"
    filename = os.path.join(data_dir, target_name)
    
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

def _fetch_json(url: str) -> Dict:
    req = urllib.request.Request(url, headers={"User-Agent": "research-ai-agent/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)

def _wikipedia_base_urls(lang: Optional[str] = None) -> Tuple[str, str]:
    chosen_lang = (lang or os.getenv("WIKIPEDIA_LANG") or "en").strip().lower() or "en"
    host = f"{chosen_lang}.wikipedia.org"
    return (
        f"https://{host}/api/rest_v1/page/summary/",
        f"https://{host}/w/api.php",
    )

def get_wikipedia_summary(query: str, *, options_limit: int = 5, lang: Optional[str] = None) -> Dict:
    try:
        rest_base, api_base = _wikipedia_base_urls(lang)
        summary_url = rest_base + urllib.parse.quote(query, safe="")

        try:
            data = _fetch_json(summary_url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                search_url = (
                    api_base
                    + "?action=opensearch&limit="
                    + str(options_limit)
                    + "&namespace=0&format=json&search="
                    + urllib.parse.quote(query)
                )
                search_data = _fetch_json(search_url)
                titles = (search_data[1] if len(search_data) > 1 else []) or []
                urls = (search_data[3] if len(search_data) > 3 else []) or []
                options = list(zip(titles, urls))
                if not titles:
                    return {
                        "ok": False,
                        "title": query,
                        "summary": "",
                        "url": "",
                        "is_disambiguation": False,
                        "options": [],
                        "error": "not_found",
                        "offline": False,
                        "not_found": True,
                        "lang": (lang or os.getenv("WIKIPEDIA_LANG") or "en"),
                    }

                best = titles[0]
                data = _fetch_json(rest_base + urllib.parse.quote(best, safe=""))
                extract = data.get("extract") or ""
                extract_lower = extract.lower()
                is_disambiguation = data.get("type") == "disambiguation" or "may refer to" in extract_lower
                url = (
                    (data.get("content_urls") or {})
                    .get("desktop", {})
                    .get("page", "")
                )
                return {
                    "ok": True,
                    "title": data.get("title") or best,
                    "summary": extract or "No summary available.",
                    "url": url,
                    "is_disambiguation": is_disambiguation,
                    "options": options if is_disambiguation else [],
                    "error": "",
                    "offline": False,
                    "not_found": False,
                    "lang": (lang or os.getenv("WIKIPEDIA_LANG") or "en"),
                }
            raise

        extract = data.get("extract") or ""
        extract_lower = extract.lower()
        is_disambiguation = data.get("type") == "disambiguation" or "may refer to" in extract_lower

        url = (
            (data.get("content_urls") or {})
            .get("desktop", {})
            .get("page", "")
        )

        options: List[Tuple[str, str]] = []
        if is_disambiguation:
            search_url = (
                api_base
                + "?action=opensearch&limit="
                + str(options_limit)
                + "&namespace=0&format=json&search="
                + urllib.parse.quote(query)
            )
            search_data = _fetch_json(search_url)
            titles = (search_data[1] if len(search_data) > 1 else []) or []
            urls = (search_data[3] if len(search_data) > 3 else []) or []
            options = list(zip(titles, urls))

        return {
            "ok": True,
            "title": data.get("title") or query,
            "summary": extract or "No summary available.",
            "url": url,
            "is_disambiguation": is_disambiguation,
            "options": options,
            "error": "",
            "offline": False,
            "not_found": False,
            "lang": (lang or os.getenv("WIKIPEDIA_LANG") or "en"),
        }
    except Exception as e:
        offline = _is_offline_error(e)
        return {
            "ok": False,
            "title": query,
            "summary": "",
            "url": "",
            "is_disambiguation": False,
            "options": [],
            "error": str(e),
            "offline": offline,
            "not_found": False,
            "lang": (lang or os.getenv("WIKIPEDIA_LANG") or "en"),
        }

def wikipedia_error_message(result: Dict) -> str:
    if result.get("offline"):
        return "Sin conexión a Internet (o fallo de DNS). No pude consultar Wikipedia."
    if result.get("not_found"):
        return "No encontré resultados en Wikipedia. Prueba con una consulta más específica."
    return "No pude consultar Wikipedia en este momento."

def _wikipedia_tool(query: str) -> str:
    result = get_wikipedia_summary(query)
    if not result["ok"]:
        return wikipedia_error_message(result)

    if result["is_disambiguation"]:
        lines = [
            f"'{query}' is ambiguous. Possible options include:"
        ]
        for title, url in result["options"][:5]:
            lines.append(f"- {title} ({url})" if url else f"- {title}")
        return "\n".join(lines)

    url = result["url"]
    header = f"{result['title']}\n"
    source = f"\nSource: {url}" if url else ""
    return f"{header}{result['summary']}{source}"

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

wiki_tool = Tool(
    name="wikipedia",
    func=_wikipedia_tool,
    description="Lookup a Wikipedia article summary. If ambiguous, refine the query (e.g. add '(musician)').",
)
