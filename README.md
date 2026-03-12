# Research AI Agent

Command-line research agent built with LangChain. It uses Anthropic (Claude) when available and automatically falls back to Wikipedia when the Anthropic API cannot be used (e.g., insufficient credits).

## Features

- **Agent mode (Anthropic + tools)**: the model can decide whether to use `search`, `wikipedia`, and `save_text_to_file`.
- **Wikipedia fallback**: if Anthropic fails due to insufficient credits, it fetches a Wikipedia summary without using an LLM.
- **Interactive disambiguation**: if the query is ambiguous, it prompts you to select an option or type a more specific query.
- **Per-topic autosave**: saves results to `data/<topic>.txt` and includes `MODE` and `ORIGINAL_QUERY`.

## Requirements

- Python 3.9+ (recommended: virtual environment)

## Installation

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Configuration (.env)

This project loads environment variables from `.env` using `python-dotenv` (optional).

Recommended variables:

- `ANTHROPIC_API_KEY`: required for Anthropic agent mode.
- `WIKIPEDIA_LANG`: Wikipedia language for the fallback and the `wikipedia` tool (default: `en`). .

## Usage

Run the CLI:

```bash
./venv/bin/python src/main.py
```

Then type a query when prompted.

### Saved output

Results are saved under `data/`:

- Autosave: `data/<topic>.txt` (for example `data/donald_trump.txt`).
- If the agent uses the `save_text_to_file` tool, it may also write to `data/research_output.txt` (a “generic” file).

## Tests

Run tests:

```bash
./venv/bin/python -m unittest discover -s tests -p "test*.py"
```

## Notes / troubleshooting

- If you have no Internet connection (or DNS issues), Wikipedia cannot be queried and the app will show a clear message.
- If the query doesn’t exist, the fallback attempts suggestions and reports when no results are found.
