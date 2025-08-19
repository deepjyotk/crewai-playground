## Crew AI Sample

Run a two-agent research-and-write workflow using CrewAI and LangChain's OpenAI chat model with a DuckDuckGo search tool.

### Setup

1) Ensure Python 3.13+ is installed.

2) Install deps (already configured via `uv`):

```bash
uv sync
```

3) Configure environment:

Create a `.env` file (or export env vars) with:

```bash
OPENAI_API_KEY="sk-..."
```

### Run

Using the CLI entry point with a default topic:

```bash
uv run crewai-sample
```

Or pass a custom topic:

```bash
uv run crewai-sample --topic "Serverless observability best practices in 2025"
```

### Notes

- The app uses `gpt-4o-mini` by default. You can change the model via `--model`.
- The `DuckDuckGoSearchRun` tool performs simple web searches.

