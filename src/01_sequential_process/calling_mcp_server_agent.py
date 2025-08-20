import os, json, sys
from typing import List, Optional
import mcp
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from crewai.tools import BaseTool  # <-- FIX: import BaseTool from crewai.tools

# -----------------------------
# OpenAI model selection
# -----------------------------
def get_openai_api_key():
    print(os.getenv("OPENAI_API_KEY"))
    return os.getenv("OPENAI_API_KEY")

_ = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

# -----------------------------
# Agent definition
# -----------------------------
CRYPTO_AGENT_ROLE = "CryptoTradingAgent"

CRYPTO_AGENT_GOAL = (
    "Act as an orchestrator: first inspect which MCP tools are attached, then pick the most "
    "appropriate tool for the user's query. Capabilities include:\n"
    "• Fetch portfolio/balances/positions\n"
    "• List tradable Coinbase products\n"
    "• Execute trades (ONLY with explicit confirmation)\n"
    "• Run tool health checks before critical actions"
)

CRYPTO_AGENT_BACKSTORY = (
    "You are a careful trading assistant with strong guardrails. You never fabricate data; you call "
    "MCP tools to get facts or perform actions, then summarize results clearly."
)

# -----------------------------
# MCP (streamable-http) config
# -----------------------------
MCP_SERVER_PARAMS = {
    "transport": "streamable-http",
    "url": "http://127.0.0.1:9000/mcp",  # no trailing slash avoids double-// on some routers
    "headers": {
        "X-User-Id": "sample-user-123",
        "X-Scopes": "portfolio:read,portfolio:trade",
    },
}

# Restrict to these tool names from your server (optional; omit to expose all)
MCP_TOOL_NAMES = (
    "get_portfolio",
    "list_coinbase_products",
    "execute_crypto_trade",
    "tool_health",
)

# -----------------------------
# Custom introspection tool
# -----------------------------
class _NoArgs(BaseModel):
    """No input needed."""

class ListAttachedTools(BaseTool):
    """
    Lists the names/descriptions (and arg schema when available) of the currently
    attached tools (those provided by the MCPServerAdapter). Use this FIRST to plan.
    """
    name: str = "list_attached_tools"
    description: str = (
        "List the tools currently attached to the agent (from the MCP adapter). "
        "Call this FIRST to decide which tool to use next."
    )
    args_schema: type = _NoArgs

    def __init__(self, tools: List[BaseTool]):
        super().__init__()
        self._tools = tools

    def _run(self) -> str:
        items = []
        for t in self._tools:
            spec = {
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", "") or "",
            }
            args_schema = getattr(t, "args_schema", None)
            if args_schema:
                try:
                    # Pydantic v2
                    spec["args_schema"] = args_schema.model_json_schema()
                except Exception:
                    # Fallback / older pydantic
                    try:
                        spec["args_schema"] = args_schema.schema()
                    except Exception:
                        spec["args_schema"] = str(args_schema)
            items.append(spec)
        return json.dumps({"tools": items})

# -----------------------------
# Main
# -----------------------------
def main(user_query: Optional[str] = None):
    user_query = user_query or "Give me my portfolio?"

    with MCPServerAdapter(MCP_SERVER_PARAMS, *MCP_TOOL_NAMES, connect_timeout=60) as mcp_tools:
        # Attach the introspection tool + MCP tools
        list_tool = ListAttachedTools(list(mcp_tools))
        # toolset = [list_tool, *mcp_tools]
        toolset = [*mcp_tools]

        agent = Agent(
            role=CRYPTO_AGENT_ROLE,
            goal=CRYPTO_AGENT_GOAL,
            backstory=CRYPTO_AGENT_BACKSTORY,
            tools=toolset,
            verbose=True,
            reasoning=False,           # keep deterministic; avoids planning loops
            allow_delegation=False,
        )

        # Orchestrator behavior: 1) list tools 2) choose 3) call 4) summarize
        task = Task(
            description=(
                f'User query: "{user_query}"\n\n'
                "Follow this strict sequence:\n"
                "1) Call `list_attached_tools` ONCE to see what's available.\n"
                "2) Decide which tool best satisfies the query, using these rules:\n"
                "   • portfolio/balance/holdings → call `get_portfolio` (no args)\n"
                "   • products/markets/pairs → call `list_coinbase_products`\n"
                "   • buy/sell/execute → call `execute_crypto_trade` BUT FIRST confirm: "
                "     intent, symbol, side, quantity or quote_size, and price/slippage. "
                "     If missing, ask one clarifying question and STOP.\n"
                "   • health/status → call `tool_health`\n"
                "3) After the chosen tool returns, produce a concise summary for the user.\n"
                "   - If JSON came back, preserve key structure and show assets, quantities, values.\n"
                "   - State which tool you used.\n"
                "Never fabricate results; prefer tool outputs as ground truth."
            ),
            expected_output=(
                "A clear, final answer stating the chosen tool and the relevant data. "
                "For portfolio: per-asset breakdown with totals."
            ),
            agent=agent,
            markdown=True,
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        print("\n=== Final Result ===\n")
        print(result)

if __name__ == "__main__":
    # Allow passing the user query from CLI: python calling_mcp_server_agent.py "Give me my portfolio?"
    cli_query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    main(cli_query)
    