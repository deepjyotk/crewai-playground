from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI
from crewai.tools import tool
import requests

# --- Tool ---
@tool("wiki_summary")
def wiki_summary(topic: str) -> str:
    """Fetch a concise summary for a topic from Wikipedia."""
    try:
        title = topic.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return f"Couldn't fetch summary (status {resp.status_code})."
        data = resp.json()
        return data.get("extract") or data.get("description") or "No summary found."
    except Exception as e:
        return f"Error fetching summary: {e}"

# --- Agents ---
research_agent = Agent(
    role="Research Agent",
    goal="Summarize a topic accurately using trusted sources.",
    backstory="You are great at finding and summarizing topics. Prefer the wiki_summary tool when possible.",
    tools=[wiki_summary],
    verbose=True,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

writer_agent = Agent(
    role="Creative Writer",
    goal="Rewrite summaries into a fun, engaging style.",
    backstory="You make any piece of text interesting and easy to read.",
    tools=[],
    verbose=True,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
)

# --- Tasks ---
research_task = Task(
    description="Research about {topic} and provide a short, accurate summary. Use `wiki_summary` tool if possible.",
    agent=research_agent,
    expected_output="A factual summary (3–5 sentences).",
    output_key="previous_summary",  # Store output in this key
    verbose=True
)

rewrite_task = Task(
    description="Rewrite the following summary in a fun, engaging tone while keeping the facts intact:\n\n{previous_summary}",
    agent=writer_agent,
    expected_output="A fun, engaging rewritten summary.",
    verbose=True
)

# --- Crew ---
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, rewrite_task],
    process=Process.sequential,  # passes outputs via keys
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": "Moon landing"})
    print("\n--- Final Output ---")
    print(result.raw if hasattr(result, "raw") else result)
