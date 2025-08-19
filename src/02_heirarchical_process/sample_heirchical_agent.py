# hierarchical_compare.py
from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI
from crewai.tools import tool
import requests
import os

# --- Simple tool: pulls a short summary from Wikipedia REST API ---
@tool("wiki_summary")
def wiki_summary(topic: str) -> str:
    """Return a concise summary for a topic from Wikipedia."""
    
    if "fastapi" in topic.lower():
        return "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+."
    elif "flask" in topic.lower():
        return "Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries."
    else:
        return "No summary available for this topic."

# --- Worker agents ---
researcher = Agent(
    role="Technology Researcher",
    goal=( 
        "Find accurate, concise facts about a single technology. "
        "Prefer the wiki_summary tool; keep notes short (5–7 bullets)."
    ),
    backstory=(
        "You are precise and pragmatic. You gather only the most useful facts "
        "an engineer would care about (what it is, typical use, key strengths/limits)."
    ),
    tools=[wiki_summary],
    verbose=True,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

comparator = Agent(
    role="Technology Comparator",
    goal=(
        "Create a compact comparison and practical recommendation for a specific use case."
    ),
    backstory=(
        "You turn two sets of notes into a side-by-side comparison table and then pick "
        "one option with a brief rationale, focusing on developer experience and common trade-offs."
    ),
    tools=[],
    verbose=True,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

editor = Agent(
    role="Technical Editor",
    goal="Polish for clarity and brevity without changing facts.",
    backstory="You trim fluff, tighten language, and keep the output scannable.",
    tools=[],
    verbose=True,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# --- Single high-level task; manager will decompose into subtasks ---
top_level_task = Task(
    description=(
        "Compare {tech_a} vs {tech_b} for beginner-friendly web projects.\n\n"
        "Do this process (you may refine it):\n"
        "1) Research {tech_a}: gather 5–7 bullet facts using wiki_summary first.\n"
        "2) Research {tech_b}: gather 5–7 bullet facts using wiki_summary first.\n"
        "3) Build a concise comparison:\n"
        "   - A 5–7 row table (Criteria | {tech_a} | {tech_b})\n"
        "   - A short recommendation (2–4 sentences) with trade-offs.\n"
        "4) Let the Technical Editor polish the final draft."
    ),
    expected_output=(
        "A short report containing:\n"
        "- Bullet notes for each technology\n"
        "- A compact comparison table\n"
        "- A brief recommendation"
    ),
    # In hierarchical mode, the manager will route work across agents;
    # assigning a default agent is fine—manager still delegates.
    agent=researcher,
    verbose=True,
)

# --- Crew in hierarchical mode with a manager LLM ---
crew = Crew(
    agents=[researcher, comparator, editor],
    tasks=[top_level_task],
    process=Process.hierarchical,  # <-- manager decomposes & delegates
    manager_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    max_iter=3,
    verbose=True,
)

if __name__ == "__main__":
    # Example inputs—swap with any pair you want
    inputs = {"tech_a": "Flask", "tech_b": "FastAPI"}

    # Optional sanity checks
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set; set it to run this example.")

    result = crew.kickoff(inputs=inputs)
    print("\n--- Final Output ---")
    print(result.raw if hasattr(result, "raw") else result)
