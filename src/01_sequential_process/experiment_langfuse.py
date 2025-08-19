# sequential_crewai_langfuse.py
from __future__ import annotations
import os

from crewai import Agent, Task, Crew, Process
# Langfuse (v3) — OTel-based client + helpers
from langfuse import get_client
# OpenInference instrumentors (wire CrewAI & LiteLLM spans to Langfuse exporter)
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

# Optional: if you also make raw OpenAI calls elsewhere, use Langfuse's drop-in wrapper:
# from langfuse.openai import OpenAI
# openai_client = OpenAI()  # auto-traced; no extra work needed

def build_sequential_crew(topic: str) -> Crew:
    """A tiny sequential pipeline: Research → Write → Review."""
    researcher = Agent(
        role="Researcher",
        goal=f"Find 3 recent, credible facts about '{topic}'.",
        backstory="Senior analyst skilled at fast, reliable web research.",
        # You can also pass model config via env or CrewAI config files
    )
    writer = Agent(
        role="Writer",
        goal="Turn notes into a crisp, well-structured summary (120-180 words).",
        backstory="Clear communicator who values accuracy and brevity.",
    )
    reviewer = Agent(
        role="Reviewer",
        goal="Check the summary for correctness, duplication, and clarity; suggest tight edits.",
        backstory="Editorial reviewer with an eye for factual consistency.",
    )

    research = Task(
        description=(
            "Gather 3 recent, credible facts about the topic. "
            "Return bullet points with source names (no links needed)."
        ),
        expected_output="3 bullet points with sources.",
        agent=researcher,
    )

    draft = Task(
        description=(
            "Using the research bullets, write a 120–180 word summary with a 1-line headline."
        ),
        expected_output="Headline + concise summary.",
        agent=writer,
    )

    review = Task(
        description=(
            "Review the draft. If issues exist, rewrite the summary concisely; "
            "otherwise approve with a one-line note."
        ),
        # expected_output="Either: (Approved + one-line note) OR (Rewritten summary).",
        expected_output="Approved + one-line note",
        agent=reviewer,
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research, draft, review],
        process=Process.sequential,  # ensures ordered, linear execution
        verbose=True,
    )
    return crew

def main(topic: str = "Langfuse + CrewAI"):
    # 1) Init Langfuse client (sets up the global OTel exporter under the hood)
    lf = get_client()
    ok = lf.auth_check()
    if not ok:
        raise RuntimeError("Langfuse auth failed. Check LANGFUSE_* env and host.")

    # 2) Instrument CrewAI & LiteLLM (captures agent/task + LLM spans automatically)
    CrewAIInstrumentor().instrument(skip_dep_check=True)
    LiteLLMInstrumentor().instrument()

    # 3) Build crew
    crew = build_sequential_crew(topic)

    # 4) Create a root trace/span to tie the whole run together
    #    You can enrich it with user/session/tags/metadata and later add scores.
    with lf.start_as_current_span(name="crewai-sequential-demo") as span:
        result = crew.kickoff()
        print("\n=== FINAL OUTPUT ===\n")
        print(result)

        # 5) Enrich the trace (input/output + attributes you care about)
        span.update_trace(
            input={"topic": topic},
            output=str(result),
            user_id="user_123",
            session_id="sess_abc",
            tags=["demo", "sequential", "crewai"],
            metadata={"environment": os.getenv("ENV", "dev")},
            version="v0.1.0",
        )

        # 6) Optional: add custom scores (numeric/categorical) for QA or human feedback
        span.score(name="completeness", value=0.9, data_type="NUMERIC")
        span.score_trace(name="feedback", value="positive", data_type="CATEGORICAL")

    # 7) Flush immediately for short-lived scripts (so traces appear right away)
    lf.flush()

if __name__ == "__main__":
    main()
