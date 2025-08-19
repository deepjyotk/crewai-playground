from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
manager_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def main():
    user_query = "My car makes a strange noise when driving, feels bumpy."

    # ---- Sub Agents ----
    diagnosis_agent = Agent(
        role="Car Diagnosis Agent",
        goal="Diagnose car problems based on user symptoms",
        backstory="Expert at figuring out car issues from brief descriptions",
        verbose=True,
        llm=llm
    )

    oil_specialist = Agent(
        role="Oil Specialist",
        goal="Give advice on oil-related car issues",
        backstory="Knows everything about oil changes and lubrication systems",
        verbose=True,
        llm=llm
    )

    tire_specialist = Agent(
        role="Tire Specialist",
        goal="Give advice on tire-related issues",
        backstory="Expert in tire maintenance and repairs",
        verbose=True,
        llm=llm
    )

    # ---- Tasks ----
    diagnosis_task = Task(
        description=f"User reports: '{user_query}'. Diagnose the car issue based on this report.",
        expected_output="A brief diagnosis identifying the likely issue type (oil, tire, suspension, brakes, or engine).",
        agent=diagnosis_agent
    )

    oil_task = Task(
        description="Based on the diagnosis, provide oil change and lubrication system maintenance advice if relevant.",
        expected_output="Oil-related maintenance tips or required service steps if applicable, otherwise state 'Not applicable'.",
        agent=oil_specialist
    )

    tire_task = Task(
        description="Based on the diagnosis, provide tire maintenance and repair advice if relevant.",
        expected_output="Tire-related maintenance tips or required service steps if applicable, otherwise state 'Not applicable'.",
        agent=tire_specialist
    )

    # ---- Crew ----
    crew = Crew(
        agents=[diagnosis_agent, oil_specialist, tire_specialist],
        tasks=[diagnosis_task, oil_task, tire_task],
        process=Process.hierarchical,
        manager_llm=manager_llm,
        verbose=True,
    )

    # ---- Run ----
    result = crew.kickoff()
    print("\n=== Final Recommendation ===\n", result)


if __name__ == "__main__":
    main()
