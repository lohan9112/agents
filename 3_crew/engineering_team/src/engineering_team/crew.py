from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class EngineeringTeam():
    """EngineeringTeam crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # --- Engineering Lead: chỉ design, không chạy code ---
    @agent
    def engineering_lead(self) -> Agent:
        return Agent(
            config=self.agents_config['engineering_lead'],
            verbose=True,
            allow_code_execution=False   # chỉ design
        )

    # --- Backend Engineer: viết + chạy code bằng Docker ---
    @agent
    def backend_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['backend_engineer'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",   # Docker sandbox
            max_execution_time=300,
            max_retry_limit=3
        )

    # --- Frontend Engineer: chỉ viết UI ---
    @agent
    def frontend_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['frontend_engineer'],
            verbose=True,
            allow_code_execution=False   # chỉ generate code
        )

    # --- Test Engineer: viết test + chạy pytest bằng Docker ---
    # @agent
    # def test_engineer(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['test_engineer'],
    #         verbose=True,
    #         allow_code_execution=True,
    #         code_execution_mode="safe",   # Docker sandbox
    #         max_execution_time=300,
    #         max_retry_limit=2
    #     )

    # ---------------- TASKS ----------------

    @task
    def design_task(self) -> Task:
        return Task(
            config=self.tasks_config['design_task']
        )

    @task
    def code_task(self) -> Task:
        return Task(
            config=self.tasks_config['code_task']
        )

    @task
    def frontend_task(self) -> Task:
        return Task(
            config=self.tasks_config['frontend_task']
        )

    # @task
    # def test_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['test_task']
    #     )

    # ---------------- CREW ----------------

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
