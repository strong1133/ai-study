import dotenv
dotenv.load_dotenv()

from crewai import Crew, Agent, Task
from crewai.project import CrewBase, agent, task, crew




@CrewBase
class TranslatorCrew:
    

    @agent
    def translator_agent(self):
        return Agent(
            config = self.agents_config["translator_agent"]
        )

    @task
    def translate_task(self):
        return Task(
            config= self.tasks_config["translate_task"]
        )

    @crew
    def assemble_crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True
        )


TranslatorCrew().assemble_crew().kickoff(
    inputs={
        "sentence":"Hello, I'm SeokJin, my job is developer and I like ai coding "
    }
) 