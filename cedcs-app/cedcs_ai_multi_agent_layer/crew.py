import os


from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from cedcs_ai_multi_agent_layer.llm_gateway import get_llm
from cedcs_ai_multi_agent_layer.tools.resource_snapshot_tool import ResourceSnapshotTool
from cedcs_ai_multi_agent_layer.tools.places_nearby_tool import PlacesNearbyTool
from cedcs_ai_multi_agent_layer.tools.facility_registry_lookup_tool import FacilityRegistryLookupTool





@CrewBase
class CedcsAiMultiAgentLayerCrew:
    """CedcsAiMultiAgentLayer crew"""

    
    @agent
    def emergency_intake_specialist(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["emergency_intake_specialist"],
            
            
            tools=[],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=get_llm(),
            
        )
        
    
    @agent
    def clarification_specialist(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["clarification_specialist"],
            
            
            tools=[],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=get_llm(),
            
        )
        
    
    @agent
    def emergency_triage_assessor(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["emergency_triage_assessor"],
            
            
            tools=[],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=get_llm(),
            
        )
        
    
    @agent
    def facility_discovery_coordinator(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["facility_discovery_coordinator"],
            
            
            tools=[				PlacesNearbyTool(),
				FacilityRegistryLookupTool()],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=get_llm(),
            
        )
        
    
    @agent
    def hospital_data_normaliser(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["hospital_data_normaliser"],
            
            
            tools=[ResourceSnapshotTool()],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=get_llm(),
            
        )
        
    
    @agent
    def recommendation_narrator(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["recommendation_narrator"],
            
            
            tools=[],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=get_llm(),
            
        )
        
    

    
    @task
    def structured_intake_task(self) -> Task:
        return Task(
            config=self.tasks_config["structured_intake_task"],
            markdown=False,
            
            
        )
    
    @task
    def clarification_task(self) -> Task:
        return Task(
            config=self.tasks_config["clarification_task"],
            markdown=False,
            
            
        )
    
    @task
    def triage_assessment_task(self) -> Task:
        return Task(
            config=self.tasks_config["triage_assessment_task"],
            markdown=False,
            
            
        )
    
    @task
    def facility_discovery_task(self) -> Task:
        return Task(
            config=self.tasks_config["facility_discovery_task"],
            markdown=False,
            
            
        )
    
    @task
    def resource_interpretation_task(self) -> Task:
        return Task(
            config=self.tasks_config["resource_interpretation_task"],
            markdown=False,
            
            
        )
    
    @task
    def explanation_task(self) -> Task:
        return Task(
            config=self.tasks_config["explanation_task"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the CedcsAiMultiAgentLayer crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,

            chat_llm=get_llm(),
        )


