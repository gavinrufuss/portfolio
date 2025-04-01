from pydantic import BaseModel
from typing import List

class ScenarioAnalysisRequest(BaseModel):
    scenario: str
    constraints: List[str]

class ScenarioAnalysisResponse(BaseModel):
    scenarioSummary: str
    potentialPitfalls: List[str]
    proposedStrategies: List[str]
    recommendedResources: List[str]
    disclaimer: str