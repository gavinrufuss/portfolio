from models import ScenarioAnalysisRequest

def build_prompt(request: ScenarioAnalysisRequest) -> str:
    return (
        "Given the following scenario and constraints, generate:\n"
        "1. A brief summary of the scenario (1-2 sentences)\n"
        "2. A list of potential pitfalls\n"
        "3. A list of proposed strategies\n"
        "4. A list of recommended resources\n"
        "5. A one-sentence disclaimer\n\n"
        f"Scenario: {request.scenario}\n"
        f"Constraints: {', '.join(request.constraints)}"
    )