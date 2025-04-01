from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ScenarioAnalysisRequest, ScenarioAnalysisResponse
from prompt_builder import build_prompt
from ai_client import call_openai_api 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze-scenario", response_model=ScenarioAnalysisResponse)
async def analyze_scenario(request: ScenarioAnalysisRequest):
    try:
        prompt = build_prompt(request)
        ai_response = call_openai_api(prompt)  # Used Mock
        return ai_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
