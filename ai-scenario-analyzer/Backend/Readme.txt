AI Scenario Analyzer - Instructions - PYTHON [USED MOCK]

-Backend (FastAPI)

1. Navigate to the backend folder.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:
   pip install -r requirements.txt

4. To run:
   uvicorn main:app --reload

5. Access Swagger UI at:
   http://localhost:8000/docs

6. Request: 
{ "scenario": "Our team has a new client project with a tight deadline and limited budget.",
"constraints": [
"Budget: $10,000",
"Deadline: 6 weeks",
"Team of 3 developers" ]}

Note: For real OpenAI API, add API key to .env file in the backend folder


