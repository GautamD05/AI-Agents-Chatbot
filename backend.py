#Step1: Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List


#defines in which format we should be receiveing the data
#It defines a schema for expected input data.
#When you instantiate RequestState, Pydantic will validate the data types and ensure the fields are provided correctly
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


#Step2: Setup AI Agent from FrontEnd Request

#Import FastAPI Python framework for building APIs quickly and efficiently
from fastapi import FastAPI
#import the get_response_from_ai_agent function from ai_agent.py that handles the logic for interacting with an AI model
from ai_agent import get_response_from_ai_agent

ALLOWED_MODEL_NAMES=["llama3-70b-8192", "deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile", "gpt-4o-mini","gpt-4.1"]

#Instantiates the FastAPI application and the title of the API, visible in the Swagger UI documentation
app=FastAPI(title="LangGraph AI Agent")

@app.post("/chat") #decorator to directly associating a route and HTTP method with the handler function(chat_endpoint)
def chat_endpoint(request: RequestState): 
    """
    Specifies that this is a POST endpoint accessible at the /chat route.
    RequestState: Indicates that the endpoint expects a RequestState object (defined using Pydantic) as input. 
    This ensures that the input data adheres to the defined schema.
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    
    #Extracts parameters from the request object
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # Calls the function which creates the AI Agent and gets response from it 
    response=get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    # Returns the response obtained from the AI agent back to the client.
    return response

#Step3: Run app & Explore Swagger UI Docs

"""
This block ensures the script runs as a standalone application.
uvicorn.run(): Launches the FastAPI application locally on 127.0.0.1 (localhost) at port 9999

FastAPI automatically generates interactive API documentation available at:
Swagger UI: Accessible at http://127.0.0.1:9999/docs
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
