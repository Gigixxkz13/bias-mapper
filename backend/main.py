# Georgia Kazara
# Reg. No. 20222216
# Thesis Bias Mapper: main.py
# ----------------------------

from fastapi import FastAPI
from pydantic import BaseModel
from backend.database import (
    initialize_database,
    insert_prompt_pair,
    create_run,
    get_all_runs,
    get_run_by_id,
    get_prompt_pair_by_id,
    get_responses_by_run_id,
    get_list_items_by_response_id
)
from backend.response_processor import process_and_store_response
from backend.llm import call_llm


# FastAPI application initialization
# ----------------------------------
app = FastAPI()

# Initialize the SQLite database when the server starts
initialize_database()


# Request models
# --------------
# These classes define the structure of incoming JSON requests

class ExperimentRequest(BaseModel):
    name: str
    bubble_type: str
    topic: str
    description: str
    prompt_A_text: str
    prompt_B_text: str
    model_name: str
    mode: str


class TestResponseRequest(BaseModel):
    run_id: int
    identifier: str
    prompt_text: str
    raw_output_text: str


# Root endpoint
# -------------
# Simple test endpoint to verify the API is running

@app.get("/")
def root():
    return {"message": "Bias Mapper API is running"}


# Experiment execution endpoint
# -----------------------------
# This endpoint runs a full experiment using a prompt pair.
# It sends both prompts to the selected LLM, processes the responses,
# calculates the analysis metrics, and stores everything in the database.

@app.post("/run-experiment")
def run_experiment(request: ExperimentRequest):

    # Store the prompt pair in the database
    pair_id = insert_prompt_pair(
        request.name,
        request.bubble_type,
        request.topic,
        request.description,
        request.prompt_A_text,
        request.prompt_B_text
    )

    # Create a new experiment run
    run_id = create_run(
        pair_id,
        request.model_name,
        request.mode
    )

    # Send Prompt A to the selected LLM
    response_A = call_llm(request.prompt_A_text, request.model_name)

    # Process and store the response along with analysis metrics
    result_A = process_and_store_response(
        run_id=run_id,
        identifier="A",
        prompt_text=request.prompt_A_text,
        raw_output_text=response_A
    )

    # Send Prompt B to the selected LLM
    response_B = call_llm(request.prompt_B_text, request.model_name)

    # Process and store the second response
    result_B = process_and_store_response(
        run_id=run_id,
        identifier="B",
        prompt_text=request.prompt_B_text,
        raw_output_text=response_B
    )

    # Return experiment results and calculated metrics
    return {
        "status": "experiment executed",
        "run_id": run_id,
        "responseA_metrics": result_A,
        "responseB_metrics": result_B
    }


# Manual response testing endpoint
# --------------------------------
# This endpoint was used during development to test the analysis pipeline
# without calling an external LLM.

@app.post("/test-response")
def test_response(request: TestResponseRequest):

    result = process_and_store_response(
        run_id=request.run_id,
        identifier=request.identifier,
        prompt_text=request.prompt_text,
        raw_output_text=request.raw_output_text
    )

    return {
        "status": "response processed",
        "result": result
    }


# Retrieve all experiment runs
# ----------------------------
# Returns a list of previously executed runs stored in the database

@app.get("/runs")
def list_runs():
    return get_all_runs()


# Retrieve detailed run information
# ---------------------------------
# Returns the full details of a specific run, including:
# - prompt pair configuration
# - raw model responses
# - calculated metrics
# - extracted list items

@app.get("/run/{run_id}")
def get_run_details(run_id: int):

    run = get_run_by_id(run_id)

    if run is None:
        return {"error": "Run not found"}

    # Retrieve the associated prompt pair if it exists
    prompt_pair = None
    if run["pair_id"] is not None:
        prompt_pair = get_prompt_pair_by_id(run["pair_id"])

    # Retrieve all responses for the run
    responses = get_responses_by_run_id(run_id)

    # Attach extracted list items to each response
    for response in responses:
        response["list_items"] = get_list_items_by_response_id(response["response_id"])

    return {
        "run": run,
        "prompt_pair": prompt_pair,
        "responses": responses
    }