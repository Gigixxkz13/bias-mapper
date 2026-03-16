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


class BatchExperimentRequest(BaseModel):
    name: str
    bubble_type: str
    topic: str
    description: str
    prompt_A_text: str
    prompt_B_text: str
    model_name: str
    mode: str
    repetitions: int


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


# Batch experiment execution endpoint
# -----------------------------------
# This endpoint runs the same experiment multiple times automatically.
# It is useful for collecting repeated observations for the thesis.

@app.post("/run-batch")
def run_batch(request: BatchExperimentRequest):

    results = []

    for i in range(request.repetitions):

        # Store the prompt pair
        pair_id = insert_prompt_pair(
            request.name,
            request.bubble_type,
            request.topic,
            request.description,
            request.prompt_A_text,
            request.prompt_B_text
        )

        # Create a new run
        run_id = create_run(
            pair_id,
            request.model_name,
            request.mode
        )

        # Run Prompt A
        response_A = call_llm(request.prompt_A_text, request.model_name)
        result_A = process_and_store_response(
            run_id=run_id,
            identifier="A",
            prompt_text=request.prompt_A_text,
            raw_output_text=response_A
        )

        # Run Prompt B
        response_B = call_llm(request.prompt_B_text, request.model_name)
        result_B = process_and_store_response(
            run_id=run_id,
            identifier="B",
            prompt_text=request.prompt_B_text,
            raw_output_text=response_B
        )

        # Store the summary of this repetition
        results.append({
            "run_number": i + 1,
            "run_id": run_id,
            "responseA_metrics": result_A,
            "responseB_metrics": result_B
        })

    return {
        "status": "batch completed",
        "model_name": request.model_name,
        "bubble_type": request.bubble_type,
        "repetitions": request.repetitions,
        "results": results
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


# Compare Prompt A and Prompt B for a specific run
# ------------------------------------------------
# This endpoint returns a simplified comparison of the main metrics
# so that the results can be used more easily in the frontend and thesis.

@app.get("/compare-run/{run_id}")
def compare_run(run_id: int):

    # Retrieve the run information
    run = get_run_by_id(run_id)

    if run is None:
        return {"error": "Run not found"}

    # Retrieve the associated prompt pair if it exists
    prompt_pair = None
    if run["pair_id"] is not None:
        prompt_pair = get_prompt_pair_by_id(run["pair_id"])

    # Retrieve the two stored responses for this run
    responses = get_responses_by_run_id(run_id)

    response_A = None
    response_B = None

    # Separate Prompt A and Prompt B
    for response in responses:
        if response["identifier"] == "A":
            response_A = response
        elif response["identifier"] == "B":
            response_B = response

    # Make sure both responses exist
    if response_A is None or response_B is None:
        return {"error": "Both Prompt A and Prompt B responses are required"}

    # Return a simplified comparison summary
    return {
        "run_id": run["run_id"],
        "model_name": run["model_name"],
        "mode": run["mode"],
        "prompt_pair_name": prompt_pair["name"] if prompt_pair else None,
        "bubble_type": prompt_pair["bubble_type"] if prompt_pair else None,

        "promptA": {
            "sentimentScore": response_A["sentimentScore"],
            "diversityScore": response_A["diversityScore"],
            "riskCount": response_A["riskCount"],
            "benefitCount": response_A["benefitCount"],
            "emotionCount": response_A["emotionCount"]
        },

        "promptB": {
            "sentimentScore": response_B["sentimentScore"],
            "diversityScore": response_B["diversityScore"],
            "riskCount": response_B["riskCount"],
            "benefitCount": response_B["benefitCount"],
            "emotionCount": response_B["emotionCount"]
        },

        "differences": {
            "sentiment_difference": round((response_B["sentimentScore"] or 0) - (response_A["sentimentScore"] or 0), 3),
            "diversity_difference": round((response_B["diversityScore"] or 0) - (response_A["diversityScore"] or 0), 3),
            "risk_difference": (response_B["riskCount"] or 0) - (response_A["riskCount"] or 0),
            "benefit_difference": (response_B["benefitCount"] or 0) - (response_A["benefitCount"] or 0),
            "emotion_difference": (response_B["emotionCount"] or 0) - (response_A["emotionCount"] or 0)
        }
    }