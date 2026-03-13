from fastapi import FastAPI
from pydantic import BaseModel
from backend.database import initialize_database, insert_prompt_pair, create_run
from backend.response_processor import process_and_store_response
from backend.llm import call_openai

app = FastAPI()

initialize_database()


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


@app.get("/")
def root():
    return {"message": "Bias Mapper API is running"}


@app.post("/run-experiment")
def run_experiment(request: ExperimentRequest):
    pair_id = insert_prompt_pair(
        request.name,
        request.bubble_type,
        request.topic,
        request.description,
        request.prompt_A_text,
        request.prompt_B_text
    )

    run_id = create_run(
        pair_id,
        request.model_name,
        request.mode
    )

    response_A = call_openai(request.prompt_A_text)

    result_A = process_and_store_response(
        run_id=run_id,
        identifier="A",
        prompt_text=request.prompt_A_text,
        raw_output_text=response_A
    )

    response_B = call_openai(request.prompt_B_text)

    result_B = process_and_store_response(
        run_id=run_id,
        identifier="B",
        prompt_text=request.prompt_B_text,
        raw_output_text=response_B
    )

    return {
        "status": "experiment executed",
        "run_id": run_id,
        "responseA_metrics": result_A,
        "responseB_metrics": result_B
    }


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