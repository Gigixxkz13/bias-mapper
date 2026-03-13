from fastapi import FastAPI
from pydantic import BaseModel
from backend.database import initialize_database, insert_prompt_pair, create_run

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

    return {
        "status": "experiment stored",
        "pair_id": pair_id,
        "run_id": run_id
    }