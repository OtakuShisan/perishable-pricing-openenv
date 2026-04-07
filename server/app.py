import uvicorn
from fastapi import FastAPI

from perishable_pricing_env.config import TASKS
from perishable_pricing_env.models import ActionModel
from server.environment import Environment

app = FastAPI(title="Perishable Pricing OpenEnv")
env = Environment(artifact_dir="artifacts", verbose=False)
DEFAULT_TASK_ID = str(TASKS[0]["task_id"])


@app.get("/")
def health() -> dict:
    return {"status": "ok", "tasks": [x["task_id"] for x in TASKS]}


@app.post("/reset")
def reset(payload: dict | None = None) -> dict:
    payload = payload or {}
    seed = int(payload.get("seed", 42))
    task_id = str(payload.get("task_id", DEFAULT_TASK_ID))
    obs = env.reset(seed=seed, task_id=task_id)
    return obs.model_dump()


@app.post("/reset/{task_id}")
def reset_task(task_id: str, seed: int = 42) -> dict:
    obs = env.reset(seed=seed, task_id=task_id)
    return obs.model_dump()


@app.post("/step")
def step(payload: dict) -> dict:
    action = ActionModel(**payload)
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state")
def state() -> dict:
    return env.state()


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

