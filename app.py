from __future__ import annotations

import json

from fastapi import FastAPI

from perishable_pricing_env.config import TASKS
from perishable_pricing_env.environment import PerishablePricingEnv
from perishable_pricing_env.models import ActionModel

app = FastAPI(title="Perishable Pricing OpenEnv")
env = PerishablePricingEnv(artifact_dir="artifacts", verbose=False)


@app.get("/")
def health() -> dict:
    return {"status": "ok", "tasks": [x["task_id"] for x in TASKS]}


@app.post("/reset/{task_id}")
def reset(task_id: str, seed: int = 42) -> dict:
    obs = env.reset(seed=seed, task_id=task_id)
    return obs.model_dump()


@app.post("/step")
def step(payload: dict) -> dict:
    action = ActionModel(**payload)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> str:
    return json.dumps(env.state())

