from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perishable_pricing_env.config import PRODUCTS, TASKS
from perishable_pricing_env.environment import PerishablePricingEnv
from perishable_pricing_env.models import ActionModel

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("BENCHMARK", "perishable_pricing_openenv")
MAX_STEPS = int(os.getenv("MAX_STEPS", "14"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.6"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def clamp_action(prices: Dict[str, float]) -> Dict[str, float]:
    return {
        "milk": min(max(float(prices["milk"]), 10.0), 120.0),
        "banana": min(max(float(prices["banana"]), 5.0), 80.0),
        "bread": min(max(float(prices["bread"]), 5.0), 90.0),
    }


def heuristic_prices(observation: Dict[str, object]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for sku, cfg in PRODUCTS.items():
        stock = int(observation["inventory_units"][sku])
        base = float(cfg.ref_price)
        if stock < int(cfg.base_demand * 0.7):
            prices[sku] = base * 1.08
        elif stock > int(cfg.base_demand * 1.2):
            prices[sku] = base * 0.93
        else:
            prices[sku] = base
    return clamp_action(prices)


def model_prices(client: OpenAI, observation: Dict[str, object]) -> Dict[str, float]:
    prompt = (
        "You are pricing milk, banana, bread for a perishable dark-store simulator. "
        "Output strict JSON only with keys price_milk, price_banana, price_bread.\n"
        f"Observation: {json.dumps(observation)}"
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only JSON with numeric prices."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=120,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    parsed = json.loads(text)
    return clamp_action(
        {
            "milk": float(parsed["price_milk"]),
            "banana": float(parsed["price_banana"]),
            "bread": float(parsed["price_bread"]),
        }
    )


def run_task(task_id: str, client: OpenAI, seed: int) -> Dict[str, float]:
    env = PerishablePricingEnv(artifact_dir="artifacts", verbose=False)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = env.reset(seed=seed, task_id=task_id).model_dump()
        done = False
        for step in range(1, MAX_STEPS + 1):
            if done:
                break
            try:
                prices = model_prices(client, obs)
            except Exception:
                prices = heuristic_prices(obs)
            action = ActionModel(
                price_milk=prices["milk"],
                price_banana=prices["banana"],
                price_bread=prices["bread"],
            )
            action_str = (
                f"set_prices(milk={prices['milk']:.2f},banana={prices['banana']:.2f},bread={prices['bread']:.2f})"
            )
            obs_model, reward, done, info = env.step(action)
            obs = obs_model.model_dump()
            rewards.append(float(reward))
            steps_taken = step
            log_step(step=step, action=action_str, reward=float(reward), done=bool(done), error=None)
            if done:
                task_score = float((info.get("episode_kpis") or {}).get("task_score", 0.0))
                score = max(0.0, min(1.0, task_score))
                success = score >= SUCCESS_SCORE_THRESHOLD
                break
        if not rewards:
            score = 0.0
            success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "steps": steps_taken, "success": success}


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "missing-token")
    seed = int(os.getenv("SEED", "42"))
    results = []
    for task in TASKS:
        results.append(run_task(task_id=str(task["task_id"]), client=client, seed=seed))
    out = Path("artifacts/inference_scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

