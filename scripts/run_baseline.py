from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perishable_pricing_env.config import PRODUCTS, TASKS
from perishable_pricing_env.environment import PerishablePricingEnv
from perishable_pricing_env.models import ActionModel


SEEDS = [11, 22, 33]


def heuristic_prices(observation: Dict[str, object]) -> Dict[str, float]:
    prices = {}
    for sku, cfg in PRODUCTS.items():
        stock = observation["inventory_units"][sku]
        base = cfg.ref_price
        if stock < int(cfg.base_demand * 0.7):
            prices[sku] = base * 1.08
        elif stock > int(cfg.base_demand * 1.2):
            prices[sku] = base * 0.93
        else:
            prices[sku] = base
    return prices


def ask_model_for_prices(client: OpenAI, observation: Dict[str, object], model: str) -> Dict[str, float]:
    prompt = (
        "You are controlling daily prices for milk, banana, bread in a perishable retail simulator.\n"
        "Return ONLY strict JSON object with keys: price_milk, price_banana, price_bread.\n"
        "Keep prices realistic. Observation:\n"
        f"{json.dumps(observation)}"
    )
    completion = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    raw = completion.output_text.strip()
    payload = json.loads(raw)
    return {
        "milk": float(payload["price_milk"]),
        "banana": float(payload["price_banana"]),
        "bread": float(payload["price_bread"]),
    }


def run_episode(env: PerishablePricingEnv, task_id: str, seed: int, client: OpenAI | None, model: str) -> Dict[str, float]:
    obs = env.reset(seed=seed, task_id=task_id).model_dump()
    done = False
    total_reward = 0.0
    while not done:
        if client is None:
            price_map = heuristic_prices(obs)
        else:
            try:
                price_map = ask_model_for_prices(client, obs, model=model)
            except Exception:
                price_map = heuristic_prices(obs)
        action = ActionModel(
            price_milk=price_map["milk"],
            price_banana=price_map["banana"],
            price_bread=price_map["bread"],
        )
        obs_model, reward, done, info = env.step(action)
        obs = obs_model.model_dump()
        total_reward += reward
    kpis = info["episode_kpis"] or {}
    return {
        "task_score": float(kpis.get("task_score", 0.0)),
        "profit": float(kpis.get("profit", 0.0)),
        "service_level": float(kpis.get("service_level", 0.0)),
        "waste_rate": float(kpis.get("waste_rate", 1.0)),
        "total_reward": float(total_reward),
    }


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key) if api_key else None
    env = PerishablePricingEnv(artifact_dir="artifacts", verbose=True)
    results: Dict[str, List[Dict[str, float]]] = {}
    for task in TASKS:
        task_id = str(task["task_id"])
        results[task_id] = []
        for seed in SEEDS:
            episode = run_episode(env, task_id=task_id, seed=seed, client=client, model=model)
            episode["seed"] = seed
            results[task_id].append(episode)

    summary = {}
    for task_id, rows in results.items():
        summary[task_id] = {
            "avg_task_score": sum(x["task_score"] for x in rows) / len(rows),
            "avg_total_reward": sum(x["total_reward"] for x in rows) / len(rows),
            "avg_profit": sum(x["profit"] for x in rows) / len(rows),
            "avg_service_level": sum(x["service_level"] for x in rows) / len(rows),
            "avg_waste_rate": sum(x["waste_rate"] for x in rows) / len(rows),
        }

    payload = {"seeds": SEEDS, "episodes": results, "summary": summary}
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    out = Path("artifacts/baseline_scores.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

