from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict
from urllib import request

from perishable_pricing_env.models import ActionModel


@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class EnvClient:
    """Client-side wrapper with async and sync interfaces."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def reset(self, **kwargs: Any) -> StepResult:
        payload = json.dumps(kwargs).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/reset",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req) as resp:
            observation = json.loads(resp.read().decode("utf-8"))
        return StepResult(observation=observation, reward=0.0, done=False, info={})

    async def step(self, action: ActionModel) -> StepResult:
        payload = json.dumps(action.model_dump()).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/step",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return StepResult(
            observation=data["observation"],
            reward=float(data["reward"]),
            done=bool(data["done"]),
            info=data.get("info", {}),
        )

    async def state(self) -> Dict[str, Any]:
        with request.urlopen(f"{self.base_url}/state") as resp:
            return json.loads(resp.read().decode("utf-8"))

    def sync(self) -> "SyncEnvClient":
        return SyncEnvClient(self)


class SyncEnvClient:
    def __init__(self, async_client: EnvClient):
        self._client = async_client

    def reset(self, **kwargs: Any) -> StepResult:
        return asyncio.run(self._client.reset(**kwargs))

    def step(self, action: ActionModel) -> StepResult:
        return asyncio.run(self._client.step(action))

    def state(self) -> Dict[str, Any]:
        return asyncio.run(self._client.state())

