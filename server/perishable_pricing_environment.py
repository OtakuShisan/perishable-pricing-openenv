"""Scaffold-style OpenEnv environment wrapper for perishable pricing."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from perishable_pricing_env.environment import PerishablePricingEnv
from perishable_pricing_env.models import ActionModel, ObservationModel


class PerishablePricingEnvironment(Environment):
    """OpenEnv-compatible adapter around the core perishable pricing simulator."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._env = PerishablePricingEnv(artifact_dir="artifacts", verbose=False)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed: int = 42, task_id: str = "easy_stable_week", **_: object) -> ObservationModel:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        obs = self._env.reset(seed=seed, task_id=task_id)
        obs.done = False
        obs.reward = 0.0
        return obs

    def step(self, action: ActionModel, **_: object) -> ObservationModel:  # type: ignore[override]
        obs, reward, done, info = self._env.step(action)
        self._state.step_count += 1
        obs.done = bool(done)
        obs.reward = float(reward)
        obs.metadata = {"info": info}
        return obs

    @property
    def state(self) -> State:
        return self._state


__all__ = ["PerishablePricingEnvironment"]
