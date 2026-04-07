"""Scaffold-style OpenEnv client for perishable pricing."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ActionModel, ObservationModel


class PerishablePricingClient(EnvClient[ActionModel, ObservationModel, State]):
    def _step_payload(self, action: ActionModel) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[ObservationModel]:
        obs_data = payload.get("observation", {})
        observation = ObservationModel(
            day_index=obs_data.get("day_index", 0),
            day_of_week=obs_data.get("day_of_week", 0),
            task_id=obs_data.get("task_id", ""),
            inventory_units=obs_data.get("inventory_units", {}),
            inventory_age_buckets=obs_data.get("inventory_age_buckets", {}),
            hours_to_expiry_min=obs_data.get("hours_to_expiry_min", {}),
            last_prices=obs_data.get("last_prices", {}),
            last_sales=obs_data.get("last_sales", {}),
            last_stockouts=obs_data.get("last_stockouts", {}),
            last_waste=obs_data.get("last_waste", {}),
            rolling_demand_estimate=obs_data.get("rolling_demand_estimate", {}),
            rolling_sales_estimate=obs_data.get("rolling_sales_estimate", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


__all__ = ["PerishablePricingClient"]
