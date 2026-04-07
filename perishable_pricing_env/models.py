from __future__ import annotations

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


SKU = str


class ObservationModel(Observation):
    day_index: int
    day_of_week: int = Field(ge=0, le=6)
    task_id: str
    inventory_units: Dict[SKU, int]
    inventory_age_buckets: Dict[SKU, List[int]]
    hours_to_expiry_min: Dict[SKU, float]
    last_prices: Dict[SKU, float]
    last_sales: Dict[SKU, int]
    last_stockouts: Dict[SKU, int]
    last_waste: Dict[SKU, int]
    rolling_demand_estimate: Dict[SKU, float]
    rolling_sales_estimate: Dict[SKU, float]


class ActionModel(Action):
    price_milk: float = Field(ge=10.0, le=120.0)
    price_banana: float = Field(ge=5.0, le=80.0)
    price_bread: float = Field(ge=5.0, le=90.0)

    def as_price_map(self) -> Dict[SKU, float]:
        return {
            "milk": self.price_milk,
            "banana": self.price_banana,
            "bread": self.price_bread,
        }


class RewardModel(BaseModel):
    reward: float
    profit_term: float
    waste_penalty: float
    stockout_penalty: float
    stability_penalty: float
    invalid_action_penalty: float
    terminal_bonus: float = 0.0


class StepInfo(BaseModel):
    day_index: int
    task_id: str
    prices: Dict[SKU, float]
    demand: Dict[SKU, int]
    noise: Dict[SKU, float]
    sales: Dict[SKU, int]
    stockouts: Dict[SKU, int]
    waste: Dict[SKU, int]
    inventory_end: Dict[SKU, int]
    profit_value: float
    reward_breakdown: RewardModel
    episode_kpis: Optional[Dict[str, float]] = None

