from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ProductConfig:
    name: str
    cost_price: float
    ref_price: float
    elasticity: float
    base_demand: int
    max_market_units: int
    shelf_life_hours: int
    restock_every_days: int
    restock_target: int


PRODUCTS: Dict[str, ProductConfig] = {
    "milk": ProductConfig(
        name="milk",
        cost_price=33.0,
        ref_price=40.0,
        elasticity=1.2,
        base_demand=800,
        max_market_units=900,
        shelf_life_hours=24,
        restock_every_days=1,
        restock_target=850,
    ),
    "banana": ProductConfig(
        name="banana",
        cost_price=15.0,
        ref_price=20.0,
        elasticity=1.0,
        base_demand=300,
        max_market_units=420,
        shelf_life_hours=36,
        restock_every_days=2,
        restock_target=360,
    ),
    "bread": ProductConfig(
        name="bread",
        cost_price=17.0,
        ref_price=24.0,
        elasticity=0.9,
        base_demand=400,
        max_market_units=520,
        shelf_life_hours=60,
        restock_every_days=2,
        restock_target=450,
    ),
}


DEFAULT_WEEKDAY_FACTORS: Dict[int, float] = {
    0: 1.00,
    1: 0.97,
    2: 1.02,
    3: 1.05,
    4: 1.08,
    5: 1.15,
    6: 1.10,
}


TASKS: List[Dict[str, object]] = [
    {
        "task_id": "easy_stable_week",
        "difficulty": "easy",
        "horizon_days": 14,
        "noise_sigma": 0.06,
        "weekday_factor_scale": 0.4,
        "demand_shock": {},
        "supply_delay_days": [],
    },
    {
        "task_id": "medium_volatile_weekend",
        "difficulty": "medium",
        "horizon_days": 14,
        "noise_sigma": 0.14,
        "weekday_factor_scale": 1.0,
        "demand_shock": {},
        "supply_delay_days": [],
    },
    {
        "task_id": "hard_shock_supply_delay",
        "difficulty": "hard",
        "horizon_days": 14,
        "noise_sigma": 0.18,
        "weekday_factor_scale": 1.0,
        "demand_shock": {
            7: {"milk": 1.3, "banana": 1.4, "bread": 0.8},
            8: {"milk": 1.2, "banana": 1.25, "bread": 0.9},
        },
        "supply_delay_days": [8],
    },
]

