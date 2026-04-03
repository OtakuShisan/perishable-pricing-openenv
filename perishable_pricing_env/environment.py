from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from perishable_pricing_env.config import DEFAULT_WEEKDAY_FACTORS, PRODUCTS, TASKS
from perishable_pricing_env.models import ActionModel, ObservationModel, RewardModel, StepInfo


@dataclass
class InventoryBatch:
    units: int
    age_hours: int


class PerishablePricingEnv:
    def __init__(self, artifact_dir: str = "artifacts", verbose: bool = False):
        self.artifact_dir = Path(artifact_dir)
        self.verbose = verbose
        self._task = TASKS[0]
        self._rng = np.random.default_rng(42)
        self._day_index = 0
        self._done = False
        self._inventory: Dict[str, List[InventoryBatch]] = {}
        self._last_prices = {sku: PRODUCTS[sku].ref_price for sku in PRODUCTS}
        self._last_sales = {sku: 0 for sku in PRODUCTS}
        self._last_stockouts = {sku: 0 for sku in PRODUCTS}
        self._last_waste = {sku: 0 for sku in PRODUCTS}
        self._rolling_demand = {sku: float(PRODUCTS[sku].base_demand) for sku in PRODUCTS}
        self._rolling_sales = {sku: 0.0 for sku in PRODUCTS}
        self._episode_rows: List[Dict[str, object]] = []
        self._episode_totals = {
            "profit": 0.0,
            "demand": 0,
            "sales": 0,
            "waste": 0,
            "stockouts": 0,
        }

    def reset(self, seed: int = 42, task_id: str = "easy_stable_week") -> ObservationModel:
        task = next((x for x in TASKS if x["task_id"] == task_id), None)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")
        self._task = task
        self._rng = np.random.default_rng(seed)
        self._day_index = 0
        self._done = False
        self._inventory = {
            sku: [InventoryBatch(units=cfg.restock_target, age_hours=0)] for sku, cfg in PRODUCTS.items()
        }
        self._last_prices = {sku: PRODUCTS[sku].ref_price for sku in PRODUCTS}
        self._last_sales = {sku: 0 for sku in PRODUCTS}
        self._last_stockouts = {sku: 0 for sku in PRODUCTS}
        self._last_waste = {sku: 0 for sku in PRODUCTS}
        self._rolling_demand = {sku: float(PRODUCTS[sku].base_demand) for sku in PRODUCTS}
        self._rolling_sales = {sku: float(PRODUCTS[sku].base_demand * 0.6) for sku in PRODUCTS}
        self._episode_rows = []
        self._episode_totals = {"profit": 0.0, "demand": 0, "sales": 0, "waste": 0, "stockouts": 0}
        return self._build_observation()

    def state(self) -> Dict[str, object]:
        return {
            "day_index": self._day_index,
            "task_id": self._task["task_id"],
            "done": self._done,
            "inventory": {
                sku: [{"units": b.units, "age_hours": b.age_hours} for b in batches]
                for sku, batches in self._inventory.items()
            },
            "episode_totals": dict(self._episode_totals),
        }

    def step(self, action: ActionModel) -> Tuple[ObservationModel, float, bool, Dict[str, object]]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        prices = action.as_price_map()
        weekday = self._day_index % 7
        demand = {}
        noise_draw = {}
        sales = {}
        stockouts = {}
        waste = {}
        opening_stock = {sku: self._total_stock(sku) for sku in PRODUCTS}
        profit_value = 0.0

        for sku, cfg in PRODUCTS.items():
            day_factor = 1.0 + (DEFAULT_WEEKDAY_FACTORS[weekday] - 1.0) * float(self._task["weekday_factor_scale"])
            shock = self._task["demand_shock"].get(self._day_index, {}).get(sku, 1.0)
            sigma = float(self._task["noise_sigma"])
            noise = float(self._rng.lognormal(mean=0.0, sigma=sigma))
            noise_draw[sku] = noise
            price_term = math.exp(-cfg.elasticity * ((prices[sku] / cfg.ref_price) - 1.0))
            expected_demand = cfg.base_demand * day_factor * shock * price_term * noise
            sampled_demand = int(max(0, min(cfg.max_market_units, round(expected_demand))))
            demand[sku] = sampled_demand
            sold = self._deplete_fifo(sku, sampled_demand)
            sales[sku] = sold
            stockouts[sku] = max(0, sampled_demand - sold)
            profit_value += sold * (prices[sku] - cfg.cost_price)
            self._rolling_demand[sku] = 0.8 * self._rolling_demand[sku] + 0.2 * float(sampled_demand)
            self._rolling_sales[sku] = 0.8 * self._rolling_sales[sku] + 0.2 * float(sold)

        self._age_inventory(24)
        waste = self._remove_expired()
        self._restock()

        reward_model = self._compute_reward(prices, demand, sales, stockouts, waste, opening_stock, profit_value)
        self._episode_totals["profit"] += profit_value
        self._episode_totals["demand"] += int(sum(demand.values()))
        self._episode_totals["sales"] += int(sum(sales.values()))
        self._episode_totals["waste"] += int(sum(waste.values()))
        self._episode_totals["stockouts"] += int(sum(stockouts.values()))

        self._last_prices = prices
        self._last_sales = sales
        self._last_stockouts = stockouts
        self._last_waste = waste

        row = {
            "day": self._day_index,
            "task_id": self._task["task_id"],
            "price_milk": prices["milk"],
            "price_banana": prices["banana"],
            "price_bread": prices["bread"],
            "noise_milk": noise_draw["milk"],
            "noise_banana": noise_draw["banana"],
            "noise_bread": noise_draw["bread"],
            "demand_milk": demand["milk"],
            "demand_banana": demand["banana"],
            "demand_bread": demand["bread"],
            "sales_milk": sales["milk"],
            "sales_banana": sales["banana"],
            "sales_bread": sales["bread"],
            "waste_milk": waste["milk"],
            "waste_banana": waste["banana"],
            "waste_bread": waste["bread"],
            "stockout_milk": stockouts["milk"],
            "stockout_banana": stockouts["banana"],
            "stockout_bread": stockouts["bread"],
            "profit": profit_value,
            "reward": reward_model.reward,
        }
        self._episode_rows.append(row)
        self._print_row(row)

        self._day_index += 1
        self._done = self._day_index >= int(self._task["horizon_days"])
        info_model = StepInfo(
            day_index=self._day_index - 1,
            task_id=str(self._task["task_id"]),
            prices=prices,
            demand=demand,
            noise=noise_draw,
            sales=sales,
            stockouts=stockouts,
            waste=waste,
            inventory_end={sku: self._total_stock(sku) for sku in PRODUCTS},
            profit_value=profit_value,
            reward_breakdown=reward_model,
            episode_kpis=self.grade_episode() if self._done else None,
        )

        if self._done:
            self._write_logs()
        return self._build_observation(), reward_model.reward, self._done, info_model.model_dump()

    def grade_episode(self) -> Dict[str, float]:
        total_demand = max(1, self._episode_totals["demand"])
        total_sales = self._episode_totals["sales"]
        service_level = total_sales / total_demand
        waste_rate = self._episode_totals["waste"] / max(1, total_sales + self._episode_totals["waste"])
        profit = self._episode_totals["profit"]
        profit_cap = float(sum(p.base_demand * (p.ref_price - p.cost_price) for p in PRODUCTS.values()) * int(self._task["horizon_days"]))
        profit_norm = max(0.0, min(1.0, profit / max(1.0, profit_cap)))
        waste_score = max(0.0, 1.0 - min(1.0, waste_rate / 0.25))
        service_score = max(0.0, min(1.0, (service_level - 0.6) / 0.35))
        if str(self._task["difficulty"]) == "easy":
            score = 0.5 * profit_norm + 0.25 * waste_score + 0.25 * service_score
        elif str(self._task["difficulty"]) == "medium":
            score = 0.4 * profit_norm + 0.3 * waste_score + 0.3 * service_score
        else:
            score = 0.35 * profit_norm + 0.3 * waste_score + 0.35 * service_score
        return {
            "task_score": float(max(0.0, min(1.0, score))),
            "profit_norm": float(profit_norm),
            "waste_score": float(waste_score),
            "service_score": float(service_score),
            "service_level": float(service_level),
            "waste_rate": float(waste_rate),
            "profit": float(profit),
        }

    def _build_observation(self) -> ObservationModel:
        age_buckets = {}
        min_hours_to_expiry = {}
        for sku, cfg in PRODUCTS.items():
            buckets = [0, 0, 0]
            for batch in self._inventory[sku]:
                remaining = cfg.shelf_life_hours - batch.age_hours
                if remaining <= 24:
                    buckets[0] += batch.units
                elif remaining <= 48:
                    buckets[1] += batch.units
                else:
                    buckets[2] += batch.units
            age_buckets[sku] = buckets
            min_hours_to_expiry[sku] = float(
                min((cfg.shelf_life_hours - b.age_hours for b in self._inventory[sku]), default=0.0)
            )

        return ObservationModel(
            day_index=self._day_index,
            day_of_week=self._day_index % 7,
            task_id=str(self._task["task_id"]),
            inventory_units={sku: self._total_stock(sku) for sku in PRODUCTS},
            inventory_age_buckets=age_buckets,
            hours_to_expiry_min=min_hours_to_expiry,
            last_prices=dict(self._last_prices),
            last_sales=dict(self._last_sales),
            last_stockouts=dict(self._last_stockouts),
            last_waste=dict(self._last_waste),
            rolling_demand_estimate=dict(self._rolling_demand),
            rolling_sales_estimate=dict(self._rolling_sales),
        )

    def _total_stock(self, sku: str) -> int:
        return int(sum(b.units for b in self._inventory[sku]))

    def _deplete_fifo(self, sku: str, demand_units: int) -> int:
        sold = 0
        idx = 0
        while idx < len(self._inventory[sku]) and demand_units > 0:
            batch = self._inventory[sku][idx]
            take = min(batch.units, demand_units)
            batch.units -= take
            demand_units -= take
            sold += take
            if batch.units == 0:
                self._inventory[sku].pop(idx)
            else:
                idx += 1
        return sold

    def _age_inventory(self, hours: int) -> None:
        for batches in self._inventory.values():
            for batch in batches:
                batch.age_hours += hours

    def _remove_expired(self) -> Dict[str, int]:
        waste = {}
        for sku, cfg in PRODUCTS.items():
            kept = []
            expired_units = 0
            for batch in self._inventory[sku]:
                if batch.age_hours >= cfg.shelf_life_hours:
                    expired_units += batch.units
                else:
                    kept.append(batch)
            self._inventory[sku] = kept
            waste[sku] = expired_units
        return waste

    def _restock(self) -> None:
        for sku, cfg in PRODUCTS.items():
            if (self._day_index + 1) in self._task["supply_delay_days"]:
                continue
            if (self._day_index + 1) % cfg.restock_every_days == 0:
                needed = max(0, cfg.restock_target - self._total_stock(sku))
                if needed > 0:
                    self._inventory[sku].append(InventoryBatch(units=needed, age_hours=0))

    def _compute_reward(
        self,
        prices: Dict[str, float],
        demand: Dict[str, int],
        sales: Dict[str, int],
        stockouts: Dict[str, int],
        waste: Dict[str, int],
        opening_stock: Dict[str, int],
        profit_value: float,
    ) -> RewardModel:
        profit_cap = sum(cfg.base_demand * (cfg.ref_price - cfg.cost_price) for cfg in PRODUCTS.values())
        normalized_profit = profit_value / max(1.0, profit_cap)
        waste_rate = sum(waste.values()) / max(1.0, float(sum(opening_stock.values())))
        stockout_rate = sum(stockouts.values()) / max(1.0, float(sum(demand.values())))
        jump = np.mean(
            [abs(prices[sku] - self._last_prices[sku]) / max(1.0, PRODUCTS[sku].ref_price) for sku in PRODUCTS]
        )
        invalid_penalty = 0.0
        for sku, price in prices.items():
            ref = PRODUCTS[sku].ref_price
            if price < 0.6 * ref or price > 1.8 * ref:
                invalid_penalty += 0.2

        reward = (
            0.75 * normalized_profit
            - 0.45 * waste_rate
            - 0.35 * stockout_rate
            - 0.10 * float(jump)
            - invalid_penalty
        )
        terminal_bonus = 0.0
        if self._day_index + 1 >= int(self._task["horizon_days"]):
            service_level = self._episode_totals["sales"] / max(1, self._episode_totals["demand"])
            cumulative_waste_rate = self._episode_totals["waste"] / max(
                1, self._episode_totals["waste"] + self._episode_totals["sales"]
            )
            if service_level >= 0.9 and cumulative_waste_rate <= 0.08:
                terminal_bonus = 0.1
                reward += terminal_bonus
        return RewardModel(
            reward=float(reward),
            profit_term=float(0.75 * normalized_profit),
            waste_penalty=float(0.45 * waste_rate),
            stockout_penalty=float(0.35 * stockout_rate),
            stability_penalty=float(0.10 * jump),
            invalid_action_penalty=float(invalid_penalty),
            terminal_bonus=float(terminal_bonus),
        )

    def _print_row(self, row: Dict[str, object]) -> None:
        if not self.verbose:
            return
        print(
            " | ".join(
                [
                    f"day={row['day']}",
                    f"prices=({row['price_milk']:.2f},{row['price_banana']:.2f},{row['price_bread']:.2f})",
                    f"sales=({row['sales_milk']},{row['sales_banana']},{row['sales_bread']})",
                    f"waste=({row['waste_milk']},{row['waste_banana']},{row['waste_bread']})",
                    f"stockouts=({row['stockout_milk']},{row['stockout_banana']},{row['stockout_bread']})",
                    f"profit={row['profit']:.2f}",
                    f"reward={row['reward']:.4f}",
                ]
            )
        )

    def _write_logs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.artifact_dir / "run_trace.csv"
        jsonl_path = self.artifact_dir / "run_trace.jsonl"
        if self._episode_rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._episode_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._episode_rows)
            with jsonl_path.open("w", encoding="utf-8") as f:
                for row in self._episode_rows:
                    f.write(json.dumps(row) + "\n")

