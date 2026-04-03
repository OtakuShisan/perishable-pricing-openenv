# Perishable Pricing OpenEnv

Perishable Pricing OpenEnv simulates a real dark-store pricing task where an agent sets daily prices for milk, banana, and bread under expiry constraints, demand uncertainty, and scheduled restocking.

## Motivation

Dynamic pricing with perishables is a real operational problem. Bad pricing causes:
- low profit from underpricing,
- stockouts from over-demand,
- waste from unsold expired units.

This environment forces an agent to balance all three over multi-day horizons.

## Environment API

Implements standard OpenEnv style methods:
- `reset(seed, task_id) -> ObservationModel`
- `step(action) -> (ObservationModel, reward, done, info)`
- `state() -> dict`

Typed models are defined in `perishable_pricing_env/models.py`.

## Observation Space

`ObservationModel` includes:
- day and weekday context (`day_index`, `day_of_week`, `task_id`)
- inventory totals per SKU
- inventory age buckets per SKU
- minimum hours-to-expiry per SKU
- lagging signals (`last_prices`, `last_sales`, `last_stockouts`, `last_waste`)
- rolling estimates (`rolling_demand_estimate`, `rolling_sales_estimate`)

## Action Space

`ActionModel`:
- `price_milk` in `[10, 120]`
- `price_banana` in `[5, 80]`
- `price_bread` in `[5, 90]`

## Reward Function

Daily shaped reward:

`reward = 0.75*normalized_profit - 0.45*waste_rate - 0.35*stockout_rate - 0.10*price_jump_penalty - invalid_action_penalty + terminal_bonus`

Where:
- `normalized_profit` scales realized daily profit to a reference cap,
- `waste_rate` penalizes expiry losses,
- `stockout_rate` penalizes poor service levels,
- `price_jump_penalty` discourages unstable pricing,
- `invalid_action_penalty` discourages unrealistic extreme pricing.

`info` returns full reward breakdown each step.

## Tasks and Difficulty

Defined in `perishable_pricing_env/config.py`:
- `easy_stable_week` (easy): low noise, stable demand.
- `medium_volatile_weekend` (medium): stronger stochasticity + weekday pattern.
- `hard_shock_supply_delay` (hard): demand shock + delayed supply event.

Each task is graded deterministically to `[0, 1]` using KPI normalization:
- profit quality,
- waste control,
- service level.

## Logging and Traceability

During baseline runs, per-step summaries print in terminal. End-of-episode traces are saved:
- `artifacts/run_trace.csv`
- `artifacts/run_trace.jsonl`
- `artifacts/baseline_scores.json`

These include chosen prices, noise draws, demand, sales, stockouts, waste, profit, and reward.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Local Usage

Run baseline (uses `OPENAI_API_KEY` if set, otherwise deterministic heuristic fallback):

```bash
python scripts/run_baseline.py
```

Run API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## OpenEnv Validation

If OpenEnv CLI is installed:

```bash
openenv validate openenv.yaml
```

## Docker

Build:

```bash
docker build -t perishable-pricing-openenv .
```

Run:

```bash
docker run --rm -p 7860:7860 perishable-pricing-openenv
```

## Hugging Face Spaces

Use Docker SDK Space and push this repository with:
- `Dockerfile`
- `openenv.yaml`
- project source files

Set Space metadata/tag to include `openenv`.

## Baseline Reproducibility

Baseline uses fixed seed list `[11, 22, 33]` across all three tasks and writes aggregate scores to `artifacts/baseline_scores.json`.

