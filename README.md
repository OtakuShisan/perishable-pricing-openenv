# Perishable Pricing OpenEnv

Perishable Pricing OpenEnv is a real-world dark-store simulation where an agent sets daily prices for milk, banana, and bread while managing expiry, demand uncertainty, and restocking delays.

## Why this environment

Retail teams do this in real life: choose prices that increase profit without causing stockouts or waste.  
This environment captures that trade-off over a multi-day horizon.

## OpenEnv Interface

The environment follows the standard OpenEnv loop:
- `reset(seed, task_id) -> ObservationModel`
- `step(action) -> (observation, reward, done, info)`
- `state() -> dict`

Typed models:
- `perishable_pricing_env.models.ObservationModel`
- `perishable_pricing_env.models.ActionModel`
- `perishable_pricing_env.models.RewardModel`

3-component structure:
- `perishable_pricing_env/models.py` (typed contracts)
- `perishable_pricing_env/client.py` (training-side client wrapper)
- `server/environment.py`, `server/app.py` (server-side simulation + FastAPI)

## Action Space

The agent sets one price per SKU each day:
- `price_milk` in `[10, 120]`
- `price_banana` in `[5, 80]`
- `price_bread` in `[5, 90]`

## Observation Space

Observations include:
- day and task context
- inventory totals by SKU
- inventory age buckets by SKU
- minimum time-to-expiry by SKU
- lagging operational signals (`last_sales`, `last_waste`, `last_stockouts`, `last_prices`)
- rolling demand and sales estimates

## Reward Function

Shaped daily reward:

`reward = 0.75*normalized_profit - 0.45*waste_rate - 0.35*stockout_rate - 0.10*price_jump_penalty - invalid_action_penalty + terminal_bonus`

This gives continuous learning signal (not only terminal success) and penalizes undesirable behavior.

## Tasks (Easy -> Medium -> Hard)

Defined in `perishable_pricing_env/config.py`:
- `easy_stable_week`
- `medium_volatile_weekend`
- `hard_shock_supply_delay`

Each task has deterministic grading to `[0.0, 1.0]` based on episode KPI normalization (profit, waste control, service level).

## Trace Logging

Every run can produce:
- `artifacts/run_trace.csv`
- `artifacts/run_trace.jsonl`
- `artifacts/baseline_scores.json`
- `artifacts/trace_plots.png`

The trace captures prices, demand noise, sales, waste, stockouts, profit, and reward per step.

## Setup (Windows PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Locally

Run baseline (OpenAI API optional; heuristic fallback if key missing):

```bash
python scripts/run_baseline.py
```

Run mandatory submission inference (strict log format):

```bash
# required env vars
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="<your_token>"
python inference.py
```

Generate reward/profit/waste trends:

```bash
python scripts/plot_trace.py --input artifacts/run_trace.csv --output artifacts/trace_plots.png
```

Run server:

```bash
python -m server.app
```

## Validate OpenEnv package

```bash
openenv validate .
```

Pre-submit validator script:

```bash
bash scripts/validate-submission.sh https://<your-space>.hf.space .
```

## Docker

```bash
docker build -t perishable-pricing-openenv .
docker run --rm -p 7860:7860 perishable-pricing-openenv
```

## Hugging Face Spaces (Docker SDK)

1. Create a Docker Space.
2. Push this repository.
3. Ensure metadata includes `openenv` tag.
4. Set runtime env vars (for baseline model runs), e.g. `OPENAI_API_KEY`.

## Reproducibility

Baseline uses fixed seeds `[11, 22, 33]` across all tasks and writes aggregate metrics to `artifacts/baseline_scores.json`.

