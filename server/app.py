from openenv.core.env_server.http_server import create_app

from perishable_pricing_env.models import ActionModel, ObservationModel
from server.perishable_pricing_environment import PerishablePricingEnvironment

app = create_app(
    PerishablePricingEnvironment,
    ActionModel,
    ObservationModel,
    env_name="perishable-pricing-openenv",
    max_concurrent_envs=2,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

