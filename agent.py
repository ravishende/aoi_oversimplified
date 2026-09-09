import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import TOOLS
from graph import build_recommendation_graph


load_dotenv()


def load_model_config(
    model_key: str = "default",
    config_path: str = "models.yaml",
) -> dict:
    """Load model configuration from YAML and resolve environment variables."""

    path = Path(config_path)
    with path.open("r") as file:
        models = yaml.safe_load(file)

    if model_key not in models:
        raise ValueError(f"Unknown model: {model_key}")

    config = models[model_key]

    for key, value in config.items():
        if isinstance(value, str):
            match = re.fullmatch(r"\$\{([^}]+)\}", value)

            if match:
                env_var = match.group(1)
                env_value = os.getenv(env_var)

                if not env_value:
                    raise ValueError(
                        f"Environment variable '{env_var}' is not set"
                    )

                config[key] = env_value

    return config


model_config = load_model_config(
    model_key="default"
)


model = ChatOpenAI(
    model=model_config["model"],
    api_key=model_config["api_key"],
    base_url=model_config["base_url"],
    temperature=0.3,
)
build_recommendation_graph(model)


agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=(
        "You are a simple AoI assistant. "
        "Answer the user's question concisely. "
        "Use tools when they are useful."
    ),
)


def run_agent(user_input: str) -> str:
    """Run the agent and return its final response."""

    result = agent.invoke(
        {"messages": [{
            "role": "user",
            "content": user_input,
        }]}
    )

    return result["messages"][-1].content