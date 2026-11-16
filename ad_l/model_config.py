"""Per-agent API configuration with environment-based credential references."""

import hashlib
import json
import os
from pathlib import Path
import re

from ad_l.models import ModelSettings, create_model

FIELDS = {"backend", "model", "model_env", "api_key_env", "base_url", "temperature",
          "max_tokens", "timeout", "mock_response"}
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("API configuration contains duplicate fields")
        result[name] = value
    return result


def _settings(spec, *, resolve_env=False):
    model = spec.get("model")
    if "model_env" in spec:
        name = spec["model_env"]
        model = os.environ.get(name, "").strip() if resolve_env else f"env:{name}"
        if not model:
            raise ValueError(f"Set {name} in the environment or local .env")
    return ModelSettings(backend=spec.get("backend", "deepseek"),
                         model=model if model is not None else "local-fixed-response",
                         temperature=spec.get("temperature"), max_tokens=spec.get("max_tokens"),
                         timeout=spec.get("timeout", 60.0), base_url=spec.get("base_url"))


def load_agent_config(path: Path, team):
    raw = Path(path).read_bytes()
    nodes = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(nodes, dict) or set(nodes) != {agent.node_id for agent in team}:
        raise ValueError("API config must contain exactly the current team node IDs, including final")
    for node_id, spec in nodes.items():
        if not isinstance(spec, dict) or set(spec) - FIELDS:
            raise ValueError(f"Unsupported API config fields for {node_id}; use api_key_env for credentials")
        if "model" in spec and "model_env" in spec:
            raise ValueError(f"{node_id}: choose model or model_env")
        if spec.get("backend", "deepseek") != "mock" and not ({"model", "model_env"} & set(spec)):
            raise ValueError(f"{node_id}: specify model or model_env")
        for field in ("model_env", "api_key_env"):
            if field in spec and (not isinstance(spec[field], str) or ENV_NAME.fullmatch(spec[field]) is None):
                raise ValueError(f"{node_id}: {field} must name an environment variable")
        if "mock_response" in spec and not isinstance(spec["mock_response"], str):
            raise ValueError(f"{node_id}: mock_response must be text")
        try:
            _settings(spec)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid model settings for {node_id}") from error
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "nodes": nodes}


def create_agent_models(config, *, allow_real_api=False, env_file: Path | None = None):
    nodes = config["nodes"]
    real_nodes = [node for node, spec in nodes.items() if spec.get("backend", "deepseek") != "mock"]
    if real_nodes and not allow_real_api:
        raise ValueError("Real agent APIs require --allow-real-api")
    if real_nodes and env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    settings = {node: _settings(spec, resolve_env=True) for node, spec in nodes.items()}
    # Resolve every credential before constructing any clients. No environment mutation is used.
    for node in real_nodes:
        prefix = "DEEPSEEK" if settings[node].backend == "deepseek" else "OPENAI"
        name = nodes[node].get("api_key_env", f"{prefix}_API_KEY")
        if not os.environ.get(name, "").strip():
            raise ValueError(f"{node}: set {name} in the environment or local .env")
    adapters = {node: create_model(settings[node], allow_real_api=allow_real_api,
                                   api_key_env=spec.get("api_key_env"),
                                   mock_response=spec.get("mock_response", "[MOCK fixed response]\nThe answer is 5"))
                for node, spec in nodes.items()}
    return adapters, {node: adapter.settings for node, adapter in adapters.items()}


def model_call_plan(config, team, rounds: int, examples: int):
    calls_by_node = {agent.node_id: examples * (1 if agent == team[-1] else rounds) for agent in team}
    real_calls = sum(calls for node, calls in calls_by_node.items()
                     if config["nodes"][node].get("backend", "deepseek") != "mock")
    return {"agent_api_config": config, "planned_model_calls_by_agent": calls_by_node,
            "planned_real_api_calls": real_calls}
