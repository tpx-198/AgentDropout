"""Local platform configuration, background experiments and result persistence."""

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import tempfile
import threading
from time import perf_counter
from uuid import uuid4

import torch
from dotenv import dotenv_values
from langsmith import tracing_context

from ad_l.datasets import DATASETS, load_training_mix, read_examples
from ad_l.evaluation import evaluate_examples, summarize_rows
from ad_l.model_config import load_agent_config
from ad_l.models import ModelSettings, create_model
from ad_l.roles import PROMPTS, create_team
from ad_l.topology import EdgePolicy, TopologyConfig, load_policy, save_policy
from ad_l.training import train_step


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def default_config():
    team = create_team()
    return {"topology": asdict(TopologyConfig()), "roles": [agent.role for agent in team[:-1]],
            "models": {agent.node_id: {"backend": "mock", "model": "local-fixed-response", "base_url": None,
                        "temperature": None, "max_tokens": None, "timeout": 60.0,
                        "mock_response": "[MOCK fixed response]\nThe answer is 5"} for agent in team},
            "sampling_temperature": 1.0,
            "training": {"datasets": list(DATASETS), "limit": 6, "steps": 3, "batch_size": 2,
                         "lr": 0.05, "sparsity": 0.01, "prune_rate": 0.25},
            "evaluation": {"datasets": list(DATASETS), "limit": 5, "mode": "sampled", "threshold": 0.5,
                           "checkpoint": "", "compare": True}}


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}须为 {low}～{high} 的整数")


def validate_config(config):
    config = copy.deepcopy(config)
    topology = TopologyConfig(**config["topology"])
    _integer(topology.agents, 2, 32, "智能体数量")
    _integer(topology.rounds, 1, 10, "讨论轮数")
    _integer(topology.seed, -2147483648, 2147483647, "随机种子")
    team = create_team(topology.agents, config["roles"])
    if set(config["models"]) != {agent.node_id for agent in team}:
        raise ValueError("每个智能体都需要单独的模型配置")
    for agent in team:
        spec = config["models"][agent.node_id]
        if set(spec) - {"backend", "model", "base_url", "temperature", "max_tokens", "timeout", "mock_response"}:
            raise ValueError("模型配置包含不支持的字段")
        spec = {**default_config()["models"]["agent_1"], **spec}
        config["models"][agent.node_id] = spec
        model = spec.get("model", "")
        if not isinstance(model, str):
            raise ValueError("模型名称须为文本")
        ModelSettings(backend=spec["backend"], model=model.strip() or "pending-configuration",
                      temperature=spec.get("temperature"), max_tokens=spec.get("max_tokens"),
                      timeout=spec.get("timeout", 60), base_url=spec.get("base_url"))
        if not isinstance(spec.get("mock_response", ""), str):
            raise ValueError("模拟回答须为文本")
    for name in ("training", "evaluation"):
        section = config[name]
        datasets = section["datasets"]
        if not isinstance(datasets, list) or not datasets or len(set(datasets)) != len(datasets) or set(datasets) - set(DATASETS):
            raise ValueError("至少选择一个有效数据集")
        _integer(section["limit"], 1, 1000, "每类抽样题数")
    train = config["training"]
    _integer(train["steps"], 1, 1000, "训练步数")
    _integer(train["batch_size"], 1, 64, "批大小")
    if not math.isfinite(train["lr"]) or train["lr"] <= 0:
        raise ValueError("学习率须为有限正数")
    if not math.isfinite(train["sparsity"]) or train["sparsity"] < 0 or not 0 <= train["prune_rate"] <= 1:
        raise ValueError("正则系数或剪枝比例无效")
    if config["evaluation"]["mode"] not in {"sampled", "fixed", "threshold"}:
        raise ValueError("评测推理模式无效")
    if not 0 <= config["evaluation"]["threshold"] <= 1 or type(config["evaluation"]["compare"]) is not bool:
        raise ValueError("评测阈值或对照设置无效")
    if not isinstance(config["evaluation"]["checkpoint"], str):
        raise ValueError("检查点选择无效")
    if not math.isfinite(config["sampling_temperature"]) or config["sampling_temperature"] <= 0:
        raise ValueError("采样温度须为有限正数")
    return config


class SettingsStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        if path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))
            validate_config(self.state["config"])
        else:
            self.state = {"config": default_config(), "keys": {}}

    def public(self):
        with self.lock:
            return {"config": copy.deepcopy(self.state["config"]),
                    "key_present": {node: bool(self.state["keys"].get(node)) for node in self.state["config"]["models"]}}

    def save(self, config, keys=None, clear_keys=None):
        config = validate_config(config)
        keys, clear_keys = keys or {}, clear_keys or []
        if not isinstance(keys, dict) or set(keys) - set(config["models"]) or set(clear_keys) - set(config["models"]):
            raise ValueError("密钥对应的智能体无效")
        for key in keys.values():
            if not isinstance(key, str) or len(key) > 8192 or any(ord(character) < 32 for character in key):
                raise ValueError("API 密钥格式无效")
        with self.lock:
            stored = {node: key for node, key in self.state["keys"].items() if node in config["models"]}
            for node in clear_keys:
                stored.pop(node, None)
            stored.update({node: key.strip() for node, key in keys.items() if key.strip()})
            state = {"config": config, "keys": stored}
            atomic_json(self.path, state)
            self.state = state
            return self.public()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)


class PlatformService:
    def __init__(self, root: Path, state_dir: Path | None = None):
        self.root = Path(root)
        self.state_dir = state_dir or self.root / "platform_state"
        self.store = SettingsStore(self.state_dir / "settings.json")
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        for path in sorted((self.state_dir / "jobs").glob("*/summary.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] in {"queued", "running", "cancelling"}:
                record.update(status="interrupted", message="平台重启，任务已中断")
                atomic_json(path, record)
            self.jobs[record["id"]] = record
        self.cancel_event = threading.Event()

    def dataset_catalog(self):
        catalog = []
        for name in DATASETS:
            directory = self.root / "data" / name
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            splits = {}
            for split in ("train", "validation", "test"):
                path = directory / f"{split}.jsonl"
                if path.exists():
                    count = manifest.get("splits", {}).get(split, {}).get("records")
                    if count is None:
                        count = len(path.read_bytes().splitlines())
                    splits[split] = count
            catalog.append({"name": name, "splits": splits})
        return catalog

    def import_local_config(self):
        state = self.store.snapshot()
        config = state["config"]
        path = self.root / "configs/agents.json"
        values = json.loads(path.read_text())
        team = create_team(len(values))
        imported = load_agent_config(path, team)
        env = {**{key: value for key, value in dotenv_values(self.root / ".env").items() if value is not None}, **os.environ}
        config["topology"]["agents"] = len(team)
        config["roles"] = [agent.role for agent in team[:-1]]
        config["models"] = {}
        keys = {}
        for agent in team:
            spec = imported["nodes"][agent.node_id]
            backend = spec.get("backend", "deepseek")
            model = spec.get("model", env.get(spec.get("model_env", ""), ""))
            config["models"][agent.node_id] = {
                "backend": backend, "model": model, "base_url": spec.get("base_url"),
                "temperature": spec.get("temperature"), "max_tokens": spec.get("max_tokens"),
                "timeout": spec.get("timeout", 60),
                "mock_response": spec.get("mock_response", "[MOCK fixed response]\nThe answer is 5")}
            key_name = spec.get("api_key_env", "DEEPSEEK_API_KEY" if backend == "deepseek" else "OPENAI_API_KEY")
            if env.get(key_name):
                keys[agent.node_id] = env[key_name]
        return self.store.save(config, keys)

    def checkpoints(self):
        items = []
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            path = self.state_dir / "jobs" / job["id"] / "checkpoint.pt"
            if path.exists() and job.get("completed_steps", 0) > 0:
                items.append({"id": f"job:{job['id']}", "label": f"{job['name']} · {job['completed_steps']} 步"})
        for path in sorted((self.root / "checkpoints").glob("*.pt")):
            items.append({"id": f"file:{path.name}", "label": path.name})
        return items

    def _checkpoint(self, identity):
        allowed = {item["id"] for item in self.checkpoints()}
        if identity not in allowed:
            raise ValueError("请选择有效的已训练检查点")
        kind, name = identity.split(":", 1)
        return self.state_dir / "jobs" / name / "checkpoint.pt" if kind == "job" else self.root / "checkpoints" / name

    def _policy(self, config, kind):
        topology = TopologyConfig(**config["topology"])
        identity = config["evaluation"]["checkpoint"] if kind == "evaluate" else ""
        if identity:
            policy, metadata = load_policy(self._checkpoint(identity))
            if policy.config != topology or metadata.get("roles") != config["roles"]:
                raise ValueError("当前智能体、轮数、拓扑或角色与检查点不一致，请先应用检查点设置")
            return policy
        return EdgePolicy(topology)

    def checkpoint_settings(self, identity):
        policy, metadata = load_policy(self._checkpoint(identity))
        fallback = [agent.role for agent in create_team(policy.config.agents)[:-1]]
        return {"topology": asdict(policy.config), "roles": metadata.get("roles", fallback)}

    def preview(self, config):
        config = validate_config(config)
        policy = self._policy(config, "evaluate")
        with torch.no_grad():
            sampled = policy.sample(torch.Generator().manual_seed(policy.config.seed), mode="fixed")
        return {**policy.report(), "execution": sampled.execution}

    def _examples(self, config, kind):
        section = config["training" if kind == "train" else "evaluation"]
        if kind == "train":
            return load_training_mix(self.root / "data", section["datasets"], section["limit"], config["topology"]["seed"])
        pools, metadata = [], {}
        for name in section["datasets"]:
            records, info = read_examples(self.root / "data" / name / "test.jsonl", expected_split="test", dataset=name)
            if section["limit"] > len(records):
                raise ValueError(f"{name} 测试集只有 {len(records)} 题，请调小抽样题数")
            indices = sorted(random.Random(config["topology"]["seed"]).sample(range(len(records)), section["limit"]))
            pools.append([records[index] for index in indices])
            metadata[name] = info
        examples = [pool[index] for index in range(section["limit"]) for pool in pools]
        return examples, {"split": "test", "datasets": metadata, "selected": len(examples)}

    def plan(self, config, kind):
        config = validate_config(config)
        if kind not in {"train", "evaluate"}:
            raise ValueError("任务类型无效")
        self._policy(config, kind)
        examples, dataset = self._examples(config, kind)
        groups = 2 if kind == "evaluate" and config["evaluation"]["checkpoint"] and config["evaluation"]["compare"] else 1
        count = config["training"]["steps"] * config["training"]["batch_size"] if kind == "train" else len(examples) * groups
        rounds = config["topology"]["rounds"]
        team = create_team(config["topology"]["agents"], config["roles"])
        calls = {agent.node_id: count * (1 if agent.node_id == "final" else rounds) for agent in team}
        real_calls = sum(value for node, value in calls.items() if config["models"][node]["backend"] != "mock")
        return {"kind": kind, "selected": len(examples), "groups": groups, "dataset": dataset,
                "model_calls": sum(calls.values()), "real_api_calls": real_calls, "calls_by_agent": calls,
                "sample_ids": [example.sample_id for example in examples]}

    def submit(self, kind, config, *, allow_real_api=False):
        config = validate_config(config)
        with self.lock:
            if self.active is not None:
                raise ValueError("已有任务正在执行，请等待完成或取消")
            plan = self.plan(config, kind)
            state = self.store.snapshot()
            keys = state["keys"]
            for node, spec in config["models"].items():
                if spec["backend"] != "mock":
                    if not allow_real_api:
                        raise ValueError("请勾选本次任务允许真实 API 调用")
                    if not spec["model"].strip() or not keys.get(node):
                        raise ValueError(f"请先在模型配置中填写 {node} 的模型名称和 API 密钥，并保存")
            identity = uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            label = "拓扑训练" if kind == "train" else "测试集评测"
            job = {"id": identity, "name": f"{label} · {now[5:16].replace('T', ' ')} UTC", "kind": kind,
                   "created_at": now, "status": "queued", "message": "正在准备数据与模型",
                   "progress": 0, "total": config["training"]["steps"] if kind == "train" else plan["selected"] * plan["groups"],
                   "completed_steps": 0, "config": config, "plan": plan, "rows": [], "losses": [], "summary": {}}
            self.jobs[identity] = job
            self.active = identity
            self.cancel_event = threading.Event()
            self._persist(job)
            threading.Thread(target=self._run, args=(identity, copy.deepcopy(keys), allow_real_api), daemon=True).start()
            return self.public_job(identity)

    def _persist(self, job):
        atomic_json(self.state_dir / "jobs" / job["id"] / "summary.json", job)

    def public_job(self, identity):
        with self.lock:
            if identity not in self.jobs:
                raise ValueError("任务不存在")
            return copy.deepcopy(self.jobs[identity])

    def history(self):
        with self.lock:
            return [{key: job[key] for key in ("id", "name", "kind", "created_at", "status", "progress", "total", "summary")}
                    for job in sorted(self.jobs.values(), key=lambda item: item["created_at"], reverse=True)]

    def cancel(self, identity):
        with self.lock:
            if self.active == identity and self.jobs[identity]["status"] in {"queued", "running", "cancelling"}:
                self.cancel_event.set()
                self.jobs[identity].update(status="cancelling", message="正在结束当前模型请求，随后停止")
                self._persist(self.jobs[identity])
            return self.public_job(identity)

    def _adapters(self, config, keys, allow_real_api):
        adapters = {}
        for node, spec in config["models"].items():
            settings = ModelSettings(**{name: spec.get(name) for name in
                ("backend", "model", "base_url", "temperature", "max_tokens", "timeout")})
            adapters[node] = create_model(settings, api_key=keys.get(node), allow_real_api=allow_real_api,
                                         mock_response=spec.get("mock_response", "[MOCK]\nThe answer is 5"))
        return adapters

    def _run(self, identity, keys, allow_real_api):
        job = self.jobs[identity]
        directory = self.state_dir / "jobs" / identity
        config = job["config"]
        started = perf_counter()
        try:
            config = copy.deepcopy(config)
            policy = self._policy(config, job["kind"])
            examples, dataset = self._examples(config, job["kind"])
            team = create_team(policy.config.agents, config["roles"])
            adapters = self._adapters(config, keys, allow_real_api)
            keys.clear()
            diagnostic = any(adapter.settings.backend == "mock" for adapter in adapters.values())
            with self.lock:
                job.update(status="running", message="任务执行中")
                self._persist(job)
            with tracing_context(enabled=False):
                if job["kind"] == "train":
                    self._train(job, directory, policy, examples, team, adapters, dataset, diagnostic, allow_real_api)
                else:
                    self._evaluate(job, directory, policy, examples, team, adapters, dataset, diagnostic, allow_real_api)
        except Exception as error:
            with self.lock:
                job.update(status="failed", message=f"任务执行失败（{type(error).__name__}），请检查模型配置和运行记录")
        finally:
            keys.clear()
            with self.lock:
                job["wall_seconds"] = perf_counter() - started
                self._persist(job)
                self.active = None

    def _train(self, job, directory, policy, examples, team, adapters, dataset, diagnostic, allow_real_api):
        config = job["config"]
        parameters = config["training"]
        optimizer = torch.optim.Adam(policy.parameters(), lr=parameters["lr"])
        generator = torch.Generator().manual_seed(policy.config.seed)
        reports, rows = [], []
        status = "completed"
        for step in range(parameters["steps"]):
            if self.cancel_event.is_set():
                status = "cancelled"
                break
            batch = [examples[(step * parameters["batch_size"] + offset) % len(examples)] for offset in range(parameters["batch_size"])]
            report = train_step(policy, optimizer, batch, adapters, generator, sparsity=parameters["sparsity"],
                                temperature=config["sampling_temperature"], allow_real_api=allow_real_api,
                                roles=config["roles"], should_stop=self.cancel_event.is_set)
            reports.append(report)
            for episode in report["episodes"]:
                rows.append({**episode, "group": "training", "elapsed_seconds": sum(call["elapsed_seconds"] for call in episode["result"]["trace"])})
            with self.lock:
                if report["status"] == "success":
                    job["completed_steps"] = step + 1
                    job["progress"] = step + 1
                    job["losses"].append({"step": step + 1, "loss": report["loss"], "policy_loss": report["policy_loss"]})
                job["rows"] = [self._compact(row) for row in rows]
                job["summary"] = summarize_rows(rows, diagnostic=diagnostic)
                job["message"] = f"已完成 {job['completed_steps']} / {parameters['steps']} 步"
                self._persist(job)
            metadata = {"status": report["status"], "roles": config["roles"], "dataset": dataset,
                        "history": reports, "settings": {node: asdict(adapter.settings) for node, adapter in adapters.items()},
                        "hyperparameters": parameters, "metrics_kind": "mock_diagnostic" if diagnostic else "model_training",
                        "prompt_source": PROMPTS["source"], "completed_steps": job["completed_steps"]}
            save_policy(directory / "checkpoint.pt", policy, metadata)
            if report["status"] != "success":
                status = report["status"]
                break
        if self.cancel_event.is_set():
            status = "cancelled"
        pruning = policy.prune(parameters["prune_rate"]) if status == "completed" else None
        if job["completed_steps"]:
            metadata.update(status=status, pruning=pruning)
            save_policy(directory / "checkpoint.pt", policy, metadata)
        atomic_json(directory / "details.json", {"kind": "train", "reports": reports, "rows": rows, "dataset": dataset})
        with self.lock:
            job.update(status=status, message={"completed": "训练完成，剪枝与检查点已保存", "cancelled": "训练已取消，已保留成功步骤的检查点"}.get(status, "模型调用失败，已保留运行记录"),
                       topology=policy.report(), pruning=pruning)

    def _evaluate(self, job, directory, policy, examples, team, adapters, dataset, diagnostic, allow_real_api):
        config = job["config"]
        options = config["evaluation"]
        policies = [("current", policy)]
        if options["checkpoint"] and options["compare"]:
            policies = [("baseline", EdgePolicy(policy.config)), ("trained", policy)]
        rows = []

        def progress(row):
            rows.append(row)
            with self.lock:
                job.update(progress=len(rows), rows=[self._compact(value) for value in rows],
                           summary={name: summarize_rows([value for value in rows if value["group"] == name], diagnostic=diagnostic)
                                    for name, _ in policies}, message=f"已完成 {len(rows)} / {job['total']} 次答题")
                self._persist(job)
            atomic_json(directory / "details.json", {"kind": "evaluate", "rows": rows, "dataset": dataset})

        status = "completed"
        for group, active in policies:
            report = evaluate_examples(active, examples, team, adapters, seed=policy.config.seed,
                mode=options["mode"], threshold=options["threshold"], temperature=config["sampling_temperature"],
                allow_real_api=allow_real_api, should_stop=self.cancel_event.is_set, progress=progress, group=group)
            if report["status"] != "completed":
                status = report["status"]
                break
        if self.cancel_event.is_set():
            status = "cancelled"
        with self.lock:
            job.update(status=status, message={"completed": "评测完成，结果已保存", "cancelled": "评测已取消，已保留已完成题目"}.get(status, "模型调用失败，已保留运行记录"),
                       topology=policy.report())

    @staticmethod
    def _compact(row):
        return {"sample_id": row["sample_id"], "dataset": row["dataset"], "group": row["group"],
                "target": row["target"], "prediction": row.get("prediction"), "correct": row["correct"],
                "status": row["result"]["status"], "model_calls": row["result"]["model_calls"],
                "elapsed_seconds": row["elapsed_seconds"]}

    def export(self, identity):
        summary = self.public_job(identity)
        path = self.state_dir / "jobs" / identity / "details.json"
        details = json.loads(path.read_text()) if path.exists() else {}
        return {"summary": summary, "details": details}
