"""逐题基线评测；保留原 GSM8K 判分语义，模拟指标明确标为诊断。"""

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from langsmith import tracing_context

from ad_l.fixed_graph import build_fixed_graph, resolve_fixed_topology
from ad_l.gsm8k_compat import gsm_get_predict
from ad_l.models import ModelAdapter, create_model_adapter
from ad_l.prompts import build_math_solver_messages
from ad_l.single_agent import build_single_agent_graph

SCORER = "agentdropout_gsm_get_predict_float_v1"
TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def score_answer(answer, target, status="success"):
    if status != "success":
        return {"prediction": None, "correct": False, "scoring_error": None}
    try:
        prediction = gsm_get_predict(answer)
        numeric = float(prediction)
        return {"prediction": prediction, "correct": math.isfinite(numeric) and numeric == float(target),
                "scoring_error": None}
    except (ValueError, TypeError, IndexError, AssertionError) as error:
        return {"prediction": None, "correct": False, "scoring_error": type(error).__name__}


def run_single(task, adapter):
    messages = build_math_solver_messages(task)
    started_at, started = datetime.now(timezone.utc).isoformat(), perf_counter()
    response, usage, error = "", None, None
    try:
        state = build_single_agent_graph(adapter).invoke({"task": task})
        response, usage = state["answer"], state["usage"]
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": "模型调用或回答处理失败",
                 "status_code": getattr(exception, "status_code", None)
                 if type(getattr(exception, "status_code", None)) is int else None}
    status = "failed" if error else "success"
    entry = {"stage": "agent", "node_id": "A", "role": "Math Solver", "round": 1,
             "spatial_info": {}, "temporal_info": {},
             "messages": [{"role": m.type, "content": m.content} for m in messages],
             "output": response if not error else None, "usage": usage, "status": status,
             "error": error, "settings": asdict(adapter.settings), "started_at_utc": started_at,
             "elapsed_seconds": perf_counter() - started,
             "real_api_attempted": adapter.settings.backend != "mock",
             "usage_source": "mock_not_measured" if adapter.settings.backend == "mock" else (
                 "provider" if usage is not None else "provider_unavailable")}
    return {"answer": response, "status": status, "trace": [entry], "round_outputs": [{"A": response}] if not error else [],
            "model_calls": 1, "real_api_calls": int(entry["real_api_attempted"])}


def summarize(rows, baselines, selected_count, backend):
    summary = {}
    for baseline in baselines:
        items = [row for row in rows if row["baseline"] == baseline]
        calls = [call for row in items for call in row["trace"]]
        known = {key: sum(call["usage"][key] for call in calls if call["usage"] is not None)
                 for key in TOKEN_KEYS}
        complete = bool(calls) and all(call["usage"] is not None for call in calls)
        correct = sum(row["correct"] for row in items)
        elapsed = sum(row["elapsed_seconds"] for row in items)
        summary[baseline] = {
            "selected": selected_count, "attempted": len(items), "unattempted": selected_count - len(items),
            "complete": len(items) == selected_count,
            "successes": sum(row["status"] == "success" for row in items),
            "failures": sum(row["status"] != "success" for row in items),
            "failure_rate": sum(row["status"] != "success" for row in items) / len(items) if items else None,
            "correct": correct if backend != "mock" else None,
            "accuracy": correct / len(items) if items and backend != "mock" else None,
            "mock_diagnostic_correct": correct if backend == "mock" else None,
            "accuracy_denominator": "all_attempted_including_failures",
            "scoring_errors": sum(row["scoring_error"] is not None for row in items),
            "elapsed_seconds": elapsed, "mean_seconds": elapsed / len(items) if items else None,
            "model_calls": len(calls), "real_api_calls": sum(call["real_api_attempted"] for call in calls),
            "usage": known if complete else None, "known_usage": known, "usage_complete": complete,
            "mean_total_tokens": known["total_tokens"] / len(items) if complete else None,
            "usage_source": "mock_not_measured" if backend == "mock" else (
                "provider" if complete else "provider_incomplete"),
        }
    return summary


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_baselines(samples, config, settings, output_dir: Path, manifest: dict, *,
                  baselines=("single", "fixed"), allow_real_api=False, adapter_factory=None,
                  learned_policy=None, learned_generator=None, learned_mode="sampled", learned_temperature=1.0,
                  policy_variants=None):
    policy_variants = policy_variants or {}
    if set(policy_variants) & {"single", "fixed", "learned"}:
        raise ValueError("配对变体名称不能覆盖原有基线")
    if policy_variants and (learned_mode != "original-fixed" or any(sample.split != "test" for sample in samples)):
        raise ValueError("剪枝配对评测只支持 test 划分和相同的 original-fixed 推理")
    if not samples or not baselines or len(set(baselines)) != len(baselines) or set(baselines) - {"single", "fixed", "learned", *policy_variants}:
        raise ValueError("需要非空样本和 single/fixed 基线")
    if "learned" in baselines:
        if learned_policy is None or learned_generator is None or any(sample.split != "test" for sample in samples):
            raise ValueError("学习后评测需要冻结策略、独立随机数生成器和 test 样本")
        if learned_mode not in {"sampled", "original-fixed"}:
            raise ValueError("未知学习后推理模式")
    if settings.backend != "mock" and not allow_real_api:
        raise ValueError("真实批量请求需要显式启用")
    if settings.max_retries != 0:
        raise ValueError("批量评测要求 max_retries=0")
    output_dir.mkdir(parents=True, exist_ok=False)
    # 两组共享相同的采样清单；工厂只接收模型配置，不接收标准答案或解题过程。
    factory = adapter_factory or (lambda: create_model_adapter(settings, allow_real_api=allow_real_api))

    def make_adapter():
        adapter = factory()
        if not isinstance(adapter, ModelAdapter) or adapter.settings != settings:
            raise ValueError("注入模型必须与本次统一的 ModelSettings 一致")
        return adapter
    rows = []
    progress = {**manifest, "status": "running", "summary": summarize(rows, baselines, len(samples), settings.backend)}
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", progress)
    with (output_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        with tracing_context(enabled=False):
            for sample in samples:
                for baseline in baselines:
                    started = perf_counter()
                    sampled_rounds = None
                    effective_topology = None
                    try:
                        if baseline == "single":
                            adapter = make_adapter()
                        else:
                            adapters = None if settings.backend == "mock" and adapter_factory is None else {
                                node: make_adapter() for node in ("A", "B", "final")}
                            active_config = config
                            round_topologies = None
                            if baseline == "learned" or baseline in policy_variants:
                                import torch
                                from dataclasses import replace
                                policy = policy_variants[baseline] if baseline in policy_variants else learned_policy
                                policy.eval()
                                with torch.no_grad():
                                    if learned_mode == "sampled":
                                        sampled_rounds = policy.sample_rounds(
                                            temperature=learned_temperature, generator=learned_generator)
                                        round_topologies = [item.for_execution() for item in sampled_rounds]
                                    else:
                                        active_config = replace(config,
                                            spatial_masks=tuple(tuple(int(v) for v in row) for row in policy.spatial_masks.tolist()),
                                            temporal_masks=tuple(tuple(int(v) for v in row) for row in policy.temporal_masks.tolist()))
                            effective_topology = resolve_fixed_topology(active_config) if round_topologies is None else None
                            graph = build_fixed_graph(active_config, adapters, allow_real_api=allow_real_api,
                                                      round_topologies=round_topologies)
                    except Exception as error:
                        state = {"status": "failed", "answer": "", "trace": [], "round_outputs": [],
                                 "model_calls": 0, "real_api_calls": 0,
                                 "setup_error": {"type": type(error).__name__, "message": "模型或流程初始化失败"}}
                    else:
                        state = run_single(sample.question, adapter) if baseline == "single" else graph.invoke(
                            {"task": sample.question})
                    row = {"sample_id": sample.sample_id, "split": sample.split, "index": sample.index,
                           "question": sample.question, "target": sample.target, "baseline": baseline,
                           "scorer": SCORER, "answer": state["answer"], "status": state["status"],
                           "trace": state["trace"], "round_outputs": state["round_outputs"],
                           "model_calls": state["model_calls"], "real_api_calls": state["real_api_calls"],
                           "setup_error": state.get("setup_error"),
                           "learned_mode": learned_mode if baseline == "learned" or baseline in policy_variants else None,
                           "effective_topology": effective_topology,
                           "sampled_rounds": [item.for_report() for item in sampled_rounds] if sampled_rounds is not None else None,
                           "elapsed_seconds": perf_counter() - started,
                           **score_answer(state["answer"], sample.target, state["status"])}
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    failed = state["status"] != "success"
                    progress = {**manifest, "status": "failed" if failed else "running",
                                "summary": summarize(rows, baselines, len(samples), settings.backend)}
                    write_json(output_dir / "summary.json", progress)
                    print(f"{sample.sample_id} {baseline}: {state['status']}")
                    if failed:
                        return progress
    progress["status"] = "success"
    write_json(output_dir / "summary.json", progress)
    return progress
