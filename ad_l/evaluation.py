"""Frozen-topology evaluation, shared by the local experiment platform."""

from time import perf_counter

import torch

from ad_l.system import build_system
from ad_l.training import score_example


def summarize_rows(rows, *, diagnostic=False):
    calls = [call for row in rows for call in row["result"]["trace"]]
    usage_complete = bool(calls) and all(call["usage"] is not None for call in calls)
    known_usage = {name: sum(call["usage"][name] for call in calls if call["usage"] is not None)
                   for name in ("input_tokens", "output_tokens", "total_tokens")}
    correct = sum(row["correct"] for row in rows)
    return {"attempted": len(rows), "correct": correct,
            "accuracy": correct / len(rows) if rows and not diagnostic else None,
            "diagnostic_ratio": correct / len(rows) if rows and diagnostic else None,
            "metrics_kind": "mock_or_mixed_diagnostic" if diagnostic else "model_evaluation",
            "failures": sum(row["result"]["status"] == "failed" for row in rows),
            "model_calls": len(calls), "real_api_calls": sum(call["real_api_attempted"] for call in calls),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
            "usage": known_usage if usage_complete else None, "known_usage": known_usage,
            "usage_complete": usage_complete}


def evaluate_examples(policy, examples, team, adapters, *, seed=0, mode="sampled", threshold=0.5,
                      temperature=1.0, allow_real_api=False, should_stop=None, progress=None, group="current"):
    policy.eval()
    generator = torch.Generator().manual_seed(seed)
    diagnostic = any(adapter.settings.backend == "mock" for adapter in adapters.values())
    rows = []
    status = "completed"
    for example in examples:
        if should_stop is not None and should_stop():
            status = "cancelled"
            break
        started = perf_counter()
        with torch.no_grad():
            sampled = policy.sample(generator, mode=mode, threshold=threshold, temperature=temperature)
        graph = build_system(team, sampled.execution, adapters, allow_real_api=allow_real_api, should_stop=should_stop)
        state = graph.invoke({"task": example.task, "answer_type": example.answer_type})
        if state["status"] == "cancelled" and not state["trace"]:
            status = "cancelled"
            break
        if state["status"] == "success":
            score = score_example(state["answer"], example)
        else:
            score = {"prediction": None, "correct": False}
        row = {"sample_id": example.sample_id, "dataset": example.dataset, "split": example.split,
               "target": example.target, "group": group, **score, "execution": sampled.execution,
               "result": state, "elapsed_seconds": perf_counter() - started}
        rows.append(row)
        if progress is not None:
            progress(row)
        if state["status"] != "success":
            status = "cancelled" if state["status"] == "cancelled" else "failed"
            break
    return {"status": status, "rows": rows, "summary": summarize_rows(rows, diagnostic=diagnostic)}
