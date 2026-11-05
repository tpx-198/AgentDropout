"""拓扑训练更新；梯度与 log_prob 始终留在训练器内存。"""

import torch
from langsmith import tracing_context

from ad_l.evaluation import score_answer
from ad_l.fixed_graph import build_fixed_graph
from ad_l.models import ModelAdapter, ModelSettings


def topology_training_step(policy, optimizer, config, examples, *, generator, temperature=1.0, delta=0.1,
                           include_regularization=True, settings=None, adapter_factory=None, allow_real_api=False):
    if not examples:
        raise ValueError("至少需要一道离线训练题")
    settings = settings or ModelSettings()
    if settings.backend != "mock" and not allow_real_api:
        raise ValueError("真实训练需要显式启用")
    if settings.max_retries != 0:
        raise ValueError("训练不允许自动重试")
    if settings.backend != "mock" and adapter_factory is None:
        raise ValueError("真实训练需要注入模型适配器")
    expected = {id(parameter) for parameter in policy.parameters()}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual:
        raise ValueError("优化器必须更新当前策略的参数")
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    log_probs, rewards, episodes, samples = [], [], [], []
    with tracing_context(enabled=False):
        for example in examples:
            sampled = policy.sample_rounds(2, temperature, generator=generator)
            # 这里只传普通边数据；log_prob 不进入 LangGraph state、JSON 或检查点。
            adapters = None
            if adapter_factory is not None:
                adapters = {node: adapter_factory() for node in ("A", "B", "final")}
                if any(not isinstance(adapter, ModelAdapter) or adapter.settings != settings for adapter in adapters.values()):
                    raise ValueError("训练适配器必须与统一模型配置一致")
            graph = build_fixed_graph(config, adapters, allow_real_api=allow_real_api,
                                      round_topologies=[item.for_execution() for item in sampled])
            state = graph.invoke({"task": example["task"]})
            if state["status"] != "success":
                episodes.append({"sample_id": example["sample_id"], "data_split": example.get("data_split", "synthetic_train"),
                                 "task": example["task"], "target": example["target"], "status": "failed",
                                 "trace": state["trace"], "round_outputs": state["round_outputs"], "setup_error": state.get("setup_error"),
                                 "sampled_rounds": [item.for_report() for item in sampled],
                                 "model_calls": state["model_calls"], "real_api_calls": state["real_api_calls"]})
                for episode, previous_sample in zip(episodes[:-1], samples):
                    episode["sampled_rounds"] = [item.for_report() for item in previous_sample]
                return {"status": "failed", "parameters_changed": False, "episodes": episodes,
                        "model_calls": sum(item["model_calls"] for item in episodes),
                        "real_api_calls": sum(item["real_api_calls"] for item in episodes)}
            score = score_answer(state["answer"], example["target"], state["status"])
            rewards.append(float(score["correct"]))
            log_probs.append(torch.stack([item.log_prob for item in sampled]).sum())
            episodes.append({"sample_id": example["sample_id"], "data_split": example.get("data_split", "synthetic_train"),
                             "task": example["task"], "target": example["target"], **score,
                             "answer": state["answer"], "trace": state["trace"],
                             "round_outputs": state["round_outputs"], "model_calls": state["model_calls"],
                             "real_api_calls": state["real_api_calls"]})
            samples.append(sampled)
    stacked = torch.stack(log_probs)
    policy_loss = -(stacked * stacked.new_tensor(rewards)).mean()
    regularization = policy.regularization_loss(delta, reference_masks=(config.spatial_masks, config.temporal_masks)) \
        if include_regularization else policy_loss * 0
    loss = policy_loss + regularization
    parameters = list(policy.parameters())
    policy_gradients = torch.autograd.grad(policy_loss, parameters, retain_graph=True)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in policy.named_parameters()}
    if not torch.isfinite(loss) or not all(torch.isfinite(gradient).all() for gradient in gradients.values()):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("损失或梯度非有限，未执行更新")
    optimizer.step()
    after = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    for episode, sampled in zip(episodes, samples):
        episode["sampled_rounds"] = [item.for_report() for item in sampled]
    calls = [call for episode in episodes for call in episode["trace"]]
    known = {key: sum(call["usage"][key] for call in calls if call["usage"] is not None)
             for key in ("input_tokens", "output_tokens", "total_tokens")}
    complete = all(call["usage"] is not None for call in calls)
    return {"status": "success", "loss_mode": "original_gsm8k_main" if include_regularization else "policy_only_ablation",
            "diff": policy.diff, "regularization_reference": "initial_config_masks" if include_regularization else None,
            "policy_loss": policy_loss.detach().item(), "regularization_loss": regularization.detach().item(),
            "loss": loss.detach().item(), "rewards": rewards, "episodes": episodes,
            "sampling_log_prob_requires_grad": stacked.requires_grad,
            "policy_gradients": {name: gradient.detach().tolist()
                                 for (name, _), gradient in zip(policy.named_parameters(), policy_gradients)},
            "gradients": {name: gradient.tolist() for name, gradient in gradients.items()},
            "before": {name: value.tolist() for name, value in before.items()},
            "after": {name: value.tolist() for name, value in after.items()},
            "probabilities_before": {name: torch.sigmoid(value / temperature).tolist() for name, value in before.items()},
            "probabilities_after": {name: torch.sigmoid(value / temperature).tolist() for name, value in after.items()},
            "parameters_changed": any(not torch.equal(before[name], after[name]) for name in before),
            "model_calls": sum(episode["model_calls"] for episode in episodes),
            "real_api_calls": sum(episode["real_api_calls"] for episode in episodes),
            "usage": known if complete else None, "known_usage": known, "usage_complete": complete,
            "usage_source": "mock_not_measured" if settings.backend == "mock" else ("provider" if complete else "provider_incomplete")}


def offline_training_step(policy, optimizer, config, examples, **kwargs):
    """保留合成题单步测试接口；完整训练入口使用 topology_training_step。"""
    return topology_training_step(policy, optimizer, config, examples, **kwargs)
