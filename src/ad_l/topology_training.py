"""一次离线 Adam 更新；梯度与 log_prob 始终留在训练器内存。"""

import torch
from langsmith import tracing_context

from ad_l.evaluation import score_answer
from ad_l.fixed_graph import build_fixed_graph


def offline_training_step(policy, optimizer, config, examples, *, generator, temperature=1.0, delta=0.1,
                          include_regularization=True):
    if not examples:
        raise ValueError("至少需要一道离线训练题")
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
            graph = build_fixed_graph(config, round_topologies=[item.for_execution() for item in sampled])
            state = graph.invoke({"task": example["task"]})
            if state["status"] != "success":
                raise RuntimeError("离线调用失败，训练更新已中止")
            score = score_answer(state["answer"], example["target"], state["status"])
            rewards.append(float(score["correct"]))
            log_probs.append(torch.stack([item.log_prob for item in sampled]).sum())
            episodes.append({"sample_id": example["sample_id"], "data_split": "synthetic_train",
                             "task": example["task"], "target": example["target"], **score,
                             "answer": state["answer"], "trace": state["trace"],
                             "round_outputs": state["round_outputs"], "model_calls": state["model_calls"],
                             "real_api_calls": state["real_api_calls"]})
            samples.append(sampled)
    stacked = torch.stack(log_probs)
    policy_loss = -(stacked * stacked.new_tensor(rewards)).mean()
    regularization = policy.regularization_loss(delta) if include_regularization else policy_loss * 0
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
    return {"loss_mode": "original_gsm8k_main" if include_regularization else "policy_only_ablation",
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
            "model_calls": sum(episode["model_calls"] for episode in episodes), "real_api_calls": 0,
            "usage": None, "usage_source": "mock_not_measured"}
