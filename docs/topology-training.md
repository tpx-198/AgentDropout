# 拓扑参数、采样与一次离线更新

在 `ad-l` 环境、项目根目录执行：

```bash
python train.py
```

入口只做一次 Adam 更新，使用两个智能体、两轮讨论和 FinalRefer。模型始终为本地预设响应，不读取 `.env`，不创建真实客户端。默认使用合成题“3 个苹果再买 2 个”、标准答案 5，标为 `synthetic_train`，不使用 GSM8K 测试集，也不报告真实模型训练效果。

可选参数包括 `--seed`、`--lr`、`--initial-probability`、`--temperature`、`--delta`、`--config`、`--output-dir`。默认种子 0、学习率 0.1、初始边概率 0.5、温度 1、delta 0.1，CPU 执行。输出目录自动生成于 `results/topology/`，显式指定时必须尚不存在。

## 参数与采样规则

`TopologyPolicy` 是 `torch.nn.Module`，含空间和时间两个 2×2 `nn.Parameter`，以及两个不训练的 mask 缓冲区。初始 logit 为 `log(p / (1-p))`。当前迁移原项目 `diff=False` 分支：各轮独立采样，但共享同一组 logit；不使用逐轮独立参数。

每轮先采空间边，再采时间边，候选顺序固定为 A→A、A→B、B→A、B→B。边概率为 `sigmoid(logit / temperature)`。

- mask=0 的边不采样，也没有采样 log_prob 项。
- 空间边按当前已接受的边进行环路检查，拒绝自连接和成环边；被拒绝的候选不消耗随机数，也没有 log_prob 项。
- 第 1 轮没有时间边。第 2 轮的时间优化分支按 mask 采样所有候选，允许自连接和反向连接；这是原项目 `optimized_temporal=True` 的行为，与固定模式的时间边过滤不同。
- 被采样的候选无论接受还是拒绝，都分别累计 `log(p)` 或 `log(1-p)`。实现用 `logsigmoid` 稳定计算，不改变概率或截断梯度。

采样使用显式 `torch.Generator`。离散选择不求导，采样概率的 log_prob 保留计算图；全空 mask 时通过零锚点支持合法的零梯度 backward。

## 与 LangGraph 的边界

每道题先在训练器内生成两轮 `SampledRound`。其中 `log_prob` 是带梯度的张量，始终留在训练器内存；只把 `for_execution()` 生成的普通边列表传给现有执行图。

执行图按每轮空间边重新确定顺序，按本轮／上一轮分别筛选消息，轮末更新快照，最后按原规则汇总最后一轮两名智能体的回答。采样后的时间边直接用于可见性，不再应用固定模式的过滤规则。既有固定图默认行为继续保留。

## 奖励和损失

奖励沿用当前迁移的原 GSM8K 数值判分，答对为 1、答错为 0。默认损失对应原 `experiments/run_gsm8k.py` 主优化循环：

```text
policy_loss = mean(-reward × sum(round_log_prob))
regularization = nuclear_norm(spatial_logits) + nuclear_norm(temporal_logits)
               + relu(frobenius_norm(spatial_mask - spatial_logits) - delta)
               + relu(frobenius_norm(temporal_mask - temporal_logits) - delta)
loss = policy_loss + regularization
```

没有新增 Token 惩罚、熵奖励或奖励基线。原正则项作用于整个 logit 矩阵，因此 masked／空间自连接位置的总梯度可能来自正则项；这些位置仍不会获得采样策略梯度或变成有效空间边。

更新前单独计算并记录 `policy_gradients`，确认策略损失本身能回传到参数，再对总损失 backward、检查有限性并执行一次 Adam step。程序接口允许关闭正则项做明确标注的 `policy_only_ablation` 验证，CLI 默认保持原主循环损失。

固定模拟回答不会随拓扑变好，合成题奖励只是用于验证训练链路的信号。一次更新后概率会变化，但不保证同一随机数下的离散边立即翻转；这不代表正确率提升或剪枝有效。

## 记录与检查点

- `training.json`：合成题及划分、模型响应、逐轮采样边与选择概率、已 detach 的 log_prob 数值、奖励、策略／正则／总损失、策略／总梯度数值、更新前后参数和概率、随机种子、版本、源码指纹与调用次数。
- `checkpoint.pt`：更新后的参数与 mask、Adam 状态、采样 Generator 状态、步数和相关超参数。只包含脱离计算图的状态，不保存训练中的 log_prob、梯度计算图或 LangGraph state。

检查点可用于恢复参数、优化器与随机数状态；完整模型实验仍需保存提示词、数据、代码和模型配置。这里未启用 LangGraph 流程检查点。

测试覆盖解析的 Bernoulli 梯度、零 mask／零奖励、空间无环、时间自连接、按轮消息可见性、参数更新及检查点恢复；还抽取原项目采样函数，逐项对照相同种子下的边、log_prob 和梯度，并核对原正则项数值及梯度。采样验证与训练能力评估分开：当前只完成两节点、两轮、共享参数的一次离线更新，节点剪枝、边剪枝、`diff=True` 与真实训练均待后续接入。

相关机制见 [PyTorch 概率分布与 score-function 梯度说明](https://docs.pytorch.org/docs/stable/distributions.html)。
