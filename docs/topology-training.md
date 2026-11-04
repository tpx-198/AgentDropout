# GSM8K 多步拓扑训练、恢复与学习后评测

## 快速运行

在 `ad-l` 环境、项目根目录准备好 GSM8K 数据后执行：

```bash
python train.py --steps 3 --output-dir results/topology/gsm8k-demo
python evaluate.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --limit 10
```

两条命令默认均为本地模拟。训练默认使用完整的 7,473 题 train 样本池，固定种子 0，每步 2 题、每题 5 次模拟调用，更新 3 步；评测在 test 划分抽取相同的 10 题，分别运行 single、fixed 和 learned 三组，共 110 次模拟调用。结果只验证流程；模拟正确率保持 null，不能视作模型性能或训练改进。

`python train.py` 现在默认进行 GSM8K 多步训练。此前的合成题单步接口 `offline_training_step` 继续保留在代码中，供梯度诊断测试使用。

## 训练计划和真实模型

```bash
python train.py --steps 3 --batch-size 2 --deepseek --plan
python train.py --steps 3 --batch-size 2 --deepseek
```

`--plan` 不读取 `.env`、不创建模型客户端；第二条命令明确允许真实请求，以上规模最多 30 次。模型配置读取现有 `.env`，支持 `--model`、`--model-temperature`、`--max-tokens`、`--timeout`。`--temperature` 单独表示拓扑采样温度，与模型温度区分。

训练只接受 train 数据，不加载 test 标签。`--limit` 可固定抽取较小的训练样本池；默认使用完整训练集。样本池按种子打乱一次后循环遍历，跨批次／跨轮次的数据位置由 cursor 精确记录。标准答案只用于执行结束后的奖励，不传入模型消息或适配器工厂。固定模拟响应不会随拓扑变化，奖励可能全为 0，此时参数仍可能受原正则项影响。

每步使用 Adam 更新一次，损失保留原 GSM8K 主训练循环：

```text
policy_loss = mean(-reward × sum(round_log_prob))
regularization = nuclear_norm(spatial_logits) + nuclear_norm(temporal_logits)
               + relu(frobenius_norm(spatial_mask - spatial_logits) - delta)
               + relu(frobenius_norm(temporal_mask - temporal_logits) - delta)
loss = policy_loss + regularization
```

奖励与批量基线共用原项目数值判分规则（包括原提取器局限）。没有新增 Token 惩罚、熵奖励、奖励基线或剪枝。

## 检查点续训

```bash
python train.py --resume results/topology/gsm8k-demo/checkpoint.pt --steps 5 --output-dir results/topology/gsm8k-resumed
```

`--steps` 表示目标总步数，上例从第 3 步继续到第 5 步；省略时默认再走 1 步。新训练默认 3 步。每次输出使用新目录，不覆盖旧实验。真实模型检查点续训仍需显式 `--deepseek`。

检查点保存参数、mask、Adam 状态、采样 Generator 状态、累计数据 cursor 和训练上下文。恢复时校验数据文件哈希、样本池编号、模型参数、训练超参数、通信配置、判分版本、Python／库版本和源码指纹；不一致时拒绝混合续训。未指定的超参数从检查点恢复，显式指定的值必须相同。

固定模拟响应下，连续训练与中断恢复的参数、优化器状态、随机数状态、采样边和样本顺序一致。真实模型输出还受远端采样、缓存与服务状态影响，不承诺完全复现。

批次失败时停止，不提交该批的参数更新或数据进度；保留失败调用记录，checkpoint 仍指向最近成功步，续训重试该批。失败尝试仍可能产生费用，恢复运行会再次调用模型。原子替换检查点可以保护最近成功的状态；当前不对硬中断时尚未返回的请求推断用量。

旧 schema v1 合成题检查点缺少训练数据位置，不用于此入口的续训或学习后测试。

## 学习后拓扑评测

```bash
python evaluate.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --limit 10
python evaluate.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --baseline learned --limit 10
```

加载检查点时冻结参数，使用独立评测 RNG，只有 test 划分可运行 learned。默认指定检查点即在同一批题上比较 single（1 次／题）、fixed（5 次／题）、learned（5 次／题）；`--baseline learned` 则只运行学习后图。添加 `--deepseek` 明确允许真实请求，默认三组 10 题最多 110 次。`--plan` 可先检查规模。结果记录检查点路径、哈希、训练步数、训练数据／模型来源及推理模式。

两种推理规则分别标记，不能混为同一种原算法结果：

- `--learned-mode sampled`（默认）：冻结 logits，沿用原优化分支的边采样规则；使用检查点的拓扑温度，不反向传播、不更新参数。逐题保存采样边和概率。这是学习概率的冻结采样评测。
- `--learned-mode original-fixed`：读取检查点 mask，按原非优化分支的空间／时间建边规则执行。这个分支忽略 logits；当前尚未剪枝，mask 未变时可能与固定基线完全相同。不会为了显示学习效果而自动加入阈值剪枝。

真实评测各组使用相同模型配置；检查点训练时的配置单独记录，可识别用模拟策略或不同模型训练后再测试的情况，不能将其误报为同模型真实训练的收益。

## 参数、采样与执行边界

当前仍为两个节点、两轮、`diff=False`：空间和时间各有一个 2×2 可训练 logit 矩阵，两轮共享参数，mask 为固定缓冲区。概率为 `sigmoid(logit / temperature)`，初始 logit 为 `log(p/(1-p))`。

候选按 A→A、A→B、B→A、B→B 顺序处理。空间边拒绝自连接与环路，mask=0 或成环的候选不抽随机数、不计 log_prob；第一轮无时间边，优化分支第二轮允许时间自连接和反向连接。接受／拒绝的候选分别累计 log(p)／log(1-p)，以 logsigmoid 稳定计算并保留梯度。

PyTorch 持有 `SampledRound.log_prob`，只把普通连接数据传入 LangGraph。LangGraph 按逐轮空间图排序、筛选本轮／上一轮消息、更新轮末快照，并按原 FinalRefer 规则汇总。训练张量不经过 JSON 或 LangGraph 检查点；仅在报告时 detach 数值。原正则项作用于整个 logit 矩阵，因此被 mask 的位置也可能获得正则梯度，但不会因而成为有效采样边。

## 结果文件与验证范围

训练目录中：`manifest.json` 记录上下文和采样顺序指纹；`history.jsonl` 逐步保存奖励、损失、梯度、概率、题号和逐次调用；`training.json` 是最近一步详情；`summary.json` 汇总本次运行范围、完成步数、状态、已知／完整用量；`checkpoint.pt` 为最近成功步的恢复状态。续训输出是新目录，本次调用汇总不重复累计父运行，检查点中的 step／cursor 则是累计值。

真实 Token 以提供方返回为准；缺失时总量为 null，已知小计单独保存。模型调用次数为尝试次数，自动重试为零。训练和评测的完整消息、失败类型等记录均不含密钥、请求头或异常原文。

测试覆盖采样与原实现的边／log_prob／梯度对照、正则项对照、多步恢复一致性、失败不提交、训练／测试隔离、冻结评测不改变参数，以及原有单智能体、固定图和批量评测回归。当前仍未实现节点／边剪枝和 diff=True，不用模拟结果证明训练效果。
