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
               + relu(frobenius_norm(initial_spatial_mask - spatial_logits) - delta)
               + relu(frobenius_norm(initial_temporal_mask - temporal_logits) - delta)
loss = policy_loss + regularization
```

奖励与批量基线共用原项目数值判分规则（包括原提取器局限）。没有新增 Token 惩罚、熵奖励或奖励基线。

## 每轮独立拓扑（diff=True）

```bash
python train.py --diff --steps 6 --prune-every 2 --prune-until 4 --output-dir results/topology/round-demo
python evaluate.py --checkpoint results/topology/round-demo/checkpoint.pt --limit 10
python prune.py --checkpoint results/topology/round-demo/checkpoint.pt --limit 10
```

默认仍是共享参数（diff=False）；`--diff` 开启原项目的每轮独立分支。当前仍为 A、B 两名智能体、两轮与 FinalRefer：空间 logit／mask 形状为 `(2, 2, 2)`，首维对应第 1、2 轮；时间 logit／mask 为 `(1, 2, 2)`，对应第 1 轮到第 2 轮的通信。第一轮没有时间边，各轮空间采样仍逐条拒绝环路，优化模式的时间边保留自连接和反向边。

原 `diff=True` 正则项对空间矩阵、时间矩阵分别计算各轮核范数均值及 Frobenius 范数均值，然后对每种 Frobenius 均值施加一次 `relu(mean - delta)`；不是对所有轮次求和，也不是先逐轮 relu 再求均值。按原 `update_masks_diff` 每张矩阵独立排序剪枝，不把所有轮次混在一起排序。训练记录包含每张矩阵的候选数、剪除数和轮次；时间矩阵标为接收轮次（这里为第 2 轮）。

恢复时自动继承检查点的 diff 模式，不必再次写 `--diff`；显式 `--diff`／`--no-diff` 必须与检查点一致，拒绝在续训中切换。共享参数旧检查点继续支持评测；缺少 diff 字段按共享模式解释，形状不符会拒绝。源码变化仍遵守下文的严格续训来源校验。

学习后评测自动识别模式。`sampled`（默认）逐轮使用各自参数采样；`--learned-mode original-fixed` 使用各轮 mask 并保留原固定时间边过滤。两种模式都记录逐轮连接、消息可见性和调用。`original-fixed` 的 `effective_topology` 在 diff=True 时是两轮列表，在共享模式下仍为原单个拓扑对象。

迁移修正：核对原训练脚本时发现，原 Frobenius 目标始终是配置中的初始 mask，AD-L 之前在剪枝后误用了当前 mask。本次对两种模式都修正为初始配置，记录 `regularization_reference=initial_config_masks`；剪枝仅改变执行候选，不改变正则目标。这是原规则对齐，不是新增训练算法。历史实验未改写，不能把修正前后的训练轨迹当作同一实验续接。

本步骤未迁移节点选择权重或节点跳过；每题仍为 5 次调用。原节点剪枝依赖独立的 `spatial_logits_1`／`temporal_logits_1`，不能将边训练参数直接冒充节点选择参数。

## 检查点续训

```bash
python train.py --resume results/topology/gsm8k-demo/checkpoint.pt --steps 5 --output-dir results/topology/gsm8k-resumed
```

`--steps` 表示目标总步数，上例从第 3 步继续到第 5 步；省略时默认再走 1 步。新训练默认 3 步。每次输出使用新目录，不覆盖旧实验。真实模型检查点续训仍需显式 `--deepseek`。

检查点保存参数、mask、Adam 状态、采样 Generator 状态、累计数据 cursor 和训练上下文。恢复时校验数据文件哈希、样本池编号、模型参数、训练超参数、通信配置、判分版本、Python／库版本和源码指纹；不一致时拒绝混合续训。未指定的超参数从检查点恢复，显式指定的值必须相同。

固定模拟响应下，连续训练与中断恢复的参数、优化器状态、随机数状态、采样边和样本顺序一致。真实模型输出还受远端采样、缓存与服务状态影响，不承诺完全复现。

批次失败时停止，不提交该批的参数更新或数据进度；保留失败调用记录，checkpoint 仍指向最近成功步，续训重试该批。失败尝试仍可能产生费用，恢复运行会再次调用模型。原子替换检查点可以保护最近成功的状态；当前不对硬中断时尚未返回的请求推断用量。

旧 schema v1 合成题检查点缺少训练数据位置，不用于此入口的续训或学习后测试。

## 训练中的周期边剪枝

```bash
python train.py --steps 6 --prune-every 2 --prune-until 4 --output-dir results/topology/cycle-demo
```

这会在第 2、4 个成功更新步之后剪枝，第 3、5、6 步继续在缩减后的候选图上训练。默认 `--prune-rate 0.25`；`--prune-every 0`（默认）关闭周期剪枝，原有不剪枝训练保持原行为。`--prune-until` 是允许剪枝的最后一个绝对步数（含），省略表示不设截止。`--steps` 仍是总训练步数，不作为剪枝截止条件。

原 GSM8K 主循环在前 4 个更新步中每 2 步调用一次 update_masks。本入口复用相同的剪枝函数和“先更新、后剪枝”顺序，并将频率、比例和截止步显式参数化；这不是对原脚本硬编码的完整训练阶段、停训／推理切换的一比一移植。

每次剪枝仅修改 mask，不重置 Adam，不抽取随机数。下一步按当前 mask 采样，正则目标仍使用初始配置。被禁用的 logit 仍可能受正则或 Adam 动量更新，但 mask 不会自动重新启用。`--prune-rate 0` 按原规则仍至少尝试剪一条；要完全关闭应使用 `--prune-every 0`。

续训时只需要恢复训练检查点并增加目标步数：

```bash
python train.py --resume results/topology/cycle-demo/checkpoint.pt --steps 8
```

计划在训练开始时确定，续训从检查点继承；不能通过续训参数静默修改频率、比例或截止步。已有检查点仍遵循源码指纹与上下文校验，不绕过版本一致性检查。若希望比较不同剪枝计划，应从相同种子分别开始新实验。

检查点新增 `pruning_state`，同时保存计划和全部已完成事件。事件步号为全局训练步号；恢复会核对数量、顺序和截止范围，避免在剪枝边界重复或遗漏。更新后参数、mask、Adam、RNG、cursor 和剪枝进度在同一个原子检查点中提交。失败批次不更新、不推进数据、不剪枝；保存前中断时从最近成功步重新执行整步。

`history.jsonl` 每步新增 `pruning_event`（无剪枝时为 null），包含 mask、实际剪除数量及固定推理连接的前后变化。`summary.json` 记录累计／本次剪枝事件数和剩余候选数。终端也显示每次候选数变化。实际通信与 Token 是否减少仍需评测；两节点样例可能一次剪空空间边，时间边固定过滤还可能使某些连接开始生效。

集成训练产生的检查点可继续训练并由现有 evaluate.py 加载。独立 prune.py 产生的 `evaluation_only` 产物仍只供评测，不能当作周期训练的续训入口。默认均为本地模拟，添加 `--deepseek` 才允许真实调用；剪枝本身不增加调用次数，上例 6 步、每步 2 题最多 60 次。可加 `--plan` 先查看全局剪枝步号和调用规模。

已验证剪枝边界恢复、剪枝前失败及保存中断后重试，与连续训练的参数、Adam、RNG、数据位置和剪枝事件完全一致；这些完全一致的断言仅针对固定模拟响应。

## 学习后拓扑评测

```bash
python evaluate.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --limit 10
python evaluate.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --baseline learned --limit 10
```

加载检查点时冻结参数，使用独立评测 RNG，只有 test 划分可运行 learned。默认指定检查点即在同一批题上比较 single（1 次／题）、fixed（5 次／题）、learned（5 次／题）；`--baseline learned` 则只运行学习后图。添加 `--deepseek` 明确允许真实请求，默认三组 10 题最多 110 次。`--plan` 可先检查规模。结果记录检查点路径、哈希、训练步数、训练数据／模型来源及推理模式。

两种推理规则分别标记，不能混为同一种原算法结果：

- `--learned-mode sampled`（默认）：冻结 logits，沿用原优化分支的边采样规则；使用检查点的拓扑温度，不反向传播、不更新参数。逐题保存采样边和概率。这是学习概率的冻结采样评测。
- `--learned-mode original-fixed`：读取检查点 mask，按原非优化分支的空间／时间建边规则执行。这个分支忽略 logits；未剪枝、mask 未变时可能与固定基线完全相同。不会自动加入阈值剪枝。

真实评测各组使用相同模型配置；检查点训练时的配置单独记录，可识别用模拟策略或不同模型训练后再测试的情况，不能将其误报为同模型真实训练的收益。

## 参数、采样与执行边界

当前仍为两个节点、两轮。默认 `diff=False` 时空间和时间各有一个 2×2 可训练 logit 矩阵，两轮共享参数；`--diff` 按上述方式逐轮独立。mask 为可由剪枝更新的缓冲区。概率为 `sigmoid(logit / temperature)`，初始 logit 为 `log(p/(1-p))`。

候选按 A→A、A→B、B→A、B→B 顺序处理。空间边拒绝自连接与环路，mask=0 或成环的候选不抽随机数、不计 log_prob；第一轮无时间边，优化分支第二轮允许时间自连接和反向连接。接受／拒绝的候选分别累计 log(p)／log(1-p)，以 logsigmoid 稳定计算并保留梯度。

PyTorch 持有 `SampledRound.log_prob`，只把普通连接数据传入 LangGraph。LangGraph 按逐轮空间图排序、筛选本轮／上一轮消息、更新轮末快照，并按原 FinalRefer 规则汇总。训练张量不经过 JSON 或 LangGraph 检查点；仅在报告时 detach 数值。原正则项作用于整个 logit 矩阵，因此被 mask 的位置也可能获得正则梯度，但不会因而成为有效采样边。

## 结果文件与验证范围

训练目录中：`manifest.json` 记录上下文和采样顺序指纹；`history.jsonl` 逐步保存奖励、损失、梯度、概率、题号和逐次调用；`training.json` 是最近一步详情；`summary.json` 汇总本次运行范围、完成步数、状态、已知／完整用量；`checkpoint.pt` 为最近成功步的恢复状态。续训输出是新目录，本次调用汇总不重复累计父运行，检查点中的 step／cursor 则是累计值。

真实 Token 以提供方返回为准；缺失时总量为 null，已知小计单独保存。模型调用次数为尝试次数，自动重试为零。训练和评测的完整消息、失败类型等记录均不含密钥、请求头或异常原文。

测试覆盖两种参数模式与原实现的边／log_prob／梯度对照、正则项对照、逐轮剪枝、多步恢复一致性、失败不提交、训练／测试隔离、冻结评测不改变参数，以及原有单智能体、固定图和批量评测回归。独立边剪枝、配对评测与周期训练边剪枝均已接入，见 [边剪枝说明](edge-pruning.md)；节点剪枝仍待迁移，不用模拟结果证明训练效果。
