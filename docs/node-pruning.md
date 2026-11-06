# 节点选择训练与剪枝评测

在 `ad-l` 环境、项目根目录运行 `python nodes.py`。默认从 GSM8K train 固定抽取 4 题训练 2 步，在 test 抽取 3 题配对评测。每道训练题最多 3 次调用，每道测试题剪枝前后合计 8 次调用，默认全程 36 次本地模拟。可用 `--steps`、`--batch-size`、`--limit`、`--seed`、`--lr` 和 `--output-dir` 调整实验。

```bash
python nodes.py --deepseek --plan
python nodes.py --deepseek
```

第一条只预览题数和调用上限。第二条显式启用真实模型，从本地 `.env` 或已有环境变量读取 DeepSeek 配置。

## 迁移规则

1. 节点选择使用独立于边拓扑训练的逐轮权重。在固定图上，对每轮每个节点的实际空间来边和出边求 logit 和，用 softmax 抽取一个跳过节点。
2. 按原脚本使用边级替代项。被选节点的每条关联边贡献 `4 × log(1 − sigmoid(logit))`，其他节点的关联边贡献 `log(sigmoid(logit))`。以最终答案对错作为 0/1 奖励，用 Adam 更新；原节点阶段的正则项为零，时间权重不参与更新。
3. 训练结束后，每轮按空间权重行和加列和、除以初始空间入度加出度来评分，剪除最低分节点。剪去该轮的空间来边和出边、前一轮到该节点的时间边，以及该节点到下一轮的时间边。同分按节点 ID 的字典序选择。
4. 训练抽样跳过时，固定图仍允许 `None.` 沿既有边传递。正式剪枝移除关联边后，普通节点收不到被剪节点的消息。两种情况下，FinalRefer 都按原规则接收最后一轮各节点的输出，包括被跳过节点的占位。跳过事件单独记录，模型用量只统计实际调用。

原脚本将部分索引写死为 5 节点。本入口按配置读取节点顺序，支持两轮，沿用原采样及替代损失语义，使用数值稳定的 `logsigmoid`。初始空间度数为零时报错；最低分使用直接取最小值，避免原脚本的固定分数上界导致错误索引。固定抽样规模和随机种子由本入口控制。

在两节点无环图中，同一条空间边对两端的采样分数贡献相同，抽取概率均为 0.5。默认配置和初始权重下，正式评分也始终并列，按顺序选 A。该版本适用于验证原规则、消息可见性与调用数。比较节点选择能力需要扩展协作图，评估准确率和 Token 则需要真实模型。

## 剪枝后训练边拓扑

将下面的 `pruning.json` 路径替换为 `nodes.py` 输出目录中的文件。

```bash
python train.py --node-pruning results/node_pruning/offline-demo/pruning.json --steps 2 --prune-every 2 --output-dir results/topology/node-edge
python evaluate.py --checkpoint results/topology/node-edge/checkpoint.pt --baseline learned
python prune.py --checkpoint results/topology/node-edge/checkpoint.pt
```

训练入口核对原配置和按跳过计划推导的掩码，自动启用逐轮独立模式。使用自定义初始配置时，传入相同的 `--config`。旧节点报告通过同目录的 `manifest.json` 核对原配置。边训练重新初始化边权重，沿用原始配置作为正则目标，使用剪过的掩码约束采样和后续边剪枝。

检查点保存报告哈希、初始剪枝掩码和跳过计划。用 `train.py --resume 检查点 --steps 目标总步数` 续训时直接继承这些信息。学习后评测在 sampled 和 original-fixed 两种模式都跳过计划中的节点。独立边剪枝的 before 和 after 都使用同一跳过计划。对 N 个智能体，两组均为每题 `2 × (N − 1) + 1` 次调用。

两节点下每轮只剩一名活跃智能体，空间候选为空，跨轮最多剩一条时间候选。sampled 可抽到跨轮自连接，original-fixed 仍按原规则过滤它。

## 三节点示例

```bash
python nodes.py --config configs/three_agents_offline.json --initial-probability 0.3 --output-dir results/node_pruning/three-node
python train.py --config configs/three_agents_offline.json --node-pruning results/node_pruning/three-node/pruning.json --output-dir results/topology/three-node
python evaluate.py --checkpoint results/topology/three-node/checkpoint.pt --baseline learned
```

A 是 Math Solver，B 是 Mathematical Analyst，C 使用原项目的 Inspector 提示词和示例。空间图为 A→B、A→C，两轮各保留一个跳过名额。使用示例的初始概率时，节点选择器从 A/B/C 约 17.6%/41.2%/41.2% 的分布开始。本地模拟的默认种子选出第 1 轮跳过 B、第 2 轮跳过 C；两轮仍分别有 A→C、A→B 候选边。这组结果用于检查概率和跨轮通信，真实题目上的选择效果有待配对评测。

自定义 JSON 用 `agents` 声明 ID 与角色，用 `responses` 提供同一批 ID 的两轮模拟输出。空间、时间掩码均按节点 ID 的字典序排列，行是来源、列是接收者。省略 `agents` 时使用原两节点配置。CLI 的调用上限按配置节点数和跳过计划计算。

## 结果文件

- `manifest.json` 保存数据哈希、样本编号、模型设置、种子与源码版本。
- `training.json` 保存每步奖励依据、替代损失、参数与梯度、逐题跳过计划及实际调用。
- `pruning.json` 保存节点分数、确定性跳过计划、剪枝后的掩码及实际连接前后对照。
- `evaluation/predictions.jsonl` 和 `comparison.json` 保存同题前后调用、可见消息、判分和用量统计。模拟结果的真实准确率和 Token 指标为 null。

## 三组同题对照

```bash
python nodes.py --config configs/three_agents_offline.json --initial-probability 0.3 --compare-random
```

在同一批 GSM8K test 题上按完整图（`before`）、随机剪枝（`random`）、学习后剪枝（`after`）的顺序评测。三组使用同一模型设置和固定建边规则。随机组每轮从全部节点均匀抽一个，两个抽样独立。它与学习后剪枝一样在全批题上复用同一计划，移除对应空间边和跨轮边，每题调用 5 次。完整图每题调用 7 次。

随机对照使用独立随机数生成器。`--random-seed` 只改变随机组，默认数值与 `--seed` 相同。实验允许随机计划与学习计划恰好一致，并在 `comparison.json` 标记。单个随机种子只代表一次计划抽样。

`comparison.json` 保留完整图与学习后剪枝的 `comparison`，新增 `random_comparisons`，分别记录随机剪枝对完整图、学习后剪枝对随机剪枝的成对差值。每个差值只使用该两组都尝试过的同题记录，效率只统计双方成功的题，Token 差值要求这些题的用量完整。逐题记录包含实际跳过节点和生效连接。

默认三节点规模为 4 道训练题、3 道测试题，共 71 次模型调用，其中训练 20 次，完整图、随机剪枝、学习后剪枝分别评测 21、15、15 次。添加 `--deepseek --plan` 只预览真实调用预算，显式使用 `--deepseek` 才会发起请求。
