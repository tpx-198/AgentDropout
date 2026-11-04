# 边剪枝与剪枝前后配对评测

## 使用方式

在 `ad-l` 环境、项目根目录运行；将检查点路径换成已有 GSM8K train 训练产物：

```bash
python prune.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt
```

默认剪枝比例 0.25，在同一批 10 道 test 题上离线运行剪枝前／后两组。结果保存到 `results/pruning/<run_id>/`；可用 `--output-dir` 指定尚不存在的新目录。

```bash
python prune.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --rate 0.1 --prune-only
python prune.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --limit 10 --deepseek --plan
python prune.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --limit 10 --deepseek
```

`--prune-only` 不加载评测数据、不创建模型，只生成剪枝检查点与报告。`--plan` 不保存文件、不读取 `.env`、不创建模型，只显示剪枝变化和调用计划。最后一条明确允许真实评测，最多 100 次模型调用，读取现有 `.env`。可指定相同的模型名称、模型温度、输出长度和超时参数。

## 原规则迁移

规则标识为 `agentdropout_update_masks_v1`，对应原项目 `Graph.update_masks` 的共享参数分支，空间／时间矩阵分别处理：

1. 计算当前 mask>0 的候选数和已禁用数。
2. 待剪数量为 `max(torch.round(active_count × rate), 1)`。
3. 将已禁用位置的排序副本权重置为全局最小 logit 减 1，按原 `torch.argsort` 排序。
4. 排序结果前“已禁用数＋待剪数”个位置置 mask=0。

只修改 mask，不改变 logit，不做重训练、不新增保连通约束。保留 torch.round 的取整、相同权重排序和 `rate=0` 仍至少尝试剪一条的行为。无剩余候选时实际剪除为 0。报告记录请求比例、待剪数量、实际剪除数量／比例和被剪边。

默认空间候选只有 A→B，一次剪枝可能将其全部移除。mask 候选数不等于实际执行边数：空间环路检查和固定时间边检查还会过滤连接。

## 统一推理与实际连接

两组统一使用 `original-fixed`：同一道题先跑剪枝前，再跑剪枝后；角色、提示词、模型参数、判分规则和最终汇总方式相同。参数冻结，不采样、不更新优化器。不能把剪枝前 sampled 与剪枝后 original-fixed 的结果混在一起比较。

原固定时间边受当前空间图的 check_cycle 影响。空间边删除后，原先被阻止的时间边可能开始生效。因此候选 mask 单调减少，但实际时间连接不保证单调减少。报告同时保存 `effective_before`、`effective_after`、`effective_removed` 和 `effective_added`；逐题记录保存实际连接及可见消息。这是原代码行为的迁移。

两组正常情况下都执行两名智能体、两轮和最终汇总，每题均为 5 次模型调用。边剪枝是否减少输入 Token 必须实测；节点剪枝尚未实现。

## 产物与来源

- `checkpoint.pt`：独立剪枝后检查点，保留训练步数、cursor、logit 和上下文，仅更新 mask，记录父文件 SHA-256、规则和剪枝历史。
- `pruning.json`：候选／实际连接变化、权重与实际剪枝比例。
- `evaluation/manifest.json`：两份检查点指纹、相同推理模式、样本编号、数据哈希、训练／评测模型来源、版本和当前源码指纹。
- `evaluation/predictions.jsonl`：每题 before、after 两行，包含消息、回答、判分、实际连接、用量及耗时。
- `evaluation/summary.json`：两组各自的完成、失败、正确率及用量统计。
- `comparison.json`：配对完整性、正确率差值、Token 差值／节省比例、调用数差值和平均耗时差值。

原检查点不修改，剪枝前后 logit 必须一致，mask 不得重新启用已有禁用项，并核验父文件指纹。剪枝产物可用 `evaluate.py --checkpoint ... --baseline learned --learned-mode original-fixed` 评测；当前标记 `evaluation_only=true`，拒绝直接续训。周期性训练剪枝和相应恢复语义将单独接入。

## 配对指标与限制

正确率差值使用 before、after 都已尝试的样本对，模型失败计为错误，缺失样本对单独列出。两组原始汇总保留所有尝试与失败。

Token、耗时和调用次数的效率差值只使用两组都成功的样本对，避免将失败少调用误报为节省。全部成功样本对均有提供方用量时才给出 Token 差值；否则为 null。Token 节省比例为 `1 - after / before`，分母为零时不计算。其他差值方向均为“剪枝后减剪枝前”。

模拟模式不输出模型正确率或 Token 节省，只保留诊断正确数与流程统计。固定 before→after 执行顺序、提供方缓存和网络状态可能影响耗时与缓存用量，单次少量样本不能证明普遍收益。任一调用失败即停，保留已完成和失败记录，不写入异常原文或密钥。

测试对照原 update_masks 在不同权重、相同权重、空候选和多个比例下的结果，并验证原文件／logit 不变、时间连接可能新增、配对消息与指标、失败不计效率收益，以及计划／仅剪枝模式不调用模型。
