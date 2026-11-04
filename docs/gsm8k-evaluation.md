# GSM8K 数据与批量基线

## 数据来源与准备

数据来自 [GSM8K 作者仓库](https://github.com/openai/grade-school-math)，遵循其 [MIT 许可证](https://github.com/openai/grade-school-math/blob/master/LICENSE)。每行包含 `question` 和 `answer`，标准数值答案位于解题过程末尾的 `####` 后面。

默认从只读原项目复制已存在的官方数据到 AD-L：

```bash
python scripts/prepare_gsm8k.py
```

没有原项目文件时，可直接从作者仓库下载：

```bash
python scripts/prepare_gsm8k.py --download
```

| 划分 | AD-L 本地文件 | 题数 | SHA-256 |
|---|---|---:|---|
| train | `data/gsm8k/train.jsonl` | 7,473 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| test | `data/gsm8k/test.jsonl` | 1,319 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

原项目中测试文件名为 `gsm8k.jsonl`，复制后明确命名为 `test.jsonl`。准备和评测入口均核验官方文件的内容指纹、条数和标签格式；已有文件先校验，不覆盖。`data/` 被 Git 忽略，重新克隆项目后需重新准备数据；运行评测不会自动联网下载。

## 运行方式

在 `ad-l` 环境、项目根目录执行。默认从测试集按种子 0 抽取 10 题，两组使用完全相同的样本：

```bash
python evaluate.py
```

这条命令仅使用预设模拟响应，验证加载、运行、判分和保存流程。模拟响应独立于标准答案，不能根据它们报告模型性能；汇总的 `accuracy` 和 `correct` 为 `null`，诊断正确数单独放在 `mock_diagnostic_correct`。

查看真实调用计划（不读取 `.env`，不创建模型、不调用 API）：

```bash
python evaluate.py --deepseek --limit 10 --plan
```

使用已配置的本地 `.env` 正式评测：

```bash
python evaluate.py --deepseek --limit 10
```

此命令明确允许真实请求：默认两组每题最多 6 次模型调用，10 题最多 60 次。顺序执行，不做自动重试；任一组某题调用失败时停止后续请求并保留部分结果，进程非零退出。模拟与真实模式共用数据抽样、提示词、通信图和判分规则。

可选参数：`--baseline single|fixed|both`（默认 both）、`--split train|test`（默认 test）、`--seed`、`--limit`、`--model`、`--temperature`、`--max-tokens`、`--timeout`、`--config`、`--output-dir`。输出目录必须尚不存在，避免覆盖既有实验。默认目录为 `results/gsm8k/<run_id>/`。

已增加学习后拓扑：指定 `--checkpoint` 后，默认比较 single、fixed、learned 三组；也可用 `--baseline learned` 或 `--baseline all`。此时只允许 test 划分，默认使用冻结参数的采样推理，三组每题最多 11 次模型调用。`--learned-mode original-fixed` 可核对原 mask 固定推理规则。详见 [训练、续训与学习后评测](topology-training.md)。

训练集用于后续优化，测试集用于评测。当前尚无训练过程；选择 `--split train` 只评测训练划分，结果会明确标记 train，不作为测试集结果。

## 两组基线与判分

| 基线 | 执行方式 | 每题成功调用数 |
|---|---|---:|
| single | 原 Math Solver 单轮、无前驱消息、直接取回答 | 1 |
| fixed | Math Solver 与 Mathematical Analyst 两轮，再由 FinalRefer 汇总 | 5 |

两组使用同一 `ModelSettings`、同一题目和相同的 Math Solver 提示词来源；多智能体独有的角色、通信和汇总保持原项目规则。默认温度与输出长度省略，沿用提供方默认值；可显式设置。种子仅决定本地抽样，不向提供方发送随机种子。

标准答案只在生成完成后的判分中使用，不传给适配器工厂，也不将该题的标准解题过程或答案拼入模型消息。记录包含文件校验值、划分、原始零基序号及 `gsm8k:test:<index>` 样本编号。

本次迁移默认保留原 `experiments/run_gsm8k.py` 的判分方式：`gsm_get_predict(answer)` 提取后，以 `float(prediction) == float(target)` 比较，规则标识为 `agentdropout_gsm_get_predict_float_v1`。这不是作者仓库 `####` 输出判分器的直接移植。原提取器对负数、小数有局限；无数字文本可能被提取为 0。保留这些行为是为了迁移对照，后续改进判分须用独立规则版本记录。提取异常计为判错，并记录 `scoring_error`，不使整批程序崩溃；模型调用失败也计为判错。

## 保存的记录

- `manifest.json`：数据来源、文件哈希、抽样种子、全部样本编号、模型参数、判分版本、基线配置、提示词来源、Python／库版本、Git 提交和工作区状态，以及源码文件指纹。
- `predictions.jsonl`：每题每组一行，保存题目、标准答案、提取结果、正确性、状态、逐次模型消息／回答／用量／耗时／错误类型和每轮输出。每完成一项立即追加并刷新。
- `summary.json`：逐项更新的汇总，分别记录已尝试／未尝试题数、完成情况、正确率、失败率、判分错误数、总与平均耗时、调用次数、Token 总量和每题平均 Token。

正确率分母为该组所有已尝试题目，包含模型失败和判分失败；中途停止时，顶层状态为 failed，未尝试题数单独列出，不把部分评测称为全量结果。

Token 总量仅在所有尝试都返回用量时给出，否则为 `null`；`known_usage` 是已知小计，不代表完整实际消耗。调用次数指适配器尝试次数，超时不能证明服务端未处理或未计费。耗时包含本地图构建和调度；不同实验的缓存与网络变化也会影响测量。密钥、请求头和异常原文不写入文件。

默认遇错即停，已有记录可检查；当前不自动恢复中断任务。如需重跑，应使用新输出目录，避免把不同运行混成一份实验。
