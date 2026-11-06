# AD-L

使用 LangChain / LangGraph 逐步重建 AgentDropout 的独立项目。

## 开发约定

- 项目目录：`/Users/tpx-198/code/AD-L`。
- Conda 环境：`ad-l`，由 `agentdropout` 克隆后安装新框架依赖。
- 原项目 `/Users/tpx-198/code/AgentDropout` 作为只读参考与实验对照。
- 已完成 GSM8K 批量基线、共享及逐轮独立拓扑训练与恢复、周期及独立边剪枝、节点选择训练和剪枝配对评测。

## 安装与环境

环境来源、安装、版本快照恢复和检查步骤统一见 [环境说明](docs/environment.md)。完成安装后，在 `ad-l` 环境、项目根目录运行下面的示例。

## 运行第一个智能体

```bash
conda activate ad-l
python scripts/run_single_agent.py
python -m unittest discover -s tests -v
```

默认例题为 3 个苹果再买 2 个。模拟模型固定返回带 `[MOCK]` 标识的演示答案；它不计算题目，也不能用来评测正确率。替换题目时可用 `--task`，替换模拟回答时可用 `--mock-response`。

流程为：`START → prepare（原角色提示词）→ generate（模型适配器）→ finish（原样返回回答）→ END`。

每次运行在 `results/single_agent/` 保存 JSON，包含输入消息、角色、提示词来源指纹、模型参数、回答、耗时、版本、随机种子和用量。模拟 Token 用量保存为 `null`，真实用量取自提供方返回；手工输入标为 `manual`，不属于基准数据集划分。

真实调用需先在本地 `.env` 设置模型和密钥，再由操作者显式运行：

```bash
python scripts/run_single_agent.py --backend deepseek --allow-real-api
```

此命令会发送真实模型请求。环境变量优先于 `.env`；离线模式不读取 `.env`，演示入口关闭外部追踪。可通过 `--model`、`--temperature` 和 `--max-tokens` 显式指定参数。原请求未传温度与长度，默认继续省略这两个参数；不是将原封装内未生效的 `0.2 / 1000` 直接当作请求参数。真实后端采用 60 秒超时、零自动重试，属于单智能体入口的运行策略，未宣称与原重试机制一致。

更多说明见 [单智能体迁移说明](docs/single-agent.md)。

## 两个智能体、两轮固定图

```bash
python run.py
python run.py --async
python -m unittest discover -s tests -v
```

此入口默认本地模拟；模拟模式不读取 `.env`。A 使用原 GSM8K `Math Solver` 提示词，B 使用 `Mathematical Analyst` 提示词，每轮按固定图执行，最后由 `FinalRefer` 汇总第二轮的两份回答，共 5 次模型调用。固定图支持分别注入三个节点的模型适配器，同步与异步共用提示词和通信规则。

需要使用 DeepSeek 时，在根目录本地 `.env` 中填写 `.env.example` 的 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`，然后显式运行：

```bash
python run.py --deepseek
```

`--deepseek` 同时选择 DeepSeek 并显式允许真实请求，完整成功运行包含 5 次调用。结果自动保存到 `results/fixed_graph/`，每次使用独立文件；需要指定路径时添加 `--output`。详细参数、注入方式和失败记录见 [固定图说明](docs/fixed-graph.md)。

预设回答与连接矩阵可在 [离线配置](configs/fixed_graph_offline.json) 中修改，或通过 `--config` 指定其他配置。修改题目时使用 `--task`；预设回答不会自动求解新题目。

每份结果包含逐次调用的完整消息、轮内／跨轮可见输入、每轮输出、最终回答、配置矩阵和实际生效连接。原项目的固定时间边也受空间环路检查限制，本版保留该行为。具体可见性与迁移验证见 [固定图说明](docs/fixed-graph.md)。

## GSM8K 批量评测

数据准备与离线检查：

```bash
python scripts/prepare_gsm8k.py
python evaluate.py
```

默认在测试集中固定抽取 10 题，对比单智能体与两轮固定图；模拟结果仅用于流程诊断。真实批量评测使用：

```bash
python evaluate.py --deepseek --limit 10
```

该命令使用 `.env` 中的 DeepSeek 配置，最多产生 60 次真实调用；添加 `--plan` 可只查看调用计划。结果自动保存到 `results/gsm8k/`，包含逐题记录与汇总。数据来源、判分规则和指标说明见 [GSM8K 批量评测](docs/gsm8k-evaluation.md)。

## GSM8K 多步拓扑训练与评测

```bash
python train.py --steps 3 --output-dir results/topology/gsm8k-demo
python evaluate.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt --limit 10
```

默认使用 GSM8K train 数据和模拟响应进行 3 步训练，每步 2 题；学习后评测冻结参数，在同一批 test 题上比较单智能体、固定图和学习后采样图。训练记录和可恢复检查点保存到 `results/topology/`。添加 `--resume 检查点 --steps 目标总步数` 可继续训练。真实训练或评测须显式添加 `--deepseek`，可先用 `--plan` 查看规模；上例真实训练最多 30 次、评测最多 110 次模型调用。模拟结果只验证流程。推理规则及恢复约束详见 [拓扑学习说明](docs/topology-training.md)。

## 边剪枝与配对评测

训练中周期剪枝可直接使用：

```bash
python train.py --steps 6 --prune-every 2 --prune-until 4
```

默认模拟，在第 2、4 步更新后剪枝，其余步继续训练；检查点保存剪枝进度，恢复时不重复或漏剪。详见 [周期剪枝与恢复](docs/topology-training.md)。

需要每轮学习各自的通信拓扑时，添加 `--diff`：

```bash
python train.py --diff --steps 6 --prune-every 2 --prune-until 4
```

两轮空间参数与跨轮时间参数独立，周期剪枝也按每张矩阵分别执行。续训自动继承模式，现有评测、独立剪枝入口自动识别检查点。默认两节点配置下，单独剪边后每题仍为 5 次调用。正则约束始终参照初始配置，与原脚本一致。详见 [每轮独立拓扑](docs/topology-training.md)。

已有检查点的独立剪枝和配对评测：

```bash
python prune.py --checkpoint results/topology/gsm8k-demo/checkpoint.pt
```

默认按原规则剪除低权重候选边，在同一批 10 道 test 题上用相同的固定推理模式对比剪枝前后。默认模拟；添加 `--deepseek` 明确允许最多 100 次真实调用，`--plan` 只查看计划，`--prune-only` 只保存剪枝产物。结果保存在 `results/pruning/`，原检查点保持只读。详见 [边剪枝说明](docs/edge-pruning.md)。

## 节点剪枝

```bash
python nodes.py
```

默认使用模拟响应，在 GSM8K train 上训练独立的节点选择权重，再在同一批 test 题上比较剪枝前后。每轮跳过一名智能体，保留最终汇总，每题调用从 5 次降为 3 次。结果写入 `results/node_pruning/`。选择规则、运行参数与两节点限制见 [节点剪枝说明](docs/node-pruning.md)。

使用 `python train.py --node-pruning 路径/pruning.json` 可在剪枝后的候选图上训练边权重。检查点保留节点跳过计划，续训、学习后评测与独立边剪枝自动继承。

三节点示例加入原项目的 Inspector：

```bash
python nodes.py --config configs/three_agents_offline.json --initial-probability 0.3
```

两轮完整运行每题调用 7 次，逐轮剪去一个节点后每题调用 5 次。三节点配置及边训练命令见 [节点剪枝说明](docs/node-pruning.md)。

添加 `--compare-random` 可在同一批测试题上对照完整图、随机剪枝与学习后剪枝。随机与学习后两组的调用预算相同，可用 `--plan` 查看训练和各组的调用上限。

## 依赖记录

- [pyproject.toml](pyproject.toml)：项目依赖与版本约束的统一声明，包括 LangChain、LangGraph、PyTorch 及原注册组件的兼容依赖；通过 `pip install -e .` 安装。
- [requirements-lock.txt](requirements-lock.txt)：完整 Python 包版本快照及本地项目安装入口，面向 macOS ARM64 / Python 3.10。
- [docs/environment.md](docs/environment.md)：环境来源、安装方式和依赖维护说明。

## 目录

- `src/ad_l/`：后续模型适配、智能体、通信调度与剪枝实现。
- `src/ad_l/resources/prompts/`：提示词资源，包含单智能体与多智能体协作提示词。
- `configs/`：非敏感实验配置。
- `scripts/`：检查及运行入口。
- `data/`：后续准备的数据。
- `results/`：后续实验输出。
- `docs/`：环境记录与设计说明。

## 下一步

1. 按明确的样本规模运行真实 GSM8K 基线，扩展固定图规模与其余角色。
2. 用真实模型和多个随机计划评估三节点剪枝的准确率与 Token 用量。
3. 在一致评测条件下与原项目比较正确率、Token、耗时。

需要真实 API 时，将 `.env.example` 复制为 `.env` 并填写配置；运行入口仅在显式选择真实后端后加载该文件。未复制原项目密钥。

官方文档：
- https://docs.langchain.com/oss/python/langchain/install
- https://docs.langchain.com/oss/python/integrations/chat/deepseek
- https://docs.langchain.com/oss/python/langgraph/overview
