# AD-L

使用 LangChain / LangGraph 逐步重建 AgentDropout 的独立项目。

## 开发约定

- 项目目录：`/Users/tpx-198/code/AD-L`。
- Conda 环境：`ad-l`，由 `agentdropout` 克隆后安装新框架依赖。
- 原项目 `/Users/tpx-198/code/AgentDropout` 作为只读参考与实验对照。
- 已完成模型适配器、GSM8K Math Solver 单轮流程，以及两个智能体、两轮固定通信图和 FinalRefer 最终汇总的离线版本；拓扑学习与剪枝训练尚未迁移。

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
python scripts/run_fixed_graph.py --output results/fixed_graph/demo.json
python scripts/run_fixed_graph.py --async
python -m unittest discover -s tests -v
```

此入口仅允许本地模拟，不读取 `.env`，不创建真实模型客户端。A 使用原 GSM8K `Math Solver` 提示词，B 使用 `Mathematical Analyst` 提示词，每轮按固定图执行，最后由 `FinalRefer` 汇总第二轮的两份回答，共 5 次模拟调用。

预设回答与连接矩阵可在 [离线配置](configs/fixed_graph_offline.json) 中修改，或通过 `--config` 指定其他配置。修改题目时使用 `--task`；预设回答不会自动求解新题目。

每份结果包含逐次调用的完整消息、轮内／跨轮可见输入、每轮输出、最终回答、配置矩阵和实际生效连接。原项目的固定时间边也受空间环路检查限制，本版保留该行为。具体可见性与迁移验证见 [固定图说明](docs/fixed-graph.md)。

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

1. 扩展固定图规模与其余角色，按授权范围建立真实模型基线。
2. 接回 PyTorch 拓扑学习和节点、边剪枝，保存训练检查点。
3. 在一致评测条件下与原项目比较正确率、Token、耗时。

需要真实 API 时，将 `.env.example` 复制为 `.env` 并填写配置；运行入口仅在显式选择真实后端后加载该文件。未复制原项目密钥。

官方文档：
- https://docs.langchain.com/oss/python/langchain/install
- https://docs.langchain.com/oss/python/integrations/chat/deepseek
- https://docs.langchain.com/oss/python/langgraph/overview
