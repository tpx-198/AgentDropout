# AD-L

使用 LangChain / LangGraph 逐步重建 AgentDropout 的独立项目。

## 开发约定

- 项目目录：`/Users/tpx-198/code/AD-L`。
- Conda 环境：`ad-l`，由 `agentdropout` 克隆后安装新框架依赖。
- 原项目 `/Users/tpx-198/code/AgentDropout` 作为只读参考与实验对照。
- 已完成模型适配器与 GSM8K Math Solver 单轮流程，支持离线模拟和显式启用的 DeepSeek 后端；多智能体通信图与剪枝训练尚未迁移。

## 启动与检查

```bash
conda activate ad-l
cd /Users/tpx-198/code/AD-L
python scripts/check_environment.py
python -m pip check
```

检查脚本只执行本地验证，不需要密钥，不调用模型 API。

在已准备好的 `ad-l` 环境中，将本项目注册为可编辑包（不下载或更新依赖）：

```bash
python -m pip install -e . --no-deps --no-build-isolation --no-index
```

安装后，Python 可直接导入 `ad_l`，修改源码即时生效，无需为每次运行设置 `PYTHONPATH`。如果 VS Code 打开的是父目录 `/Users/tpx-198/code`，`AD-L/.vscode/` 中的配置不会作为该窗口的工作区配置生效；选择 `ad-l` 解释器并刷新窗口即可重新读取已安装的本地包，也可以单独打开 AD-L 文件夹。

## 运行第一个智能体

```bash
conda activate ad-l
python scripts/run_single_agent.py
PYTHONPATH=src python -m unittest discover -s tests -v
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

## 依赖记录

- `requirements-langchain.txt`：在克隆环境上安装的框架与兼容性依赖。
- `requirements-lock.txt`：完成安装后的 Python 包版本快照，面向当前 macOS / Python 3.10 环境。
- `docs/source-environment.json`：原 agentdropout 环境的包版本快照。
- `docs/environment.json`：克隆来源、安装结果与验证记录。

重新创建同类环境时，先克隆原环境，再安装框架依赖并检查：

```bash
conda create -n ad-l --clone agentdropout
conda activate ad-l
python -m pip install -r requirements-langchain.txt
python -m pip check
```

## 目录

- `src/ad_l/`：后续模型适配、智能体、通信调度与剪枝实现。
- `configs/`：非敏感实验配置。
- `scripts/`：检查及运行入口。
- `data/`：后续准备的数据。
- `results/`：后续实验输出。
- `docs/`：环境记录与设计说明。

## 下一步

1. 扩展智能体角色，迁移最终汇总节点。
2. 验证固定通信图中的轮内和跨轮消息可见性。
3. 接回 PyTorch 拓扑学习和节点、边剪枝，保存训练检查点。
4. 在一致评测条件下与原项目比较正确率、Token、耗时。

需要真实 API 时，将 `.env.example` 复制为 `.env` 并填写配置；运行入口仅在显式选择真实后端后加载该文件。未复制原项目密钥。

官方文档：
- https://docs.langchain.com/oss/python/langchain/install
- https://docs.langchain.com/oss/python/integrations/chat/deepseek
- https://docs.langchain.com/oss/python/langgraph/overview
