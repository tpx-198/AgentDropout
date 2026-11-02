# AD-L

使用 LangChain / LangGraph 逐步重建 AgentDropout 的独立项目。

## 开发约定

- 项目目录：`/Users/tpx-198/code/AD-L`。
- Conda 环境：`ad-l`，由 `agentdropout` 克隆后安装新框架依赖。
- 原项目 `/Users/tpx-198/code/AgentDropout` 作为只读参考与实验对照。
- 当前完成环境与目录初始化，尚未迁移模型调用、通信图或剪枝训练。

## 启动与检查

```bash
conda activate ad-l
cd /Users/tpx-198/code/AD-L
python scripts/check_environment.py
python -m pip check
```

检查脚本只执行本地验证，不需要密钥，不调用模型 API。

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

1. 使用 ChatDeepSeek 实现模型适配器，统一读取环境变量。
2. 跑通单个智能体，保留原提示词和生成参数。
3. 验证固定通信图中的轮内和跨轮消息可见性。
4. 接回 PyTorch 拓扑学习和节点、边剪枝，保存训练检查点。
5. 在一致评测条件下与原项目比较正确率、Token、耗时。

后续需要真实 API 时，将 `.env.example` 复制为 `.env` 并填写配置；程序接入时需显式加载 `.env`。当前未复制原项目密钥。

官方文档：
- https://docs.langchain.com/oss/python/langchain/install
- https://docs.langchain.com/oss/python/integrations/chat/deepseek
- https://docs.langchain.com/oss/python/langgraph/overview
