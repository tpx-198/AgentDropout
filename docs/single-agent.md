# 模型适配器与第一个智能体

## 两者分别做什么

模型适配器是智能体与具体模型之间的统一入口。智能体向它提交 LangChain 的 `SystemMessage`、`HumanMessage`，拿回 `AIMessage`。本地模拟与 DeepSeek 使用同一调用接口，支持 `invoke` 和 `ainvoke`；模型调用本身交给 LangChain 的聊天模型组件。模拟后端无需密钥，真实后端通过环境变量或指定的本地 `.env` 读取密钥，结果记录不保存密钥或完整客户端配置。

单智能体流程负责把“题目变成回答”的步骤串起来。当前 LangGraph 中有三个节点：

1. `prepare`：将题目放入原项目 GSM8K Math Solver 的示例与提示语，生成角色消息和用户消息。
2. `generate`：通过模型适配器生成一次回答。
3. `finish`：保留完整文本和提供方返回的用量，不额外调用模型汇总。

例如输入“Anna 有 3 个苹果，又买了 2 个”，离线模型返回预设的演示文本；将后端切换为真实模型时，消息组织和执行流程保持相同。离线模拟只验证程序运行和信息传递，不能证明模型会答对。

## 本次迁移范围

- 原类：`AgentDropout/agents/math_solver.py` 中的 `MathSolver`。
- 原提示词：`AgentDropout/prompt/gsm8k_prompt_set.py` 中 `Math Solver` 角色。
- 固定一个角色、一个智能体、一轮，无空间前驱与时间前驱，不自动循环分配角色。
- 系统提示词、示例和消息拼接逐字保留，包括空前驱时的 `(Hint: The answer is near to).`。这不是新增提示词设计。
- 提示词保存在 `src/ad_l/resources/gsm8k_math_solver.json`，附原文件 SHA-256；正常运行不依赖原项目目录。
- 对照测试仅抽取原提示词常量和 `_process_inputs` 方法执行，不导入原项目模型模块，也不读取原密钥。
- 回答原样返回，不执行答案提取、代码执行或多智能体最终汇总。这是单节点迁移检查，尚未完成原 `Graph.run/arun` 的完整行为对齐。

原模型封装定义了 `temperature=0.2`、`max_tokens=1000`，但底层聊天请求未传递它们，因此本实现默认均为 `None`，实际沿用提供方默认值。更换模型提供方本身可能改变生成结果，提示词相同不等于模型输出等价。单节点入口的超时为 60 秒、自动重试为零，与原图及原请求的超时／重试不同；后续完整图迁移需单独核对。没有改动拓扑学习或剪枝算法。

## 结果与复核

```bash
conda activate ad-l
python scripts/run_single_agent.py --output results/single_agent/demo.json
PYTHONPATH=src python -m unittest discover -s tests -v
```

JSON 保存后端、模型名称、显式生成参数、完整输入消息、原始回答、耗时、相关库版本和提示词来源。`local_seed` 只设定本地 Python 随机数种子；`provider_seed=null` 表示未向模型提供方传种子，不承诺真实生成结果可重复。

`data_split=manual` 表示手工题目。`usage=null` 配合 `usage_source=mock_not_measured` 表示没有测量 Token；真实后端没有返回用量时标为 `provider_unavailable`。当前配置无重试，成功运行有一次模型调用；模拟运行的 `real_api_calls=0`。

离线测试覆盖原提示词逐字一致、同步与异步流程、调用次数、运行间消息隔离、回答与用量透传、真实后端显式启用和缺失密钥处理、环境变量优先级、模拟模式不加载 `.env` 以及命令行记录。原项目目录不在时，只跳过本地原代码对照测试。

## 文件分工

- `src/ad_l/models.py`：模型配置与调用适配。
- `src/ad_l/prompts.py`：当前角色的消息构造。
- `src/ad_l/single_agent.py`：LangGraph 流程。
- `scripts/run_single_agent.py`：运行和保存记录。
- `tests/test_single_agent.py`：离线验证与原项目对照。

实现参照 [ChatDeepSeek 官方集成说明](https://docs.langchain.com/oss/python/integrations/chat/deepseek) 和 [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)，并在本地锁定版本环境中运行验证。
