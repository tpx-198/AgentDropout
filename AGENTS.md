# AD-L 工作约定

- 后续 LangChain / LangGraph 重建工作在本目录完成，使用 Conda 环境 `ad-l`。
- 原项目 `/Users/tpx-198/code/AgentDropout` 和原环境 `agentdropout` 用作只读参考；不要为本项目修改它们。
- 未激活环境时使用 `/Users/tpx-198/miniconda3/envs/ad-l/bin/python`。
- LangChain 负责模型和消息组件，LangGraph 负责执行流程，PyTorch 负责拓扑学习与剪枝。
- 模型密钥从环境变量或本地 `.env` 读取；不写入源码、日志、版本记录。
- 默认用本地模拟响应检查流程；真实模型请求按用户授权范围执行，并明确报告是否产生调用。
- 迁移时保留原角色提示词、轮内/跨轮通信可见性和最终汇总规则；变更算法需与迁移验证分开记录。
- PyTorch 的采样 log_prob 保持计算图；不通过 JSON 或流程检查点传递训练中的梯度张量。
- 模型、参数、随机种子、数据划分和 Token 用量随实验保存，基于实测报告结果。
