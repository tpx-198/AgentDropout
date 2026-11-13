# 推理数据集

拓扑训练默认混合以下四个数据集的 train 部分。数值题和选择题分别判分，标准答案保留在训练器中计算奖励，模型接收题目、选项和可见回答。

| 数据集 | train | validation | test | 答案类型 |
| --- | ---: | ---: | ---: | --- |
| GSM8K | 7473 | — | 1319 | 数值 |
| AQuA | 97467 | 254 | 254 | A～E 选项 |
| SVAMP | 800 | — | 200 | 数值 |
| MultiArith | 479 | — | 121 | 数值 |

GSM8K 和 AQuA 使用官方划分。SVAMP 的完整题库为 1000 题，MultiArith 为 600 题。本项目将这两个完整题库按规范化问题分组，以随机种子 0 划出约 80% 的训练题组和 20% 的评测题组，相同题目不会跨训练与评测两侧。MultiArith 有重复题，分组后得到 479/121 题。论文中的完整测试集评测分数不能直接与这两个本地划分的分数比较。

AQuA 的训练数据会剔除与其官方 validation、test 完全相同的问题及选项组合，本次剔除数为 0。跨文件分组检查针对规范化文本完全相同的题目。

来源和授权信息分别保存在各子目录的 `manifest.json` 和上游提供的 `LICENSE` 文件中。manifest 同时记录下载 URL、原始文件指纹、输出文件指纹、题数和划分方式。原有 GSM8K 文件保留原始 JSONL 格式，新增文件统一为 `id`、`dataset`、`split`、`question`、`target`、`answer_type` 和可选 `choices` 字段。AQuA 的 rationale 不进入模型消息。

数据集选择参考 AgentDropout 的数学推理实验。[AgentDropout 论文](https://aclanthology.org/2025.acl-long.1170.pdf)

- GSM8K 数据来自作者仓库。[OpenAI grade-school-math](https://github.com/openai/grade-school-math)
- AQuA 数据来自作者仓库。[Google DeepMind AQuA](https://github.com/google-deepmind/AQuA)
- SVAMP 数据来自作者仓库。[SVAMP](https://github.com/arkilpatel/SVAMP)
- MultiArith 使用 LLM-Adapters 研究项目维护的原始数据镜像。[MultiArith 文件](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/dataset/MultiArith/MultiArith.json)

在项目根目录运行 `python prepare_data.py` 可重新准备缺失的 AQuA、SVAMP 和 MultiArith 数据。已有文件会先校验，新的实验划分可通过 `--seed` 和独立的 `--output-dir` 创建。训练时使用 `--data-dir` 选择对应目录。
