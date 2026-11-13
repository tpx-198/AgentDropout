# AD-L

使用 LangChain 接入模型，LangGraph 执行通信流程，PyTorch 学习边权重。
默认共 5 个智能体，依次为 Math Solver、Mathematical Analyst、Programming Expert、Inspector、FinalRefer。前 4 个参与讨论，最后一个汇总。默认两轮，每题共 9 次模型调用。

```bash
conda activate ad-l
python -m pip install -r requirements.txt
python run.py
python run.py --agents 8 --rounds 3 --topology random --seed 42
python run.py --topology layered --layers 2 --async
```

`--agents` 包含汇总者。增加数量时循环分配四种讨论角色，`--roles "Math Solver" "Mathematical Analyst" "Programming Expert" "Inspector"` 可指定各讨论节点角色。

`--topology` 提供 `full`、`random`、`layered`、`chain`、`star`、`debate`。默认 `full`。随机图使用 `--density` 和 `--seed` 控制候选连接。分层图只连接相邻层，链式依次相连，星形由第一个讨论节点向其他节点发送，辩论式只进行跨轮通信。

`--temporal` 独立选择 `upper`、`full`、`self`、`none`，默认 `upper`，即编号较小节点向下一轮编号较大节点发送。第一轮没有跨轮输入。每轮独立保存回答快照，最终汇总只读取末轮中实际连接的讨论节点。

默认边权重为 0.5，通过 `sigmoid(logit)` 保持在 0～1。每轮轮内边、相邻轮间边及最终汇总边分别学习，矩阵行表示发送者，列表示接收者。`--mode sampled` 按权重进行伯努利采样，并过滤会形成轮内环路的边。`fixed` 启用所有尚未剪除的候选边并过滤环路，`threshold` 仅选择达到 `--threshold` 的边。`--sampling-temperature` 控制边采样温度。

全连接、随机、分层拓扑和边采样参考论文第 3.3 节与附录 A.4。链式、星形和辩论式用于扩展实验。[AgentDropout 论文](https://aclanthology.org/2025.acl-long.1170.pdf)

训练入口默认混合本地 GSM8K、AQuA、SVAMP 和 MultiArith 的 train 数据，以最终答案是否正确作为奖励，执行 REINFORCE，再按各轮矩阵内最低权重剪除指定比例的候选边。数值题比较数值答案，AQuA 比较选项字母，提示词随题型调整。

```bash
python train.py --steps 3 --batch-size 2 --limit 6 --prune-rate 0.25
python train.py --datasets gsm8k aqua --limit 10 --output checkpoints/gsm8k-aqua.pt
python train.py --datasets svamp --limit 20 --output checkpoints/svamp.pt
python run.py --checkpoint checkpoints/topology.pt
python run.py --checkpoint checkpoints/topology.pt --mode threshold --threshold 0.5
python run.py --mode fixed --output results/run.json
```

`--datasets` 选择训练来源，默认使用四类数据。`--limit` 表示每个数据集的抽样题数，默认每类 6 题，组成 24 题样本池，按数据集轮流取题。实际训练题次为 `--steps × --batch-size`，超过样本池大小时循环使用。随机种子控制样本选择。原有 `--data data/gsm8k/train.jsonl` 单文件方式继续可用，与 `--datasets` 互斥。

数据目录同时提供训练与评测文件。GSM8K 和 AQuA 沿用官方划分，AQuA 还提供 validation。SVAMP 和 MultiArith 使用固定种子 0、按相同题目分组的 80%/20% 本地实验划分。训练入口检查 split 和数据指纹，拒绝 test、validation 文件。各数据集的来源、规模和划分说明见 [数据说明](data/README.md)。

```bash
python prepare_data.py
python train.py --datasets gsm8k aqua svamp multiarith --plan
```

`prepare_data.py` 下载缺失的数据并保存来源 URL、原文件与输出文件的 SHA-256。已经准备好的数据只进行本地校验。

剪枝数量为每张矩阵有效边数乘 `--prune-rate` 后向下取整，剪除边的权重为 0，后续训练不会重新启用。检查点保存参数、配置、数据指纹、抽样清单、提示词来源、逐步运行记录和模型用量。每步更新后保存，模型调用失败时保留之前成功更新的权重。`--output` 指定新路径以开展独立实验。

默认使用固定模拟回答，模拟运行只验证流程，无法衡量模型准确率或剪枝收益。`--sparsity` 控制概率矩阵核范数与汇总权重均值的惩罚项。可用 `--sparsity 0` 单独研究策略梯度。

本实现保留原项目的角色与示例提示词，通信消息统一传入可见回答原文。Programming Expert 生成的代码作为文本参与讨论。汇总输入边也参与学习，属于本项目的扩展。训练目标采用概率矩阵正则化，这些实现选择与论文完整的两阶段节点及边剪枝实验有区别。剪边作用于消息输入，默认五节点两轮运行仍有 9 次模型调用。

真实解题和任务驱动训练需要大模型 API。将 `.env.example` 复制为 `.env`，填写 `DEEPSEEK_API_KEY` 和账户可用的 `DEEPSEEK_MODEL`，再显式启用真实请求。已有环境变量优先于 `.env`。

```bash
python run.py --deepseek --plan
python run.py --deepseek --task "What is 3 + 2?"
python train.py --deepseek --steps 3 --batch-size 2 --limit 6 --plan
```

`--plan` 仅显示规模，不创建模型客户端。默认单题最多 9 次真实请求，上述训练最多 54 次。超时默认 60 秒，自动重试为 0。真实 Token 用量来自提供方，模拟用量为 `null`。

核心代码位于 `ad_l/`，运行入口为 `run.py`，训练入口为 `train.py`。所有测试脚本独立放在 `tests/`，可以整目录删除。

```bash
python -m unittest discover -s tests -v
```
