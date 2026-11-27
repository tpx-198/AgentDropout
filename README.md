# AD-L

使用 LangChain 接入模型，LangGraph 执行通信流程，PyTorch 学习边权重。
默认共 5 个智能体，依次为 Math Solver、Mathematical Analyst、Programming Expert、Inspector、FinalRefer。前 4 个参与讨论，最后一个汇总。默认两轮，每题共 9 次模型调用。可在「实验工作台」或「训练与评测」设置 2～32 个初始智能体，总数包含 1 个汇总节点。

日常实验可直接双击项目中的 `启动评测平台.command`。平台会启动本机服务并打开浏览器，默认地址为 `http://127.0.0.1:8765`，使用已有的 `ad-l` 环境。再次双击会打开已经运行的平台。

- 实验工作台设置智能体数量、讨论轮数、拓扑、边权重与随机种子，并预览通信图。
- 智能体与模型页面可为每个节点自定义角色名称和职责，并设置服务类型、模型名称、API 地址和密钥。也可按当前团队的节点 ID 导入 `configs/agents.json` 与本地 `.env` 中对应的 API 配置。
- 训练与评测页面选择数据集与参数，点击按钮启动任务，查看进度、损失和逐题结果，并可取消任务。
- 测试集评测可选择已训练检查点，应用其团队设置，与初始拓扑进行同题、同模式对照。
- 运行记录保存实验状态、训练检查点和完整调用记录，支持浏览器下载结果 JSON 和检查点。

平台默认使用本地模拟。使用真实模型时，在页面保存各节点配置与密钥，再勾选本次任务允许真实 API 调用。运行计划会展示调用上限。模拟或混合后端的结果标为诊断指标，全部为真实模型时计算评测正确率。

平台设置、密钥和实验记录位于 `platform_state/`，该目录已加入 Git 忽略规则。密钥以本地明文文件保存，目录权限为 `0700`，设置文件权限为 `0600`。页面返回密钥是否已保存的状态，实验导出包含模型参数及调用记录。平台绑定本机回环地址，适用于本机单用户实验。取消任务会等待当前模型请求结束，并保留成功训练步骤的检查点。

```bash
conda activate ad-l
python -m pip install -r requirements.txt
python run.py
python run.py --agents 8 --rounds 3 --topology random --seed 42
python run.py --topology layered --layers 2 --async
```

`run.py` 和 `train.py` 都可用 `--agents` 设置初始人数，包含汇总者。增加数量时循环分配四种讨论角色示例，`--roles "Math Solver" "Mathematical Analyst" "Programming Expert" "Inspector"` 可为各讨论节点选择默认示例。

### 自定义角色职责

在 **智能体与模型** 中，打开节点卡片，填写「角色名称」和「角色职责」，再点击「保存智能体配置」。每个讨论节点和 `final` 都可以独立编辑。填写职责会替换该节点的内置系统提示词和 few-shot 示例。清空职责即恢复默认示例。角色名称显示在拓扑图与调用记录中。

系统按拓扑给节点传入题目和可见回答，`final` 负责提交团队的最终答案。训练或评测时，自定义 `final` 会自动收到当前题型的判分格式。应用检查点会载入当时保存的职责，也可以在同一拓扑上修改职责后另跑一组评测。

命令行用 `--role-prompts` 指定一个按节点 ID 编写的 JSON 文件。`name` 和 `instructions` 均为可选，省略的节点使用默认示例。可将以下内容保存为 `roles.json`。

```json
{
  "agent_1": {"name": "任务分析员", "instructions": "识别题目目标与约束，列出需要解决的问题。"},
  "agent_2": {"name": "证据核验员", "instructions": "根据题目和已有回答核验证据，指出有依据的结论和信息缺口。"},
  "final": {"name": "结论编辑", "instructions": "整理可见回答，给出清晰结论，并说明影响结论的不确定性。"}
}
```

```bash
python run.py --agents 3 --role-prompts roles.json --task "比较两种课程安排，给出选择依据。"
python train.py --agents 3 --role-prompts roles.json --datasets gsm8k --output checkpoints/custom-roles.pt
python run.py --checkpoint checkpoints/custom-roles.pt
```

`run.py --checkpoint` 默认恢复保存的职责。显式传入 `--role-prompts` 时使用指定文件的配置。

`--topology` 提供 `full`、`random`、`layered`、`chain`、`star`、`debate`。默认 `full`。随机图使用 `--density` 和 `--seed` 控制候选连接。分层图只连接相邻层，链式依次相连，星形由第一个讨论节点向其他节点发送，辩论式只进行跨轮通信。

`--temporal` 独立选择 `upper`、`full`、`self`、`none`，默认 `upper`，即编号较小节点向下一轮编号较大节点发送。第一轮没有跨轮输入。每轮独立保存回答快照，最终汇总只读取末轮中实际连接的讨论节点。

默认边权重为 0.5，通过 `sigmoid(logit)` 保持在 0～1。每轮轮内边、相邻轮间边及最终汇总边分别学习，矩阵行表示发送者，列表示接收者。`--mode sampled` 按权重进行伯努利采样，并过滤会形成轮内环路的边。`fixed` 启用所有尚未剪除的候选边并过滤环路，`threshold` 仅选择达到 `--threshold` 的边。`--sampling-temperature` 控制边采样温度。

全连接、随机、分层拓扑和边采样参考论文第 3.3 节与附录 A.4。链式、星形和辩论式用于扩展实验。[AgentDropout 论文](https://aclanthology.org/2025.acl-long.1170.pdf)

### 节点剪枝

在 **训练与评测 → 拓扑训练** 设置「每轮剪除节点数」。训练先更新完整团队的边权，结束后按各轮节点相连的有效边平均权重从低到高剪节点，再按配置剪剩余边。分数相同按节点编号排序。每轮至少保留一个讨论节点，最终汇总者始终执行。检查点记录每轮保留的节点，并可直接用于评测。

**测试集评测**中的「每轮额外剪除节点数」对当前初始拓扑或已选检查点生效。设置为 0 可直接使用检查点已有的节点掩码。拓扑图按轮次标出被剪节点，运行预算按各节点实际参与的轮数计算。默认四个讨论节点、两轮，每轮剪一个后单题调用从 9 次降到 7 次。

```bash
python run.py --node-prune-count 1 --plan
python train.py --datasets gsm8k --node-prune-count 1 --prune-rate 0 --output checkpoints/node-only.pt
python run.py --checkpoint checkpoints/node-only.pt
```

设置节点剪枝数或边剪枝比例为 0，即可做不剪枝、只剪边、只剪节点和节点加边剪枝的对照。旧检查点载入时默认每个讨论节点在每轮均可执行。

### 训练与数据

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

可额外导入 **APTBench**。准备完成后，打开 **训练与评测 → 测试集评测**，勾选所需套件并设置题数。APTBench 的四个套件只用于评测。 [APTBench 数据说明](data/README.md#aptbench)

```bash
python prepare_data.py --datasets aptbench_swe aptbench_dr aptbench_tool aptbench_math
```

剪枝数量为每张矩阵有效边数乘 `--prune-rate` 后向下取整，剪除边的权重为 0，后续训练不会重新启用。检查点保存参数、配置、数据指纹、抽样清单、提示词来源、逐步运行记录和模型用量。每步更新后保存，模型调用失败时保留之前成功更新的权重。`--output` 指定新路径以开展独立实验。

默认使用固定模拟回答，模拟运行只验证流程，无法衡量模型准确率或剪枝收益。`--sparsity` 控制概率矩阵核范数与汇总权重均值的惩罚项。可用 `--sparsity 0` 单独研究策略梯度。

平台在测试集评测前，将默认系统提示词、few-shot 示例和自定义角色文本与所选数据集的完整测试集比较。发现提示词含有测试题干时会阻止启动，并显示节点或角色和样本 ID。匹配会归一化大小写、Unicode 和空白字符。通信消息统一传入可见回答原文。汇总输入边也参与学习，训练目标采用概率矩阵正则化。节点和边都在同一次训练后剪枝，论文另设剪节点后重新训练边权的阶段。

真实解题和任务驱动训练需要大模型 API。将 `.env.example` 复制为 `.env`，填写 `DEEPSEEK_API_KEY` 和账户可用的 `DEEPSEEK_MODEL`，再显式启用真实请求。已有环境变量优先于 `.env`。

```bash
python run.py --deepseek --plan
python run.py --deepseek --task "What is 3 + 2?"
python train.py --deepseek --steps 3 --batch-size 2 --limit 6 --plan
```

`--plan` 仅显示规模，不创建模型客户端。默认单题最多 9 次真实请求，上述训练最多 54 次。超时默认 60 秒，自动重试为 0。真实 Token 用量来自提供方，模拟用量为 `null`。

按智能体分别配置 API 时，编辑 `configs/agents.json`。默认映射为 `agent_1` 数学求解、`agent_2` 数学分析、`agent_3` 编程专家、`agent_4` 检查员、`final` 最终汇总者。每个节点分别创建模型客户端，可以使用不同的模型、服务地址和密钥。

每个节点可填写 `backend`、`base_url`、`model_env`、`api_key_env`、`temperature`、`max_tokens` 和 `timeout`。`backend` 支持 `deepseek`、`openai` 和 `mock`，其中 `openai` 使用 OpenAI 兼容协议。`base_url` 填写服务提供方的 API 根地址。生成参数为 `null` 时采用提供方默认值。

`model_env` 和 `api_key_env` 填环境变量名称，实际模型名称和密钥填写到本地 `.env`，各节点字段见 `.env.example`。例如 `AGENT_1_MODEL` 和 `AGENT_1_API_KEY` 对应第一个智能体。模型名称也可以在 JSON 中通过 `model` 直接填写，此时删除同节点的 `model_env`。多个节点可以引用同一个密钥变量名以复用账户。

```bash
python run.py --api-config configs/agents.json --plan
python run.py --api-config configs/agents.json --allow-real-api
python train.py --api-config configs/agents.json --allow-real-api --output checkpoints/per-agent.pt
```

配置文件须覆盖当前团队的全部节点，扩展智能体数量时同步添加对应的 `agent_5` 等条目，并保留 `final`。使用配置文件时，将模型生成参数填在文件内。`--plan` 可在尚未填写密钥时校验结构并显示按节点计算的调用预算。实际请求通过 `--allow-real-api` 启用，已有 `--deepseek` 方式继续可用。

运行记录和训练检查点按节点保存模型参数、服务地址及配置文件指纹。密钥值留在环境变量中。混合模拟和真实模型的训练标记为 `mixed_backend_diagnostic`。

核心代码位于 `ad_l/`，运行入口为 `run.py`，训练入口为 `train.py`。所有测试脚本独立放在 `tests/`，可以整目录删除。

```bash
python -m unittest discover -s tests -v
```
