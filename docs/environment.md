# 环境来源与安装说明

本文统一记录环境来源、安装方式和依赖维护规则。初始化历史来自 2026-11-02，当时仅完成环境和项目骨架搭建；初始化依赖变化与检查结果保留为历史记录。

## 环境来源

- 平台：macOS ARM64，Python 3.10.21。
- 项目目录：`/Users/tpx-198/code/AD-L`。
- 开发环境：`ad-l`，路径为 `/Users/tpx-198/miniconda3/envs/ad-l`。
- 克隆来源：`/Users/tpx-198/miniconda3/envs/agentdropout`；原环境的包版本在初始化过程中保持不变。
- 原项目 `/Users/tpx-198/code/AgentDropout` 作为只读参考。

## 初始化依赖变化

从原环境的 73 个 Python 包出发，新增 22 个包、更新 2 个包，形成 95 个包的初始化快照。新增部分主要包括 LangChain、LangGraph、DeepSeek 集成及其依赖。

| 关键包 | 原环境 | 初始化后的 AD-L |
|---|---|---|
| langchain | 未安装 | 1.4.7 |
| langchain-core | 未安装 | 1.7.3 |
| langchain-deepseek | 未安装 | 1.1.1 |
| langgraph | 未安装 | 1.2.19 |
| openai | 1.58.1 | 3.42.1 |
| certifi | 2026.7.22 | 2026.10.23 |
| torch | 2.5.1 | 2.5.1 |
| class-registry | 2.1.2 | 2.1.2 |
| setuptools | 69.1.1 | 69.1.1 |

兼容性处理：`class-registry 2.1.2` 仍依赖 `pkg_resources`，因此保留原环境可用的 `setuptools 69.1.1`。克隆遗留的 `setuptools 83.0.0` 重复 egg-info 当时已移至临时备份。

## 初始化检查结果

`pip check`、LangChain 本地消息链、LangGraph 本地流程、DeepSeek 与 class-registry 导入、PyTorch 自动求导检查均通过。检查未发送模型 API 请求，未复制原项目密钥。

## 依赖文件分工

- [pyproject.toml](../pyproject.toml)：项目直接依赖及兼容约束的统一声明，包含框架、PyTorch 和原注册组件所需依赖。构建与运行均固定使用 `setuptools 69.1.1`。
- [requirements-lock.txt](../requirements-lock.txt)：完整 Python 包版本快照。目前记录 95 个外部包，以及通过 `-e .` 安装的本地 AD-L 包（`agentdropout-langchain 0.1.0`）。95 个外部包的版本与初始化快照相同，本地包记录于 2026-11-03 补入。快照面向 macOS ARM64 / Python 3.10，不包含 Conda 系统库。
- 本文：环境来源、安装和维护的统一说明；README 仅链接到此处。

## 安装与检查

如果尚未创建 `ad-l`，先克隆只读参考环境：

```bash
conda create -n ad-l --clone agentdropout
```

已有 `ad-l` 时跳过创建，激活环境并安装本地项目及声明的依赖：

```bash
conda activate ad-l
cd /Users/tpx-198/code/AD-L
python -m pip install -e .
python scripts/check_environment.py
python -m pip check
```

可编辑安装使 Python 直接读取 `src/ad_l/` 的源码，修改源码后重新运行即可，无需手动设置 `PYTHONPATH`。修改依赖或项目元数据后需要重新安装。检查脚本只执行本地验证，不读取模型密钥，也不请求模型 API。

如果要按完整快照中的包版本安装，将上面的安装命令替换为：

```bash
python -m pip install -r requirements-lock.txt
```

必须在项目根目录运行：快照中的 `-e .` 同时安装本地 AD-L。它指向当前源码；本地版本记录在快照注释中，源码版本由 Git 管理。

依赖和构建工具已经齐备时，也可仅离线刷新本地项目安装信息：

```bash
python -m pip install -e . --no-deps --no-build-isolation --no-index
```

此离线命令不会补装依赖。编辑器需选用 `/Users/tpx-198/miniconda3/envs/ad-l/bin/python`；若 VS Code 打开的是父目录 `/Users/tpx-198/code`，AD-L 子目录中的工作区配置不会自动应用，可单独打开 AD-L 文件夹，或选好环境后刷新窗口。

## 后续维护

修改直接依赖时先更新 `pyproject.toml`，在 `ad-l` 中安装并验证，再按实测环境同步 `requirements-lock.txt` 中的完整包版本及本地项目版本记录。安装步骤和环境约定统一更新本文，避免在多个文档中重复维护。
