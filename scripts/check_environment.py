"""验证安装及本地执行；不创建真实模型客户端，不发送 API 请求。"""
import importlib.metadata as metadata
import sys
from pathlib import Path
from typing_extensions import TypedDict

import torch
from class_registry import ClassRegistry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    answer: str

# 检查 LangChain、LangGraph、DeepSeek 导入、原注册组件与 PyTorch 梯度是否正常。
def main():
    expected = Path('/Users/tpx-198/miniconda3/envs/ad-l')
    if Path(sys.prefix).resolve() != expected.resolve():
        raise SystemExit('请先 conda activate ad-l，再运行此脚本。')
    print('Python:', sys.version.split()[0])
    print('Environment:', sys.prefix)
    for name in ['langchain', 'langchain-core', 'langchain-deepseek', 'langgraph', 'torch', 'openai', 'setuptools']:
        print(f'{name}: {metadata.version(name)}')

    chain = ChatPromptTemplate.from_messages([('human', '{question}')]) | RunnableLambda(lambda value: value.to_messages()[0].content)
    if chain.invoke({'question': 'offline-check'}) != 'offline-check':
        raise RuntimeError('LangChain 本地消息检查失败。')
    builder = StateGraph(State)
    builder.add_node('check', lambda state: {'answer': 'ok'})
    builder.add_edge(START, 'check')
    builder.add_edge('check', END)
    if builder.compile().invoke({'answer': ''})['answer'] != 'ok':
        raise RuntimeError('LangGraph 本地流程检查失败。')

    logit = torch.nn.Parameter(torch.tensor(0.0))
    (-torch.nn.functional.logsigmoid(logit)).backward()
    if logit.grad is None or not torch.isfinite(logit.grad):
        raise RuntimeError('PyTorch 梯度检查失败。')
    print('PASS: LangChain、LangGraph、DeepSeek 导入、原注册组件与 PyTorch 梯度均正常。')
    print('本次检查没有发送模型 API 请求。')


if __name__ == '__main__':
    main()
