"""GSM8K JSONL 加载、来源校验与可复核抽样。"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

GSM8K_FILES = {
    "train": ("train.jsonl", 7473, "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
    "test": ("test.jsonl", 1319, "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
}
SOURCE = "https://github.com/openai/grade-school-math"


@dataclass(frozen=True)
class GSM8KSample:
    sample_id: str
    split: str
    index: int
    question: str
    target: str


def load_gsm8k(path: Path, split: str, *, verify_official: bool = False):
    if split not in GSM8K_FILES:
        raise ValueError("split 必须为 train 或 test")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    samples = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            question, answer = record["question"], record["answer"]
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
                raise ValueError()
            if "\n####" not in answer:
                raise ValueError()
            target = answer.rsplit("\n####", 1)[1].replace(",", "").strip()
            if not target or not float("-inf") < float(target) < float("inf"):
                raise ValueError()
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path.name} 第 {line_number} 行不符合 GSM8K question/answer 格式") from error
        index = len(samples)
        samples.append(GSM8KSample(f"gsm8k:{split}:{index}", split, index, question, target))
    if not samples:
        raise ValueError("数据集不能为空")
    filename, count, expected = GSM8K_FILES[split]
    official = digest == expected and len(samples) == count
    if verify_official and not official:
        raise ValueError(f"{split} 文件校验失败；请运行 python scripts/prepare_gsm8k.py")
    return samples, {"name": "gsm8k", "split": split, "path": str(path.resolve()),
                     "sha256": digest, "records": len(samples), "verified_official": official,
                     "source": f"{SOURCE}/blob/master/grade_school_math/data/{filename}"}


def sample_gsm8k(samples: list[GSM8KSample], limit: int, seed: int):
    if type(limit) is not int or not 1 <= limit <= len(samples):
        raise ValueError(f"limit 必须在 1 到 {len(samples)} 之间")
    # 按原文件序号执行；保存样本编号后，无需依赖未来 Python 的随机实现复现选择。
    indices = sorted(random.Random(seed).sample(range(len(samples)), limit))
    return [samples[i] for i in indices]
