"""准备 GSM8K 数据，默认从只读原项目复制；--download 从作者仓库下载。"""

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ad_l.datasets import GSM8K_FILES, load_gsm8k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--source-dir", type=Path, default=Path("/Users/tpx-198/code/AgentDropout/datasets/gsm8k"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/gsm8k")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, (filename, _, digest) in GSM8K_FILES.items():
        target = args.output_dir / filename
        if target.exists():
            load_gsm8k(target, split, verify_official=True)
            print(f"已校验：{target}")
            continue
        if args.download:
            url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{filename}"
            with urlopen(url, timeout=60) as response:
                raw = response.read()
        else:
            source = args.source_dir / ("gsm8k.jsonl" if split == "test" else filename)
            if not source.is_file():
                parser.error("找不到原项目数据，请使用 --source-dir 或 --download")
            raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            parser.error(f"{split} 来源文件校验失败，不写入目标")
        with target.open("xb") as stream:
            stream.write(raw)
        samples, _ = load_gsm8k(target, split, verify_official=True)
        print(f"已准备：{target}（{len(samples)} 题）")


if __name__ == "__main__":
    main()
