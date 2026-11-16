#!/bin/zsh
cd -- "${0:A:h}"
task_python="${ADL_PYTHON:-/Users/tpx-198/miniconda3/envs/ad-l/bin/python}"
exec "$task_python" -B platform_server.py
