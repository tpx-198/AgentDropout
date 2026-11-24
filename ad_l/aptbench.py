"""Normalize the tasks used by the upstream APTBench evaluation scripts."""

from collections import Counter
from dataclasses import dataclass
import json
import re
from string import ascii_uppercase


REVISION = "9aab1301ab2c743be6bf11d1a2f3c3c6bc123ffc"
ARCHIVE_URL = f"https://codeload.github.com/TencentYoutuResearch/APTBench/tar.gz/{REVISION}"
RAW_URL = f"https://raw.githubusercontent.com/TencentYoutuResearch/APTBench/{REVISION}/"


@dataclass(frozen=True)
class Task:
    instruction: str
    fields: tuple[tuple[str, str], ...]
    answer_type: str = "choice"
    target_field: str = "answer"


RESEARCH_CONTEXT = (("Research question", "query"), ("Research trajectory so far", "trajectory"))
RESEARCH_PLAN = Task("Select the most appropriate next research step.",
                     (*RESEARCH_CONTEXT, ("Available tools for reference", "tool_set")))
RESEARCH_ANSWER = Task("Give a concise answer to the research question using the supplied evidence.",
                       RESEARCH_CONTEXT, "text")
RESEARCH_CITATION = Task("Select all passages that can be supported by the provided web page.",
                         (("Article", "article"), ("Web page content", "url_content")), "multi_choice")
RESEARCH_QUALITY = Task("Select the best report for the research question.", (("Research question", "query"),))
TOOL_CONTEXT = (("User request", "user_prompt"), ("Tool definitions", "function"))
TOOL_SELECT = Task("Give the exact name of the tool that best fits the request.",
                    TOOL_CONTEXT, "text", "answer4select")
TOOL_PARAM = Task("Give the exact argument value for the named parameter of the specified tool.",
                   (*TOOL_CONTEXT, ("Specified tool", "answer4select"), ("Parameter", "param_name")),
                   "text", "answer4param")

TASKS = {
    "aptbench_swe": {
        "env_setup/plan": Task("Select the best plan for setting up the software environment.",
                               (("Repository information", "repo_info"),)),
        "env_setup/action": Task("Give the next shell command for the setup plan.",
                                 (("Setup plan", "execution_plan"), ("Commands already run", "executed_cmds")),
                                 "text", "target_cmd"),
        "env_setup/error": Task("Select the best plan for resolving the setup issue.",
                                (("Setup instructions", "setup_instruct"), ("Issue title", "issue_title"),
                                 ("Issue body", "issue_body"))),
        "issue_fix/locate": Task("Select the code snippet that should be changed to fix the issue.",
                                (("Issue", "issue_statement"),)),
        "issue_fix/fix_patch": Task("Select the patch that best fixes the issue.",
                                   (("Issue and code context", "issue_statement"),)),
        "issue_fix/plan": Task("Select the most appropriate next step for resolving the issue.",
                              (("Trajectory so far", "trajs"),)),
        "issue_fix/action": Task("Give the next command or editor action as one line of text.",
                                (("Trajectory so far", "trajs"),), "text"),
        "issue_fix/test_patch": Task("Select the test case that reproduces the described software issue.",
                                    (("Issue", "problem_statement"),)),
    },
    "aptbench_dr": {
        "deepresearch/plan_en": RESEARCH_PLAN,
        "deepresearch/plan_zh": RESEARCH_PLAN,
        "deepresearch/summ_ans_en": RESEARCH_ANSWER,
        "deepresearch/summ_ans_zh": RESEARCH_ANSWER,
        "deepresearch/openend_plan_en": Task("Select the best overall plan for answering the research question.",
                                             (("Research question", "question"),)),
        "deepresearch/openend_citation_en": RESEARCH_CITATION,
        "deepresearch/openend_citation_zh": RESEARCH_CITATION,
        "deepresearch/openend_quality_en": RESEARCH_QUALITY,
        "deepresearch/openend_quality_zh": RESEARCH_QUALITY,
    },
    "aptbench_tool": {
        "tool/acebench_api_select": TOOL_SELECT,
        "tool/acebench_api_param": TOOL_PARAM,
        "tool/bfcl_v4_api_select": TOOL_SELECT,
        "tool/bfcl_v4_api_param": TOOL_PARAM,
    },
    "aptbench_math": {
        "agentic_math/planning_single": Task("Select the most relevant capability to solve the problem.",
                                              (("Math problem", "repo_info"),)),
        "agentic_math/feedback_tf": Task("Judge the proposed solution. Answer exactly correct or wrong.",
                                          (("Math problem and proposed solution", "repo_info"),), "text"),
        "agentic_math/action_cal": Task("Select the correct calculation result.", (("Calculation", "repo_info"),)),
    },
}


def sources(dataset):
    return {task: f"{RAW_URL}data/{task}/input_data.jsonl" for task in TASKS[dataset]} | {"license": RAW_URL + "LICENSE"}


def _choices(text, *, inline):
    separator = r"(?:^|\s)" if inline else r"(?:^|\n[ \t]*)"
    parts = re.split(separator + r"\(([A-Z])\)[ \t]*", text.strip())
    labels = "".join(parts[1::2])
    options = [part.strip() for part in parts[2::2]]
    if parts[0] or sum(bool(option) for option in options) < 2 or not ascii_uppercase.startswith(labels):
        raise ValueError("Expected consecutive APTBench option labels and at least two valid options")
    return options


def normalize_aptbench(dataset, content):
    rows = []
    counts = Counter()
    excluded = []
    for name, task in TASKS[dataset].items():
        for index, line in enumerate(content[name].splitlines(), 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = f"{dataset}:{name}:{index}"
            if task == TOOL_PARAM and not sample["param_name"].strip():
                excluded.append({"id": sample_id, "reason": "parameter_name_missing"})
                continue
            sections = [task.instruction]
            # Only these upstream fields enter the task. Gold answers, explanations and next steps are labels.
            sections.extend(f"{heading}\n{sample[field].strip()}" for heading, field in task.fields)
            target = sample[task.target_field]
            row = {"id": sample_id, "dataset": dataset, "split": "test",
                   "question": "\n\n".join(sections), "answer_type": task.answer_type,
                   "target": ",".join(sorted(set(target))) if task.answer_type == "multi_choice" else target.strip()}
            if task.answer_type != "text":
                row["choices"] = _choices(sample["choices"], inline=name.startswith(("env_setup/", "agentic_math/")))
                available = {label for label, option in zip(ascii_uppercase, row["choices"]) if option}
                if set(row["target"].split(",")) - available:
                    excluded.append({"id": row["id"], "reason": "gold_option_missing"})
                    continue
            rows.append(row)
            counts[name] += 1
    return {"test": rows}, {"kind": "upstream_test_only", "upstream_revision": REVISION,
                           "tasks": dict(counts), "excluded": excluded}
