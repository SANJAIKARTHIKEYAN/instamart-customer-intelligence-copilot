
"""Part D — ReAct vs Plan-and-Execute comparison."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

AGENT_RUNS_FILE = ROOT / "outputs" / "part_d" / "agent_runs.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "part_d"

REPORT_FILE = OUTPUT_DIR / "agent_comparison.txt"


def load_react_runs():
    """Load actual ReAct benchmark results."""

    if not AGENT_RUNS_FILE.exists():
        raise FileNotFoundError(
            "Agent run log not found. Run Task 6 first."
        )

    records = []

    with AGENT_RUNS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("Agent run log is empty.")

    return pd.DataFrame(records)


def calculate_react_metrics(results):
    """Calculate metrics from actual ReAct executions."""

    return {
        "scenarios": len(results),
        "total_llm_calls": int(results["llm_calls"].sum()),
        "total_tokens": int(results["tokens_used"].sum()),
        "average_llm_calls": round(
            results["llm_calls"].mean(), 2
        ),
        "average_tokens": round(
            results["tokens_used"].mean(), 2
        ),
    }


def build_comparison(metrics):
    """Build a concise architecture comparison."""

    report = f"""
REACT VS PLAN-AND-EXECUTE COMPARISON
===================================

1. REACT APPROACH

ReAct combines reasoning and actions. The agent observes the
customer request, calls a tool, examines its result and decides
the next action.

Actual benchmark results:
- Scenarios: {metrics['scenarios']}
- Total model calls: {metrics['total_llm_calls']}
- Total tokens: {metrics['total_tokens']}
- Average model calls: {metrics['average_llm_calls']}
- Average tokens: {metrics['average_tokens']}

2. PLAN-AND-EXECUTE APPROACH

Plan-and-Execute first creates a sequence of actions and then
executes that plan. It can reduce repeated planning calls when
requests follow a predictable workflow. However, unexpected
tool observations may require replanning or human intervention.

No Plan-and-Execute benchmark was implemented in this project,
so comparative token and call savings are not claimed.

3. UNEXPECTED OBSERVATIONS

The missing order in S6 returned NOT_FOUND, allowing the agent
to stop without attempting a refund.

The cancelled order in S5 was denied because only completed
orders qualify for refunds.

These observations demonstrate why the agent must inspect
actual tool results rather than blindly follow a fixed plan.

4. SELECTED APPROACH

ReAct is selected for this project because refund decisions
depend on live tool observations, including order status,
delivery time, policy retrieval and refund calculations.
Its step-by-step approach supports safe decisions and human
handoff when unexpected conditions occur.

Plan-and-Execute may be more efficient for predictable workflows,
but ReAct provides better flexibility for this support scenario.
""".strip()

    return report


def main():

    print("\n========== PART D TASK 9 ==========")

    results = load_react_runs()

    metrics = calculate_react_metrics(results)

    report = build_comparison(metrics)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    word_count = len(report.split())

    print("\nReport word count:", word_count)

    assert len(results) == 7
    assert metrics["total_llm_calls"] == 30
    assert metrics["total_tokens"] == 11784
    assert word_count <= 250

    print("\nSaved:", REPORT_FILE)
    print("\nALL TASK 9 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
