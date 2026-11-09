
"""Part D — Agent checkpointing, simulated crash and resume."""

from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import mock_llm

from src.part_d.refund_tools import TOOLS_SPEC, TOOL_MAP


ROOT = Path(__file__).resolve().parents[2]

SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
STATE_DIR = ROOT / "state"
OUTPUT_DIR = ROOT / "outputs" / "part_d"

MAX_ITERATIONS = 6
CRASH_AFTER_TOOLS = 2


class SimulatedCrash(Exception):
    """Raised to demonstrate recovery from an interruption."""


def checkpoint_path(run_id):
    """Return the checkpoint file path for a run."""

    return STATE_DIR / f"{run_id}.json"


def save_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    step_no,
    executed_tools,
    llm_calls,
    iteration,
):
    """Save the complete agent state after a tool observation."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_no": step_no,
        "executed_tools": executed_tools,
        "llm_calls": llm_calls,
        "iteration": iteration,
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    path = checkpoint_path(run_id)

    path.write_text(
        json.dumps(state, indent=2, default=str),
        encoding="utf-8",
    )

    return path


def load_checkpoint(run_id):
    """Load a previously saved agent state."""

    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def extract_tool_call(tool_call):
    """Extract tool ID, name and arguments."""

    tool_id = getattr(tool_call, "id", None)
    tool_name = getattr(tool_call, "name", None)
    arguments = getattr(tool_call, "arguments", {})

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dictionary.")

    return tool_id, tool_name, arguments


def _run_from_state(state, crash_after_tools=None):
    """Continue execution from an existing state."""

    run_id = state["run_id"]
    scenario_id = state["scenario_id"]

    messages = state["messages"]
    tokens_used = int(state["tokens_used"])
    step_no = int(state["step_no"])
    executed_tools = list(state["executed_tools"])
    llm_calls = int(state["llm_calls"])
    iteration = int(state["iteration"])

    while iteration < MAX_ITERATIONS:

        response = mock_llm.chat(
            messages,
            tools=TOOLS_SPEC,
        )

        llm_calls += 1
        iteration += 1

        tokens_used += int(response.usage.total_tokens)

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # No more tool calls: final answer received.
        if not tool_calls:

            final_answer = (
                message.content
                or "The agent did not provide a final answer."
            )

            status = (
                final_answer.split(":", 1)[0].strip()
                if ":" in final_answer
                else "COMPLETED"
            )

            result = {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_answer,
                "tokens_used": tokens_used,
                "step_no": step_no,
                "llm_calls": llm_calls,
                "executed_tools": executed_tools,
            }

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            result_file = OUTPUT_DIR / "checkpoint_resume_result.json"

            result_file.write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )

            return result

        assistant_tool_calls = []

        for tool_call in tool_calls:
            tool_id, tool_name, arguments = (
                extract_tool_call(tool_call)
            )

            assistant_tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls,
        })

        for tool_call in tool_calls:

            tool_id, tool_name, arguments = (
                extract_tool_call(tool_call)
            )

            if tool_name not in TOOL_MAP:
                observation = {
                    "error": f"Unknown tool: {tool_name}"
                }
            else:
                try:
                    observation = TOOL_MAP[tool_name](
                        **arguments
                    )
                except Exception as error:
                    observation = {
                        "error": str(error)
                    }

            step_no += 1
            executed_tools.append(tool_name)

            # Append the observation before checkpointing.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": json.dumps(
                    observation,
                    default=str,
                ),
            })

            saved_path = save_checkpoint(
                run_id=run_id,
                scenario_id=scenario_id,
                messages=messages,
                tokens_used=tokens_used,
                step_no=step_no,
                executed_tools=executed_tools,
                llm_calls=llm_calls,
                iteration=iteration,
            )

            print(
                f"Checkpoint saved after tool {step_no}: "
                f"{tool_name}"
            )

            # Simulate a crash only AFTER the checkpoint exists.
            if (
                crash_after_tools is not None
                and step_no == crash_after_tools
            ):
                print("Checkpoint file:", saved_path)
                raise SimulatedCrash(
                    f"Simulated interruption after tool {step_no}."
                )

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": (
            "FALLBACK_MAX_ITERATIONS: "
            "The agent reached its iteration limit."
        ),
        "tokens_used": tokens_used,
        "step_no": step_no,
        "llm_calls": llm_calls,
        "executed_tools": executed_tools,
    }


def start_agent(run_id, scenario_id, user_message):
    """Start a fresh agent run and simulate a crash after two tools."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    path = checkpoint_path(run_id)

    # This is a fresh demonstration run.
    if path.exists():
        path.unlink()

    initial_state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {
                "role": "system",
                "content": """
You are an AGENT responsible for handling customer refund requests.

Always look up the order first.
Search the refund policy before calculating a refund.
Only completed orders qualify.
Delivery exceeding 45 minutes qualifies for a full refund.
Refunds above INR 1000 require human approval.
Never claim a refund or escalation unless the tool confirms it.
""".strip(),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "tokens_used": 0,
        "step_no": 0,
        "executed_tools": [],
        "llm_calls": 0,
        "iteration": 0,
    }

    return _run_from_state(
        initial_state,
        crash_after_tools=CRASH_AFTER_TOOLS,
    )


def resume_agent(run_id):
    """Resume a previously interrupted agent run."""

    state = load_checkpoint(run_id)

    print("\nResuming run:", run_id)
    print("Previously completed tools:", state["executed_tools"])
    print("Previous step number:", state["step_no"])

    # No tools are re-executed here.
    # The saved tool observations are already in messages.
    return _run_from_state(
        state,
        crash_after_tools=None,
    )


def main():

    print("\n========== PART D TASK 8 ==========")

    scenarios = pd.read_csv(
        SCENARIOS_FILE,
        keep_default_na=False,
    )

    s1 = scenarios[
        scenarios["scenario_id"] == "S1"
    ]

    if s1.empty:
        raise ValueError("S1 scenario not found.")

    scenario = s1.iloc[0]

    run_id = "S1_checkpoint_demo"

    # The mock agent is deterministic. Reset before starting.
    mock_llm.reset()

    print("\nSTEP 1 — START S1 AGENT")

    try:
        start_agent(
            run_id=run_id,
            scenario_id="S1",
            user_message=scenario["message"],
        )

        raise AssertionError(
            "The simulated crash did not occur."
        )

    except SimulatedCrash as error:
        print("\nSIMULATED CRASH:", error)

    # Verify that the checkpoint contains two completed tools.
    saved_state = load_checkpoint(run_id)

    assert saved_state["step_no"] == 2
    assert saved_state["executed_tools"] == [
        "get_order",
        "search_policy",
    ]

    assert len([
        message
        for message in saved_state["messages"]
        if message["role"] == "tool"
    ]) == 2

    print("\nSTEP 2 — CHECKPOINT VERIFIED")
    print("Saved steps:", saved_state["step_no"])
    print("Saved tools:", saved_state["executed_tools"])
    print("Saved token usage:", saved_state["tokens_used"])

    print("\nSTEP 3 — RESUME AGENT")

    result = resume_agent(run_id)

    print("\nSTEP 4 — FINAL RESULT")
    print("Status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Total model calls:", result["llm_calls"])
    print("Total tools executed:", result["step_no"])
    print("All executed tools:", result["executed_tools"])

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]

    # Verify exactly four tool executions, in order.
    assert result["executed_tools"] == expected_tools
    assert result["step_no"] == 4

    # S1 must finish with the expected refund.
    assert result["status"] == "REFUND"
    assert "128.25" in result["final_answer"]

    # Five model calls: four tool requests and one final answer.
    assert result["llm_calls"] == 5

    # Verify the checkpoint has been updated through step four.
    final_state = load_checkpoint(run_id)

    assert final_state["step_no"] == 4
    assert final_state["executed_tools"] == expected_tools

    print("\nCheckpoint file:", checkpoint_path(run_id))
    print(
        "Final result file:",
        OUTPUT_DIR / "checkpoint_resume_result.json",
    )

    print("\nALL TASK 8 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
