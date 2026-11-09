
"""Part D — ReAct agent orchestration and safe stopping."""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import pandas as pd
import mock_llm

from src.part_d.refund_tools import TOOLS_SPEC, TOOL_MAP


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_d"

AGENT_RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"
SAFETY_OUTPUT_FILE = OUTPUT_DIR / "safe_stopping_demo.json"

MAX_ITERATIONS = 6
TOKEN_LIMIT = 6000
TIME_LIMIT_SECONDS = 10

SYSTEM_PROMPT = """
You are an AGENT responsible for handling customer refund requests.

Always look up the order before making a refund decision.
Search the refund policy before calculating a refund.
Only completed orders are eligible for refunds.
Delivery exceeding 45 minutes qualifies for a full refund.
Delivery of 45 minutes or less does not qualify.
Refunds above INR 1000 require human approval.
Never claim a refund or escalation unless the relevant tool confirms it.
If a safety limit is reached, hand the case to human support.
"""


def _get_tool_call_data(tool_call):
    tool_id = getattr(tool_call, "id", None)
    tool_name = getattr(tool_call, "name", None)
    arguments = getattr(tool_call, "arguments", {})

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dictionary.")

    return tool_id, tool_name, arguments


def _fallback(status, message):
    return {
        "status": status,
        "final_answer": (
            f"{status}: {message} "
            "Please contact human support for assistance."
        ),
    }


def run_agent(
    user_message,
    scenario_id="MANUAL",
    max_iterations=MAX_ITERATIONS,
    token_limit=TOKEN_LIMIT,
    time_limit_seconds=TIME_LIMIT_SECONDS,
):
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    if token_limit < 1:
        raise ValueError("token_limit must be at least 1.")

    if time_limit_seconds < 0:
        raise ValueError("time_limit_seconds cannot be negative.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": str(user_message)},
    ]

    steps = []
    executed_tools = []
    llm_calls = 0
    tokens_used = 0

    start_time = time.monotonic()

    for iteration in range(max_iterations):

        if time.monotonic() - start_time >= time_limit_seconds:
            return {
                **_fallback(
                    "FALLBACK_TIMEOUT",
                    "The agent exceeded its time limit.",
                ),
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "executed_tools": executed_tools,
            }

        try:
            response = mock_llm.chat(
                messages,
                tools=TOOLS_SPEC,
            )
        except mock_llm.TransientLLMError:
            return {
                **_fallback(
                    "FALLBACK_LLM_ERROR",
                    "The language model request failed.",
                ),
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "executed_tools": executed_tools,
            }

        llm_calls += 1
        call_tokens = int(response.usage.total_tokens)
        tokens_used += call_tokens

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # Stop BEFORE executing tools requested by an
        # over-budget model response.
        if tokens_used > token_limit:
            return {
                **_fallback(
                    "FALLBACK_TOKEN_LIMIT",
                    "The agent exceeded its token budget.",
                ),
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "executed_tools": executed_tools,
            }

        if not tool_calls:
            final_answer = (
                message.content
                or "The agent did not provide a final response."
            )

            status = (
                final_answer.split(":", 1)[0].strip()
                if ":" in final_answer
                else "COMPLETED"
            )

            return {
                "status": status,
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "executed_tools": executed_tools,
            }

        assistant_tool_calls = []

        for tool_call in tool_calls:
            tool_id, tool_name, arguments = (
                _get_tool_call_data(tool_call)
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
                _get_tool_call_data(tool_call)
            )

            if tool_name not in TOOL_MAP:
                observation = {
                    "error": f"Unknown tool: {tool_name}"
                }
            else:
                try:
                    observation = TOOL_MAP[tool_name](**arguments)
                except Exception as error:
                    observation = {"error": str(error)}

            executed_tools.append(tool_name)

            steps.append({
                "iteration": iteration + 1,
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": json.dumps(observation, default=str),
            })

    return {
        **_fallback(
            "FALLBACK_MAX_ITERATIONS",
            "The agent reached its maximum iteration limit.",
        ),
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "executed_tools": executed_tools,
    }


def _save_agent_run(scenario_id, result, output_file):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario_id": scenario_id,
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
        "executed_tools": result["executed_tools"],
    }

    with output_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def run_all_scenarios():
    scenarios = pd.read_csv(
        SCENARIOS_FILE,
        keep_default_na=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with AGENT_RUNS_FILE.open("w", encoding="utf-8"):
        pass

    results = []

    for _, scenario in scenarios.iterrows():
        scenario_id = scenario["scenario_id"]
        mock_llm.reset()

        result = run_agent(
            user_message=scenario["message"],
            scenario_id=scenario_id,
        )

        _save_agent_run(
            scenario_id,
            result,
            AGENT_RUNS_FILE,
        )

        results.append({
            "scenario_id": scenario_id,
            "expected_outcome": scenario["expected_outcome"],
            "actual_status": result["status"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "executed_tools": result["executed_tools"],
            "final_answer": result["final_answer"],
        })

    return pd.DataFrame(results)


def run_safe_stopping_tests():
    scenarios = pd.read_csv(
        SCENARIOS_FILE,
        keep_default_na=False,
    )

    s1 = scenarios[
        scenarios["scenario_id"] == "S1"
    ]

    s7 = scenarios[
        scenarios["scenario_id"] == "S7"
    ]

    if s1.empty or s7.empty:
        raise ValueError("S1 or S7 missing from scenario file.")

    s1_message = s1.iloc[0]["message"]
    s7_message = s7.iloc[0]["message"]

    results = {}

    # TEST 1: TOKEN LIMIT
    mock_llm.reset()

    token_result = run_agent(
        user_message=s1_message,
        scenario_id="TOKEN_LIMIT_DEMO",
        token_limit=200,
    )

    assert token_result["status"] == "FALLBACK_TOKEN_LIMIT"
    assert token_result["tokens_used"] > 200

    # Each executed tool must belong to a previously completed
    # model iteration. The final over-budget response cannot
    # execute its requested tool.
    assert len(token_result["executed_tools"]) < (
        token_result["llm_calls"]
    )

    results["token_limit"] = token_result

    # TEST 2: TIMEOUT
    mock_llm.reset()

    timeout_result = run_agent(
        user_message=s1_message,
        scenario_id="TIMEOUT_DEMO",
        time_limit_seconds=0,
    )

    assert timeout_result["status"] == "FALLBACK_TIMEOUT"
    assert timeout_result["llm_calls"] == 0
    assert timeout_result["executed_tools"] == []

    results["timeout"] = timeout_result

    # TEST 3: MAXIMUM ITERATIONS
    mock_llm.reset()

    iteration_result = run_agent(
        user_message=s7_message,
        scenario_id="MAX_ITERATIONS_DEMO",
        max_iterations=6,
    )

    assert (
        iteration_result["status"]
        == "FALLBACK_MAX_ITERATIONS"
    )

    assert iteration_result["llm_calls"] == 6
    assert len(iteration_result["executed_tools"]) == 6

    results["max_iterations"] = iteration_result

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    SAFETY_OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    return results


def main():
    print("\n========== PART D TASK 7 ==========")

    results = run_safe_stopping_tests()

    token_result = results["token_limit"]
    print("\n1. TOKEN LIMIT TEST")
    print("Status:", token_result["status"])
    print("LLM calls:", token_result["llm_calls"])
    print("Tokens used:", token_result["tokens_used"])
    print("Previously executed tools:", token_result["executed_tools"])
    print("Over-budget response tool execution: prevented")

    timeout_result = results["timeout"]
    print("\n2. TIMEOUT TEST")
    print("Status:", timeout_result["status"])
    print("LLM calls:", timeout_result["llm_calls"])
    print("Executed tools:", timeout_result["executed_tools"])

    iteration_result = results["max_iterations"]
    print("\n3. MAXIMUM ITERATIONS TEST")
    print("Status:", iteration_result["status"])
    print("LLM calls:", iteration_result["llm_calls"])
    print("Executed tools:", iteration_result["executed_tools"])

    print("\nSaved safety results:", SAFETY_OUTPUT_FILE)
    print("\nALL TASK 7 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
