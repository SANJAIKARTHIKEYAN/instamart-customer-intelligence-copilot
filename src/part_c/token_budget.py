
"""Part C — Output validation and token budget demonstration."""

from pathlib import Path
import json
import re

import pandas as pd
import mock_llm

from src.part_c.ticket_classification import (
    ALLOWED_CATEGORIES,
    build_classification_messages,
)
from src.part_c.retry_validation import call_with_retry


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "outputs" / "part_c" / "tickets_redacted.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "part_c"

TOKEN_BUDGET = 6000


# ============================================================
# TASK 5 — MODEL OUTPUT VALIDATION
# ============================================================

def validate_model_output(model_text):
    """
    Validate model output.

    1. Valid JSON with allowed category -> classified_json
    2. Invalid JSON with exactly one allowed category word
       -> classified_keyword
    3. All other outputs -> flagged_invalid_output
    """

    text = str(model_text).strip()

    # First: JSON validation.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            category = parsed.get("category")
            reason = parsed.get("reason", "")

            if isinstance(category, str):
                category = category.strip().lower()

                if category in ALLOWED_CATEGORIES:
                    return (
                        category,
                        str(reason),
                        "classified_json",
                    )

        # Valid JSON but unsupported category or structure.
        return "", "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        pass

    # Second: keyword fallback.
    words = re.findall(r"[a-z]+", text.lower())

    found_categories = [
        word for word in words
        if word in ALLOWED_CATEGORIES
    ]

    if len(found_categories) == 1:
        return (
            found_categories[0],
            "Category extracted from keyword fallback.",
            "classified_keyword",
        )

    return "", "", "flagged_invalid_output"


def run_validation_tests():
    """Verify JSON, keyword and invalid-output handling."""

    tests = [
        (
            '{"category": "billing", "reason": "Duplicate charge"}',
            "billing",
            "classified_json",
        ),
        (
            "This ticket appears to concern delivery.",
            "delivery",
            "classified_keyword",
        ),
        (
            "This ticket may involve billing and fraud.",
            "",
            "flagged_invalid_output",
        ),
        (
            '{"category": "shipping", "reason": "Unknown category"}',
            "",
            "flagged_invalid_output",
        ),
        (
            "I cannot determine the category.",
            "",
            "flagged_invalid_output",
        ),
    ]

    for model_text, expected_category, expected_status in tests:
        category, reason, status = validate_model_output(
            model_text
        )

        assert category == expected_category
        assert status == expected_status

    return len(tests)


# ============================================================
# TASK 7A — TOKEN BUDGET DEMONSTRATION
# ============================================================

def run_budget_demo(token_budget=TOKEN_BUDGET):
    """Run zero-shot classification with a token budget."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. "
            "Complete Tasks 1 and 2 first."
        )

    tickets = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    # Reset immediately before the complete run.
    mock_llm.reset()

    results = []
    retry_log = []
    accumulated_tokens = 0

    for _, ticket in tickets.iterrows():

        original_status = ticket["status"]

        result = {
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "category": "",
            "reason": "",
            "status": original_status,
            "tokens_used": 0,
        }

        # Empty and long tickets never reach the model.
        if original_status != "eligible":
            results.append(result)
            continue

        # Check budget BEFORE making a model call.
        if accumulated_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        messages = build_classification_messages(
            ticket["text"],
            few_shot=False,
        )

        response = call_with_retry(
            messages=messages,
            ticket_id=ticket["ticket_id"],
            retry_log=retry_log,
        )

        model_text = response.choices[0].message.content

        tokens = int(response.usage.total_tokens)
        accumulated_tokens += tokens

        category, reason, status = validate_model_output(
            model_text
        )

        result.update({
            "category": category,
            "reason": reason,
            "status": status,
            "tokens_used": tokens,
        })

        results.append(result)

    result_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR / "triage_budget_demo.csv"
    )

    retry_file = (
        OUTPUT_DIR / "budget_demo_retry_log.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    pd.DataFrame(
        retry_log,
        columns=[
            "ticket_id",
            "retry_number",
            "wait_seconds",
        ],
    ).to_csv(
        retry_file,
        index=False,
    )

    budget_count = int(
        (
            result_df["status"] == "budget_exhausted"
        ).sum()
    )

    return (
        result_df,
        accumulated_tokens,
        budget_count,
        output_file,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    validation_count = run_validation_tests()

    print("\n========== TASK 5 VALIDATION ==========")
    print("Validation test cases passed:", validation_count)

    print("\n========== PART C TASK 7A ==========")

    result_df, total_tokens, exhausted, output_file = (
        run_budget_demo()
    )

    print("Token budget:", TOKEN_BUDGET)
    print("Actual tokens used:", total_tokens)
    print("Budget exhausted tickets:", exhausted)

    print("\nStatus counts:")
    print(result_df["status"].value_counts())

    print("\nSaved:", output_file)

    # Acceptance checks.
    assert len(result_df) == 164

    assert (
        result_df["status"] == "skipped_empty"
    ).sum() == 4

    assert (
        result_df["status"] == "flagged_long"
    ).sum() == 4

    assert exhausted > 0
    assert total_tokens >= TOKEN_BUDGET

    print("\nALL TASK 5 AND TASK 7A CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
