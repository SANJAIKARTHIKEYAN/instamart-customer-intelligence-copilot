
"""Part C — Ticket classification prompts and evaluation pipeline."""

from pathlib import Path
import pandas as pd
import mock_llm

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "outputs" / "part_c" / "tickets_redacted.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_c"

ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud",
}

ZERO_SHOT_SYSTEM_PROMPT = """
You are a customer support ticket classification assistant.

CLASSIFY the customer's ticket into exactly one of these categories:
- billing
- delivery
- technical
- fraud

Return JSON only, with exactly these two keys:
{
  "category": "one allowed category",
  "reason": "brief explanation for the classification"
}

Rules:
1. The category must be one of the four allowed categories.
2. Do not include Markdown or code fences.
3. Do not include any text outside the JSON object.
4. Keep the reason brief and relevant.
"""

FEW_SHOT_SYSTEM_PROMPT = """
You are a customer support ticket classification assistant.

CLASSIFY the customer's ticket into exactly one of these categories:
- billing
- delivery
- technical
- fraud

Return JSON only, with exactly these two keys:
{
  "category": "one allowed category",
  "reason": "brief explanation for the classification"
}

Rules:
1. The category must be one of the four allowed categories.
2. Do not include Markdown or code fences.
3. Do not include any text outside the JSON object.
4. Keep the reason brief and relevant.

EXAMPLES:

Ticket: I was charged twice for the same order.
Output: {"category": "billing", "reason": "Duplicate payment reported."}

Ticket: My order has not arrived even though it is delayed.
Output: {"category": "delivery", "reason": "The customer reports a delayed delivery."}

Ticket: I cannot log in because the application keeps crashing.
Output: {"category": "technical", "reason": "The customer reports an application issue."}

Ticket: Someone used my account to place an order without permission.
Output: {"category": "fraud", "reason": "The customer reports unauthorized account activity."}
"""


def build_classification_messages(redacted_ticket, few_shot=False):
    """Build messages with only redacted ticket text as user content."""
    system_prompt = (
        FEW_SHOT_SYSTEM_PROMPT
        if few_shot
        else ZERO_SHOT_SYSTEM_PROMPT
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": str(redacted_ticket)},
    ]


def validate_prompts():
    for prompt in (
        ZERO_SHOT_SYSTEM_PROMPT,
        FEW_SHOT_SYSTEM_PROMPT,
    ):
        assert "CLASSIFY" in prompt
        assert all(category in prompt for category in ALLOWED_CATEGORIES)
        assert '"category"' in prompt
        assert '"reason"' in prompt
        assert "JSON only" in prompt

    assert "EXAMPLES:" not in ZERO_SHOT_SYSTEM_PROMPT
    assert "EXAMPLES:" in FEW_SHOT_SYSTEM_PROMPT
    assert '"category": "fraud"' in FEW_SHOT_SYSTEM_PROMPT

    test_ticket = "[EMAIL] I was charged twice."
    messages = build_classification_messages(test_ticket)

    assert messages[1]["content"] == test_ticket
    return True


def run_pipeline(few_shot=False):
    """Run one complete classification pipeline."""

    # Local imports avoid circular imports between Part C modules.
    from src.part_c.retry_validation import call_with_retry
    from src.part_c.token_budget import validate_model_output

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Run Tasks 1 and 2 first. Missing: {INPUT_FILE}"
        )

    tickets = pd.read_csv(INPUT_FILE, keep_default_na=False)

    # Reset immediately before each complete pipeline run.
    mock_llm.reset()

    results = []
    retry_log = []
    total_tokens = 0

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

        messages = build_classification_messages(
            ticket["text"],
            few_shot=few_shot,
        )

        response = call_with_retry(
            messages=messages,
            ticket_id=ticket["ticket_id"],
            retry_log=retry_log,
        )

        model_text = response.choices[0].message.content
        tokens = int(response.usage.total_tokens)
        total_tokens += tokens

        category, reason, status = validate_model_output(model_text)

        result.update({
            "category": category,
            "reason": reason,
            "status": status,
            "tokens_used": tokens,
        })

        results.append(result)

    result_df = pd.DataFrame(results)

    prefix = "few_shot" if few_shot else "zero_shot"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_file = OUTPUT_DIR / f"{prefix}_results.csv"
    retry_file = OUTPUT_DIR / f"{prefix}_retry_log.csv"

    result_df.to_csv(result_file, index=False)
    pd.DataFrame(
        retry_log,
        columns=["ticket_id", "retry_number", "wait_seconds"],
    ).to_csv(retry_file, index=False)

    classified = result_df[
        result_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    correct = (
        classified["category"] == classified["true_category"]
    ).sum()

    accuracy = (
        correct / len(classified)
        if len(classified) > 0
        else 0.0
    )

    metrics = {
        "total_tickets": len(result_df),
        "classified": len(classified),
        "correct": int(correct),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified": (
            total_tokens / len(classified)
            if len(classified) > 0
            else 0.0
        ),
        "status_counts": result_df["status"].value_counts().to_dict(),
        "results_file": result_file,
        "retry_file": retry_file,
        "confusion_matrix": pd.crosstab(
            classified["true_category"],
            classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"],
        ),
    }

    return result_df, metrics


def main():
    validate_prompts()

    print("\n========== ZERO-SHOT PIPELINE ==========")
    zero_results, zero_metrics = run_pipeline(few_shot=False)

    print("Total tickets:", zero_metrics["total_tickets"])
    print("Classified:", zero_metrics["classified"])
    print("Correct:", zero_metrics["correct"])
    print("Accuracy:", f"{zero_metrics['accuracy']:.2%}")
    print("Total tokens:", zero_metrics["total_tokens"])
    print(
        "Tokens per classified ticket:",
        f"{zero_metrics['tokens_per_classified']:.2f}",
    )
    print("Status counts:", zero_metrics["status_counts"])
    print("\nZero-shot confusion matrix:")
    print(zero_metrics["confusion_matrix"])
    print("Saved:", zero_metrics["results_file"])

    print("\n========== FEW-SHOT PIPELINE ==========")
    few_results, few_metrics = run_pipeline(few_shot=True)

    print("Total tickets:", few_metrics["total_tickets"])
    print("Classified:", few_metrics["classified"])
    print("Correct:", few_metrics["correct"])
    print("Accuracy:", f"{few_metrics['accuracy']:.2%}")
    print("Total tokens:", few_metrics["total_tokens"])
    print(
        "Tokens per classified ticket:",
        f"{few_metrics['tokens_per_classified']:.2f}",
    )
    print("Status counts:", few_metrics["status_counts"])
    print("Saved:", few_metrics["results_file"])

    # Reference-result acceptance checks.
    expected_status_counts = {
        "classified_json": 135,
        "classified_keyword": 12,
        "flagged_invalid_output": 9,
        "skipped_empty": 4,
        "flagged_long": 4,
    }

    for name, metrics in (
        ("zero-shot", zero_metrics),
        ("few-shot", few_metrics),
    ):
        for status, expected in expected_status_counts.items():
            actual = metrics["status_counts"].get(status, 0)
            assert actual == expected, (
                f"{name}: expected {expected} {status}, got {actual}"
            )

        assert metrics["classified"] == 147, name

    assert round(zero_metrics["accuracy"], 3) == 0.857
    assert round(few_metrics["accuracy"], 3) == 1.000

    print("\n========== TASK 6 COMPARISON ==========")
    print(
        f"Zero-shot accuracy: {zero_metrics['accuracy']:.2%}"
    )
    print(
        f"Few-shot accuracy:  {few_metrics['accuracy']:.2%}"
    )
    print("Zero-shot tokens:", zero_metrics["total_tokens"])
    print("Few-shot tokens:", few_metrics["total_tokens"])
    print("\nALL TASK 6 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
