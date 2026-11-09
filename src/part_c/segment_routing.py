
"""Part C — Final few-shot triage and customer segment routing."""

from pathlib import Path

import pandas as pd
import mock_llm

from src.part_c.ticket_classification import (
    build_classification_messages,
)
from src.part_c.retry_validation import call_with_retry
from src.part_c.token_budget import validate_model_output


ROOT = Path(__file__).resolve().parents[2]

TICKETS_FILE = (
    ROOT / "outputs" / "part_c" / "tickets_redacted.csv"
)

SEGMENTS_FILE = (
    ROOT / "outputs" / "customer_segments.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "part_c"

OUTPUT_FILE = OUTPUT_DIR / "triage_results.csv"
RETRY_FILE = OUTPUT_DIR / "final_retry_log.csv"

TOKEN_BUDGET = 100000


def load_customer_segments():
    """Load customer segment names from Part B."""

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Part B segment file not found: {SEGMENTS_FILE}"
        )

    segments = pd.read_csv(
        SEGMENTS_FILE,
        keep_default_na=False,
    )

    required = {"customer_id", "segment_name"}
    missing = required - set(segments.columns)

    if missing:
        raise ValueError(
            f"Missing columns in customer_segments.csv: {missing}. "
            f"Available columns: {list(segments.columns)}"
        )

    segments = segments[
        ["customer_id", "segment_name"]
    ].drop_duplicates(
        subset=["customer_id"],
        keep="first",
    )

    return segments


def is_premium_customer(segment_name):
    """Identify Premium customers from their segment label."""

    return "premium" in str(segment_name).strip().lower()


def route_ticket(status, category, segment_name):
    """Apply the official Part C routing rules."""

    human_review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    if status in human_review_statuses:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium_customer(segment_name):
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


def run_final_triage(token_budget=TOKEN_BUDGET):
    """Run the complete few-shot pipeline and route all tickets."""

    if not TICKETS_FILE.exists():
        raise FileNotFoundError(
            f"Missing ticket file: {TICKETS_FILE}. "
            "Complete Part C Tasks 1 and 2 first."
        )

    tickets = pd.read_csv(
        TICKETS_FILE,
        keep_default_na=False,
    )

    segments = load_customer_segments()

    # Build customer ID -> segment name lookup.
    segment_lookup = dict(
        zip(
            segments["customer_id"].astype(str),
            segments["segment_name"].astype(str),
        )
    )

    # Reset immediately before this complete run.
    mock_llm.reset()

    results = []
    retry_log = []
    accumulated_tokens = 0

    for _, ticket in tickets.iterrows():

        customer_id = str(ticket["customer_id"])
        original_status = ticket["status"]

        segment_name = segment_lookup.get(
            customer_id,
            "",
        )

        category = ""
        status = original_status
        tokens_used = 0

        # Only eligible tickets may reach the model.
        if original_status == "eligible":

            # Check budget before calling the model.
            if accumulated_tokens >= token_budget:
                status = "budget_exhausted"

            else:
                messages = build_classification_messages(
                    ticket["text"],
                    few_shot=True,
                )

                response = call_with_retry(
                    messages=messages,
                    ticket_id=ticket["ticket_id"],
                    retry_log=retry_log,
                )

                model_text = response.choices[0].message.content

                tokens_used = int(
                    response.usage.total_tokens
                )

                accumulated_tokens += tokens_used

                category, reason, status = (
                    validate_model_output(model_text)
                )

        route, priority = route_ticket(
            status,
            category,
            segment_name,
        )

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
        })

    result_df = pd.DataFrame(
        results,
        columns=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
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
        RETRY_FILE,
        index=False,
    )

    return result_df, accumulated_tokens, retry_log


def main():

    print("\n========== PART C TASK 7B ==========")

    result_df, total_tokens, retry_log = (
        run_final_triage()
    )

    print("Total tickets:", len(result_df))
    print("Token budget:", TOKEN_BUDGET)
    print("Total tokens used:", total_tokens)
    print("Retry events:", len(retry_log))

    print("\nClassification status counts:")
    print(result_df["status"].value_counts())

    print("\nRouting distribution:")
    print(result_df["route"].value_counts())

    print("\nPriority distribution:")
    print(result_df["priority"].value_counts())

    print("\nFraud routing check:")
    fraud_rows = result_df[
        result_df["category"] == "fraud"
    ]

    print("Fraud tickets:", len(fraud_rows))
    print(
        "Correctly routed to security:",
        (
            (fraud_rows["route"] == "security_team")
            & (fraud_rows["priority"] == "HIGH")
        ).sum(),
    )

    # Acceptance checks.
    assert len(result_df) == 164

    assert list(result_df.columns) == [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    # All fraud classifications must reach the security team.
    assert (
        result_df.loc[
            result_df["category"] == "fraud",
            "route",
        ] == "security_team"
    ).all()

    # Human-review statuses must never be automatically routed.
    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    human_rows = result_df[
        result_df["status"].isin(human_statuses)
    ]

    assert (human_rows["route"] == "human_review").all()

    # Premium routing must have HIGH priority.
    premium_rows = result_df[
        (result_df["route"] == "premium_desk")
    ]

    assert (premium_rows["priority"] == "HIGH").all()

    # Every valid classified ticket must have a route.
    valid_rows = result_df[
        result_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    assert valid_rows["route"].isin([
        "security_team",
        "premium_desk",
        "support_queue",
    ]).all()

    assert total_tokens <= TOKEN_BUDGET

    print("\nSaved final results:", OUTPUT_FILE)
    print("Saved retry log:", RETRY_FILE)
    print("\nALL TASK 7B CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
