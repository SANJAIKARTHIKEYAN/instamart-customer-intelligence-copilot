
"""Part C — Ticket input loading and guardrails."""

from pathlib import Path

import pandas as pd
from mock_llm import est_tokens

ROOT = Path(__file__).resolve().parents[2]
TICKET_FILE = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_c"

MAX_INPUT_TOKENS = 400


def load_tickets():
    """Load tickets while preserving empty text as empty strings."""
    tickets = pd.read_csv(TICKET_FILE, keep_default_na=False)

    required_columns = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }

    missing = required_columns - set(tickets.columns)
    if missing:
        raise ValueError(f"Missing ticket columns: {sorted(missing)}")

    return tickets


def apply_input_guardrails(tickets):
    """Classify empty and over-length tickets before any model call."""
    results = []

    for _, ticket in tickets.iterrows():
        text = str(ticket["text"])

        if not text.strip():
            status = "skipped_empty"
            token_count = 0

        else:
            token_count = est_tokens(text)

            if token_count > MAX_INPUT_TOKENS:
                status = "flagged_long"
            else:
                status = "eligible"

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "order_id": ticket["order_id"],
            "text": text,
            "true_category": ticket["true_category"],
            "estimated_tokens": token_count,
            "status": status,
        })

    return pd.DataFrame(results)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickets = load_tickets()
    guarded = apply_input_guardrails(tickets)

    # Verify the official reference counts.
    assert len(guarded) == 164
    assert (guarded["status"] == "skipped_empty").sum() == 4
    assert (guarded["status"] == "flagged_long").sum() == 4
    assert (guarded["status"] == "eligible").sum() == 156

    # No empty or long ticket is eligible for a model call.
    assert not (
        guarded["status"].isin(["skipped_empty", "flagged_long"])
        & guarded["status"].eq("eligible")
    ).any()

    output_file = OUTPUT_DIR / "ticket_guardrail_audit.csv"
    guarded.to_csv(output_file, index=False)

    print("\n========== PART C TASK 1 ==========")
    print("Total tickets:", len(guarded))
    print("Empty tickets:",
          (guarded["status"] == "skipped_empty").sum())
    print("Long tickets:",
          (guarded["status"] == "flagged_long").sum())
    print("Eligible tickets:",
          (guarded["status"] == "eligible").sum())
    print("Maximum estimated tokens:",
          guarded["estimated_tokens"].max())
    print("\nStatus counts:")
    print(guarded["status"].value_counts())
    print("\nALL TASK 1 CHECKPOINTS PASSED")
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
