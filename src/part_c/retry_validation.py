
"""Part C — Retry logic with exponential backoff and jitter."""

from pathlib import Path
import random
import time

import pandas as pd
import mock_llm

from src.part_c.ticket_classification import build_classification_messages

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "outputs" / "part_c" / "tickets_redacted.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_c"
RETRY_LOG_FILE = OUTPUT_DIR / "retry_log.csv"

MAX_RETRIES = 3
BASE_DELAY = 0.05


def call_with_retry(messages, ticket_id, retry_log):
    """Retry transient model failures up to three times."""

    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES:
                raise

            retry_number = attempt

            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            retry_log.append({
                "ticket_id": ticket_id,
                "retry_number": retry_number,
                "wait_seconds": wait_seconds,
            })

            print(
                f"Ticket {ticket_id}: "
                f"retry {retry_number}, "
                f"waiting {wait_seconds:.6f}s"
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected retry termination.")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. "
            "Run the previous Part C tasks first."
        )

    tickets = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False
    )

    eligible = tickets[
        tickets["status"] == "eligible"
    ].copy()

    assert len(eligible) == 156, (
        f"Expected 156 eligible tickets, got {len(eligible)}"
    )

    assert not eligible["text"].apply(
        lambda text: not str(text).strip()
    ).any(), "An empty ticket was included."

    retry_log = []

    # Reset immediately before the complete run.
    mock_llm.reset()

    successful_calls = 0

    for _, ticket in eligible.iterrows():

        messages = build_classification_messages(
            ticket["text"],
            few_shot=False
        )

        call_with_retry(
            messages=messages,
            ticket_id=ticket["ticket_id"],
            retry_log=retry_log
        )

        successful_calls += 1

    retry_df = pd.DataFrame(
        retry_log,
        columns=[
            "ticket_id",
            "retry_number",
            "wait_seconds"
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    retry_df.to_csv(
        RETRY_LOG_FILE,
        index=False
    )

    # Acceptance checks.
    assert successful_calls == 156
    assert len(retry_df) == 19

    # The supplied mock recovers on retry number 0.
    assert (retry_df["retry_number"] == 0).all()

    # Verify the actual jitter range without rounding.
    for _, row in retry_df.iterrows():

        n = int(row["retry_number"])

        minimum = BASE_DELAY * (2 ** n) * 0.9
        maximum = BASE_DELAY * (2 ** n) * 1.15

        wait = float(row["wait_seconds"])

        assert minimum - 1e-12 <= wait <= maximum + 1e-12

    print("\n========== PART C TASK 4 ==========")
    print("Eligible tickets:", len(eligible))
    print("Successful calls:", successful_calls)
    print("Transient failures recovered:", len(retry_df))
    print("Retry numbers:", sorted(retry_df["retry_number"].unique()))
    print("Retry log:", RETRY_LOG_FILE)

    print("\nALL TASK 4 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
