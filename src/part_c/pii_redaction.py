
"""Part C — PII detection and redaction."""

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "outputs" / "part_c" / "ticket_guardrail_audit.csv"
OUTPUT_FILE = ROOT / "outputs" / "part_c" / "tickets_redacted.csv"

PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact_text(text):
    """Redact PII sequentially and return redacted text plus counts."""
    redacted = str(text)
    counts = {}

    for name, pattern, replacement in PII_PATTERNS:
        matches = pattern.findall(redacted)
        counts[name] = len(matches)
        redacted = pattern.sub(replacement, redacted)

    return redacted, counts


def redact_tickets(tickets):
    """Redact ticket text while preserving all other columns."""
    result = tickets.copy()
    total_counts = {name: 0 for name, _, _ in PII_PATTERNS}

    redacted_texts = []

    for text in result["text"]:
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)

        for name, count in counts.items():
            total_counts[name] += count

    result["text"] = redacted_texts
    return result, total_counts


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Run Task 1 first. Missing: {INPUT_FILE}"
        )

    tickets = pd.read_csv(INPUT_FILE, keep_default_na=False)

    redacted, counts = redact_tickets(tickets)

    # Verify expected redaction counts.
    assert counts["email"] == 55, counts
    assert counts["phone"] == 91, counts
    assert counts["order_id"] == 132, counts
    assert sum(counts.values()) == 278, counts

    # Verify no original PII patterns remain.
    for _, pattern, _ in PII_PATTERNS:
        assert not redacted["text"].apply(
            lambda value: bool(pattern.search(value))
        ).any()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    redacted.to_csv(OUTPUT_FILE, index=False)

    print("\n========== PART C TASK 2 ==========")
    print("Email redactions:", counts["email"])
    print("Phone redactions:", counts["phone"])
    print("Order ID redactions:", counts["order_id"])
    print("Total redactions:", sum(counts.values()))
    print("Remaining PII matches: 0")
    print("Output rows:", len(redacted))
    print("\nALL TASK 2 CHECKPOINTS PASSED")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
