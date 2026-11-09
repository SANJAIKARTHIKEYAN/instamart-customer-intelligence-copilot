
"""Part D — Policy document chunking."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

POLICY_DIR = ROOT / "data" / "policies"
OUTPUT_DIR = ROOT / "outputs" / "part_d"
OUTPUT_FILE = OUTPUT_DIR / "policy_chunks.csv"

CHUNK_SIZE = 40
OVERLAP = 10
STEP_SIZE = CHUNK_SIZE - OVERLAP

EXPECTED_CHUNKS = {
    "refund_policy.txt": 4,
    "delivery_policy.txt": 3,
    "escalation_policy.txt": 3,
    "payment_policy.txt": 3,
    "privacy_policy.txt": 2,
}


def chunk_document(text, source):
    """Split a document into 40-word chunks with 10-word overlap."""

    words = text.split()
    chunks = []

    start = 0
    chunk_number = 1

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))

        chunk_text = " ".join(words[start:end])

        chunks.append({
            "source": source,
            "chunk_id": f"{Path(source).stem}_{chunk_number}",
            "text": chunk_text,
            "start_word": start,
            "end_word": end,
            "word_count": end - start,
        })

        # Stop after the first chunk that reaches the end.
        if end >= len(words):
            break

        start += STEP_SIZE
        chunk_number += 1

    return chunks


def load_and_chunk_policies():
    """Load all policy text files and generate chunks."""

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {POLICY_DIR}"
        )

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    if len(policy_files) != 5:
        raise ValueError(
            f"Expected 5 policy files, found {len(policy_files)}"
        )

    all_chunks = []

    for policy_file in policy_files:
        text = policy_file.read_text(encoding="utf-8")

        document_chunks = chunk_document(
            text=text,
            source=policy_file.name,
        )

        all_chunks.extend(document_chunks)

    return pd.DataFrame(all_chunks)


def main():
    chunks_df = load_and_chunk_policies()

    counts = (
        chunks_df.groupby("source")
        .size()
        .to_dict()
    )

    print("\n========== PART D TASK 1 ==========")

    print("Policy documents:", len(EXPECTED_CHUNKS))
    print("Total chunks:", len(chunks_df))

    print("\nChunks per document:")

    for filename, expected_count in EXPECTED_CHUNKS.items():
        actual_count = counts.get(filename, 0)

        print(
            f"{filename}: {actual_count} "
            f"(expected {expected_count})"
        )

        assert actual_count == expected_count, (
            f"Unexpected chunk count for {filename}: "
            f"{actual_count}"
        )

    assert len(chunks_df) == 15

    # Validate chunk sizes and source metadata.
    assert chunks_df["source"].notna().all()
    assert chunks_df["text"].notna().all()
    assert chunks_df["word_count"].between(
        1, CHUNK_SIZE
    ).all()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSample chunks:")
    for _, row in chunks_df.head(3).iterrows():
        print(f"\nSource: {row['source']}")
        print(f"Chunk ID: {row['chunk_id']}")
        print(f"Words: {row['word_count']}")
        print(f"Text: {row['text']}")

    print("\nSaved:", OUTPUT_FILE)
    print("\nALL TASK 1 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
