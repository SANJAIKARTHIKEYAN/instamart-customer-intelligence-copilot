
"""Part D — RAG answer generation and evaluation."""

from pathlib import Path
import pandas as pd

import mock_llm

from src.part_d.retrieval import PolicyRetriever
from src.part_d.document_chunking import POLICY_DIR, CHUNK_SIZE

ROOT = Path(__file__).resolve().parents[2]

EVALUATION_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_d"

OVERLAP_CHUNKS_FILE = OUTPUT_DIR / "policy_chunks.csv"
NO_OVERLAP_CHUNKS_FILE = OUTPUT_DIR / "policy_chunks_no_overlap.csv"

FALLBACK_ANSWER = (
    "I could not find that information in the policy documents."
)

SYSTEM_PROMPT = """
You are a POLICY ASSISTANT.

Answer the user's question using only the supplied policy context.

Rules:
1. Use only information explicitly supported by the supplied context.
2. Do not invent policy rules, amounts, timelines, or exceptions.
3. If the answer is not available in the context, begin your
   response with: I could not find
4. Keep the answer clear and concise.
"""


class PolicyRAG:
    """Retrieve policy context and generate grounded answers."""

    def __init__(self, chunks_file=OVERLAP_CHUNKS_FILE):
        self.retriever = PolicyRetriever(chunks_file)
        self.retriever.fit()

    def build_context(self, retrieved_chunks):
        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"[{chunk['source']}]\n{chunk['text']}"
            )

        return "\n\n".join(context_parts)

    def answer(self, question):
        if not str(question).strip():
            raise ValueError("Question cannot be empty.")

        retrieved_chunks = self.retriever.retrieve(
            question,
            k=3,
        )

        context = self.build_context(retrieved_chunks)

        user_message = (
            "CONTEXT:\n"
            + context
            + "\nQUESTION:\n"
            + question
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        response = mock_llm.chat(messages)

        return {
            "answer": response.choices[0].message.content,
            "sources": [
                chunk["source"]
                for chunk in retrieved_chunks
            ],
            "retrieved_chunks": retrieved_chunks,
            "tokens_used": int(response.usage.total_tokens),
        }


def create_chunks_without_overlap():
    """Create 40-word chunks without overlap."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    for policy_file in policy_files:
        text = policy_file.read_text(encoding="utf-8")
        words = text.split()

        start = 0
        chunk_number = 1

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            rows.append({
                "source": policy_file.name,
                "chunk_id": f"{policy_file.stem}_{chunk_number}",
                "text": " ".join(words[start:end]),
                "start_word": start,
                "end_word": end,
                "word_count": end - start,
            })

            start = end
            chunk_number += 1

    no_overlap_df = pd.DataFrame(rows)

    no_overlap_df.to_csv(
        NO_OVERLAP_CHUNKS_FILE,
        index=False,
    )

    return no_overlap_df


def evaluate_rag(chunks_file, evaluation_df, mode):
    """Evaluate document retrieval, fact grounding and fallback."""

    assistant = PolicyRAG(chunks_file)
    rows = []

    for _, item in evaluation_df.iterrows():
        question = item["question"]
        expected_doc = str(item["expected_doc"]).strip()

        expected_fact = (
            ""
            if pd.isna(item["expected_fact"])
            else str(item["expected_fact"]).strip()
        )

        result = assistant.answer(question)

        retrieved_sources = [
            chunk["source"]
            for chunk in result["retrieved_chunks"]
        ]

        answer = result["answer"]

        unsupported = expected_doc.upper() == "NONE"

        document_found = (
            expected_doc in retrieved_sources
            if not unsupported
            else False
        )

        # Case-sensitive substring check, as required.
        fact_found = (
            expected_fact in answer
            if not unsupported and expected_fact
            else False
        )

        # The fallback must BEGIN with this phrase.
        fallback_used = answer.strip().lower().startswith(
            "i could not find"
        )

        rows.append({
            "mode": mode,
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answer": answer,
            "retrieved_sources": ";".join(retrieved_sources),
            "document_found": document_found,
            "fact_found": fact_found,
            "unsupported_question": unsupported,
            "fallback_used": fallback_used,
            "tokens_used": result["tokens_used"],
        })

    return pd.DataFrame(rows)


def summarize_evaluation(results, mode):
    answerable = results[
        ~results["unsupported_question"]
    ]

    unsupported = results[
        results["unsupported_question"]
    ]

    summary = {
        "mode": mode,
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "unsupported_questions": len(unsupported),
        "document_hits": int(answerable["document_found"].sum()),
        "fact_hits": int(answerable["fact_found"].sum()),
        "fallback_hits": int(unsupported["fallback_used"].sum()),
    }

    summary["document_accuracy"] = (
        summary["document_hits"] / len(answerable)
    )

    summary["fact_accuracy"] = (
        summary["fact_hits"] / len(answerable)
    )

    summary["fallback_accuracy"] = (
        summary["fallback_hits"] / len(unsupported)
    )

    return summary


def main():
    print("\n========== PART D TASK 4 ==========")

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file missing: {EVALUATION_FILE}"
        )

    evaluation_df = pd.read_csv(
        EVALUATION_FILE,
        keep_default_na=False,
    )

    assert len(evaluation_df) == 14

    answerable_count = (
        evaluation_df["expected_doc"] != "NONE"
    ).sum()

    unsupported_count = (
        evaluation_df["expected_doc"] == "NONE"
    ).sum()

    assert answerable_count == 12
    assert unsupported_count == 2

    no_overlap_df = create_chunks_without_overlap()

    print("Benchmark questions:", len(evaluation_df))
    print("Answerable:", answerable_count)
    print("Unsupported:", unsupported_count)
    print("No-overlap chunks:", len(no_overlap_df))

    # Evaluate with 10-word overlap.
    with_overlap = evaluate_rag(
        OVERLAP_CHUNKS_FILE,
        evaluation_df,
        "with_overlap",
    )

    # Evaluate with zero overlap.
    without_overlap = evaluate_rag(
        NO_OVERLAP_CHUNKS_FILE,
        evaluation_df,
        "without_overlap",
    )

    with_summary = summarize_evaluation(
        with_overlap,
        "with_overlap",
    )

    without_summary = summarize_evaluation(
        without_overlap,
        "without_overlap",
    )

    summary_df = pd.DataFrame([
        with_summary,
        without_summary,
    ])

    all_results = pd.concat(
        [with_overlap, without_overlap],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results.to_csv(
        OUTPUT_DIR / "rag_evaluation_details.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR / "rag_evaluation_summary.csv",
        index=False,
    )

    print("\n========== WITH 10-WORD OVERLAP ==========")
    for key, value in with_summary.items():
        print(f"{key}: {value}")

    print("\n========== WITHOUT OVERLAP ==========")
    for key, value in without_summary.items():
        print(f"{key}: {value}")

    print("\n========== COMPARISON ==========")
    print(summary_df.to_string(index=False))

    # Identify answerable questions that failed fact grounding.
    failures = without_overlap[
        (~without_overlap["unsupported_question"])
        & (~without_overlap["fact_found"])
    ]

    print("\n========== NO-OVERLAP FAILURES ==========")

    for _, failure in failures.iterrows():
        print("\nQuestion:", failure["question"])
        print("Expected fact:", failure["expected_fact"])
        print("Generated answer:", failure["answer"])
        print("Retrieved sources:", failure["retrieved_sources"])

    # Acceptance checks.
    assert with_summary["document_hits"] == 12
    assert with_summary["fact_hits"] == 12
    assert with_summary["fallback_hits"] == 2

    assert without_summary["document_hits"] == 12
    assert without_summary["fact_hits"] < 12
    assert without_summary["fallback_hits"] == 2

    assert len(failures) >= 1

    print("\nSaved detailed evaluation:")
    print(OUTPUT_DIR / "rag_evaluation_details.csv")

    print("\nSaved summary:")
    print(OUTPUT_DIR / "rag_evaluation_summary.csv")

    print("\nALL TASK 4 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
