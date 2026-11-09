
"""Part D — TF-IDF policy retrieval using cosine similarity."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    ROOT / "outputs" / "part_d" / "policy_chunks.csv"
)

TOP_K = 3


class PolicyRetriever:
    """Retrieve relevant policy chunks using TF-IDF and cosine similarity."""

    def __init__(self, chunks_file=CHUNKS_FILE):
        self.chunks_file = Path(chunks_file)
        self.chunks = None
        self.vectorizer = None
        self.chunk_vectors = None

    def fit(self):
        """Load policy chunks and fit the TF-IDF vectorizer."""

        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Policy chunks not found: {self.chunks_file}. "
                "Run Part D Task 1 first."
            )

        self.chunks = pd.read_csv(
            self.chunks_file,
            keep_default_na=False,
        )

        required_columns = {"source", "text"}
        missing = required_columns - set(self.chunks.columns)

        if missing:
            raise ValueError(
                f"Missing required chunk columns: {missing}"
            )

        if self.chunks.empty:
            raise ValueError("Policy chunk file is empty.")

        self.vectorizer = TfidfVectorizer()

        self.chunk_vectors = self.vectorizer.fit_transform(
            self.chunks["text"].astype(str)
        )

        return self

    def retrieve(self, query, k=TOP_K):
        """
        Return the top-k policy chunks for a query.

        Each result contains:
            source
            score
            text
        """

        if self.vectorizer is None or self.chunk_vectors is None:
            raise RuntimeError(
                "Retriever is not fitted. Call fit() first."
            )

        if not str(query).strip():
            raise ValueError("Query cannot be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.chunk_vectors,
        ).flatten()

        ranked_indices = similarities.argsort()[::-1][:k]

        results = []

        for index in ranked_indices:
            row = self.chunks.iloc[index]

            results.append({
                "source": row["source"],
                "score": float(similarities[index]),
                "text": row["text"],
            })

        return results


def retrieve(query, k=TOP_K):
    """Convenience function for retrieving policy chunks."""

    retriever = PolicyRetriever()
    retriever.fit()

    return retriever.retrieve(
        query=query,
        k=k,
    )


def main():
    print("\n========== PART D TASK 2 ==========")

    retriever = PolicyRetriever()
    retriever.fit()

    print("Policy chunks loaded:", len(retriever.chunks))
    print("TF-IDF vocabulary size:", len(
        retriever.vectorizer.vocabulary_
    ))
    print("TF-IDF matrix shape:", retriever.chunk_vectors.shape)

    query = "refund policy late delivery"

    results = retriever.retrieve(query, k=3)

    print("\nQuery:", query)
    print("\nTop 3 retrieved chunks:")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Source:", result["source"])
        print("Similarity score:", round(result["score"], 4))
        print("Text:", result["text"])

    # Acceptance check: all three top results should come
    # from the refund policy document.
    assert len(results) == 3
    assert all(
        result["source"] == "refund_policy.txt"
        for result in results
    )

    assert all(
        0.0 <= result["score"] <= 1.0
        for result in results
    )

    print("\nTop 3 refund retrieval check: PASSED")
    print("\nALL TASK 2 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
